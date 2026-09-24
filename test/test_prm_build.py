# SPDX-License-Identifier: BSD-3-Clause

"""prm/build_dataset.py 单测（docs/prm_training_plan.md §10）。

场景对照 §10 test_prm_build 行：非 leaf 0.8→0.96 / 0.4→0.08、leaf 链 1,1,0 /
len=1 仅负样本、root 不产样本、去重优先级 node_mc>leaf_chain、min_rollouts
跳过、split 防泄漏断言、DB mtime/行数不变、manifest 字段齐全。
全部离线（迷你 DB；test/prm_fixtures.py）。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prm_fixtures import (  # noqa: E402
    _DDL,
    add_tree,
    make_builder_config,
    make_splits_parquet,
    make_step,
    make_steps,
)

from mcts.steps import Step, prefix_node_key  # noqa: E402

from prm.build_dataset import (  # noqa: E402
    DEFAULT_CONFIG,
    SCHEMA,
    PRMDatasetBuilder,
    deep_merge,
    load_config,
)
from prm.prompts import template_hash  # noqa: E402


# ---------------------------------------------------------------------------
# fixture：一棵结构丰富的树 + 若干应被过滤的树
# ---------------------------------------------------------------------------

def key_of(steps: list[dict]) -> str:
    return prefix_node_key([Step.from_json(s) for s in steps])


@pytest.fixture
def env(tmp_path: Path):
    """迷你 DB + splits；返回 (cfg, 依赖的 key 字典)。"""
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(_DDL)

    s4 = make_steps(4)   # 非leaf节点前缀（mc=0.8 = 4/5）
    s3 = [make_step(i, command=f"probe_{i}") for i in (1, 2, 3)]  # 槽位不足节点（内容不同）
    s1 = make_steps(1)   # mc=1.0 节点前缀
    s2 = make_steps(2)   # mc=0.4 节点前缀（2/5）
    keys = {"s4": key_of(s4), "s3": key_of(s3), "s1": key_of(s1), "s2": key_of(s2)}

    # instA（train）：root(3/5) + 非leaf 0.8 + 非leaf 0.4 + mc=1.0 + 槽位不足(<3)
    add_tree(conn, "instA", root_correct=[1, 1, 1, 0, 0], nodes={
        keys["s4"]: {"prefix": s4, "corrects": [1, 1, 1, 1, 0], "mc": 0.8,
                     "in_pool": 1, "visits": 2},
        keys["s2"]: {"prefix": s2, "corrects": [1, 0, 1, 0, 0], "mc": 0.4},
        keys["s3"]: {"prefix": s3, "corrects": [0, 0]},  # 2 条 rollouts < min_rollouts → skip
        keys["s1"]: {"prefix": s1, "corrects": [1, 1, 1, 1, 1], "mc": 1.0},
    })
    # instA 的 leaf（3 步，全错；leaf_* 独立内容 → 与其它节点无前缀撞键）
    leaf3 = [make_step(i, command=f"leaf_{i}") for i in (1, 2, 3)]
    keys["leaf3"] = key_of(leaf3)
    add_tree_leaf(conn, "instA", leaf3)
    # instB（test）：只有 root 全错 → 不产样本
    add_tree(conn, "instB", root_correct=[0, 0, 0, 0, 0])
    # instC：done 但不在 splits → skip；instD：failed → skip
    add_tree(conn, "instC", root_correct=[1, 1, 1, 1, 1])
    add_tree(conn, "instD", status="failed", root_correct=[1, 1, 1, 1, 1])
    conn.close()

    splits = make_splits_parquet(tmp_path / "splits.parquet", [
        ("instA", "repoA", "train"), ("instB", "repoB", "test"),
        ("instD", "repoD", "train"),  # failed 树不应进数据
    ])
    cfg = make_builder_config(db, splits, tmp_path / "out", min_rollouts=3, leaf_chain=True)
    return cfg, keys


def add_tree_leaf(conn: sqlite3.Connection, instance_id: str, prefix: list[dict]) -> None:
    """向已有树追加一个 leaf 节点（5 条全错 rollout）。"""
    from prm_fixtures import _add_rollout
    key = key_of(prefix)
    conn.execute(
        "INSERT INTO nodes (instance_id, node_key, status, prefix_json, mc_score, visits, in_pool) "
        "VALUES (?, ?, 'ready', ?, 0.0, 0, 0)",
        (instance_id, key, json.dumps(prefix)))
    for idx in range(5):
        _add_rollout(conn, instance_id, key, idx, 0, n_steps=len(prefix) + 2)
    conn.commit()


def _expected_w_neg(manifest: dict) -> float:
    return min(manifest["class_weights"]["cap"],
               manifest["class_weights"]["n_pos"] / max(1, manifest["class_weights"]["n_neg"]))


@pytest.fixture
def built(env):
    cfg, keys = env
    builder = PRMDatasetBuilder(cfg)
    manifest = builder.build()
    return builder, manifest, keys


def read_split(out_dir: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(out_dir / f"{split}.parquet").to_pylist()


# ---------------------------------------------------------------------------
# 样本派生规则（§5.1）
# ---------------------------------------------------------------------------

class TestSampleDerivation:
    def test_non_leaf_mc08_one_sample_label_096(self, built):
        _, manifest, keys = built
        rows = read_split(Path(built[0].out_dir), "train")
        row = next(r for r in rows if r["node_key"] == keys["s4"])
        assert row["label_source"] == "node_mc"
        assert row["label"] == pytest.approx(0.96)   # 0.8·1 + 0.2·0.8
        assert row["label_binary"] == 1
        assert row["label_soft"] == pytest.approx(0.8)
        assert row["step_index"] == row["step_count"] == 4
        assert row["mc_score"] == pytest.approx(0.8)
        assert row["n_rollouts"] == 5

    def test_non_leaf_mc04_label_008(self, built):
        _, _, keys = built
        rows = read_split(Path(built[0].out_dir), "train")
        row = next(r for r in rows if r["node_key"] == keys["s2"])
        assert row["label"] == pytest.approx(0.08)   # 0.8·0 + 0.2·0.4
        assert row["label_binary"] == 0

    def test_mc10_node_positive_label(self, built):
        _, _, keys = built
        rows = read_split(Path(built[0].out_dir), "train")
        row = next(r for r in rows if r["node_key"] == keys["s1"])
        assert row["label"] == pytest.approx(1.0) and row["label_binary"] == 1

    def test_leaf_len3_chain_labels_110(self, built):
        """leaf len=3 → 3 条样本，标签 1,1,0。"""
        _, _, keys = built
        rows = read_split(Path(built[0].out_dir), "train")
        chain = sorted((r for r in rows if r["node_key"] == keys["leaf3"]),
                       key=lambda r: r["step_index"])
        assert [r["label"] for r in chain] == [1.0, 1.0, 0.0]
        assert all(r["label_source"] == "leaf_chain" for r in chain)
        assert chain[-1]["label_binary"] == 0 and chain[0]["label_binary"] == 1
        # 第 i 条样本的 step_count 均为 leaf 前缀长度
        assert all(r["step_count"] == 3 for r in chain)

    def test_root_produces_no_sample(self, built):
        builder, manifest, _ = built
        all_rows = []
        for split in ("train", "dev", "test"):
            all_rows += read_split(Path(builder.out_dir), split)
        assert all(r["node_key"] != "root" for r in all_rows)

    def test_min_rollouts_skipped(self, built):
        """2 条 rollouts 的节点（< min_rollouts=3）跳过且计入审计。"""
        builder, manifest, keys = built
        rows = read_split(Path(builder.out_dir), "train")
        # s3 槽位节点被 leaf3 覆盖删除，这里验证审计计数正确即可
        assert manifest["skipped"]["nodes"]["min_rollouts"] == 1

    def test_mc_cache_mismatch_skipped(self, tmp_path: Path):
        """重算 MC 与缓存差 >1e-6 → 节点跳过并计入审计（§2.1-4）。"""
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(_DDL)
        s2 = make_steps(2)
        k2 = key_of(s2)
        add_tree(conn, "instA", root_correct=[1, 1, 1, 1, 1], nodes={
            k2: {"prefix": s2, "corrects": [1, 1, 1, 1, 0], "mc": 0.5},  # 重算=0.8 ≠ 缓存 0.5
        })
        conn.close()
        splits = make_splits_parquet(tmp_path / "splits.parquet", [("instA", "repoA", "train")])
        cfg = make_builder_config(db, splits, tmp_path / "out", leaf_chain=True)
        builder = PRMDatasetBuilder(cfg)
        with pytest.raises(RuntimeError, match="为空"):
            builder.build()  # 全部样本被跳过 → 快速失败
        assert builder.skip_stats["mc_mismatch"] == 1

    def test_done_and_split_filters(self, built):
        builder, manifest, _ = built
        assert manifest["skipped"]["prepare"]["not_done_tree"] == 1   # instD failed
        assert manifest["skipped"]["prepare"]["no_split"] == 1        # instC 无 split
        # instB（root 全错）在 test split 但无样本
        assert manifest["counts"]["test"] == 0


# ---------------------------------------------------------------------------
# 去重（§5.1 例子）
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_node_mc_beats_leaf_chain(self, tmp_path: Path):
        """leaf 链回填与真实节点撞键 → 保留 node_mc 样本（§5.1 probe①/② 例子）。"""
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(_DDL)
        s6 = make_steps(6)   # 真实节点前缀（probe①，mc=0.8）
        s9 = make_steps(9)   # leaf 前缀（probe②，mc=0）
        k6, k9 = key_of(s6), key_of(s9)
        add_tree(conn, "instA", root_correct=[1, 0, 0, 0, 0], nodes={
            k6: {"prefix": s6, "corrects": [1, 1, 1, 1, 0], "mc": 0.8, "in_pool": 1},
            k9: {"prefix": s9, "corrects": [0, 0, 0, 0, 0], "mc": 0.0},
        })
        conn.close()
        splits = make_splits_parquet(tmp_path / "splits.parquet", [("instA", "repoA", "train")])
        cfg = make_builder_config(db, splits, tmp_path / "out", leaf_chain=True)
        builder = PRMDatasetBuilder(cfg)
        manifest = builder.build()
        rows = read_split(Path(builder.out_dir), "train")
        # 撞键位置：leaf 链 i=6 的前缀 == probe① 的 6 步前缀
        dup = [r for r in rows if r["step_index"] == 6 and r["step_count"] == 9]
        real = [r for r in rows if r["node_key"] == k6]
        assert not dup, "leaf 链 i=6 应被真实节点样本压过（去重键相同）"
        assert len(real) == 1 and real[0]["label_source"] == "node_mc"
        assert real[0]["label"] == pytest.approx(0.96)  # 保留 0.96 那条（§5.1）
        # 去重统计：9 条链 + 1 条真实 - 1 撞键 = 9
        assert manifest["dedupe"]["collisions"] == 1
        assert manifest["dedupe"]["before"] == 10 and manifest["dedupe"]["after"] == 9

    def test_no_leaf_chain_mode(self, tmp_path: Path):
        """--no-leaf-chain：leaf 只产 1 条 node_mc 负样本。"""
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(_DDL)
        s3 = make_steps(3)
        k3 = key_of(s3)
        add_tree(conn, "instA", root_correct=[0, 0, 0, 0, 0], nodes={
            k3: {"prefix": s3, "corrects": [0, 0, 0, 0, 0], "mc": 0.0},
        })
        conn.close()
        splits = make_splits_parquet(tmp_path / "splits.parquet", [("instA", "repoA", "train")])
        cfg = make_builder_config(db, splits, tmp_path / "out", leaf_chain=False)
        builder = PRMDatasetBuilder(cfg)
        builder.build()
        rows = read_split(Path(builder.out_dir), "train")
        assert len(rows) == 1
        assert rows[0]["label_source"] == "node_mc" and rows[0]["label"] == pytest.approx(0.0)
        assert rows[0]["step_index"] == 3

    def test_leaf_len1_only_negative(self, tmp_path: Path):
        """leaf len=1 → 仅一条负样本 [0.0]。"""
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(_DDL)
        s1 = make_steps(1)
        k1 = key_of(s1)
        add_tree(conn, "instA", root_correct=[0, 0, 0, 0, 0], nodes={
            k1: {"prefix": s1, "corrects": [0, 0, 0, 0, 0], "mc": 0.0},
        })
        conn.close()
        splits = make_splits_parquet(tmp_path / "splits.parquet", [("instA", "repoA", "train")])
        cfg = make_builder_config(db, splits, tmp_path / "out", leaf_chain=True)
        builder = PRMDatasetBuilder(cfg)
        builder.build()
        rows = read_split(Path(builder.out_dir), "train")
        assert [r["label"] for r in rows] == [0.0]
        assert rows[0]["step_index"] == 1 and rows[0]["label_source"] == "leaf_chain"


# ---------------------------------------------------------------------------
# 防泄漏（§5.6）/ 只读约束（§11）/ manifest（§5.4）
# ---------------------------------------------------------------------------

class TestSafetyAndManifest:
    def test_split_leakage_instance_raises(self):
        df = pd.DataFrame([
            {"instance_id": "a", "repo": "r1", "prm_split": "train"},
            {"instance_id": "a", "repo": "r1", "prm_split": "dev"},   # instance 跨 split
        ])
        with pytest.raises(ValueError, match="instance"):
            PRMDatasetBuilder._assert_no_leakage(df)

    def test_split_leakage_repo_raises(self):
        df = pd.DataFrame([
            {"instance_id": "a", "repo": "r1", "prm_split": "train"},
            {"instance_id": "b", "repo": "r1", "prm_split": "test"},  # repo 跨 split
        ])
        with pytest.raises(ValueError, match="repo"):
            PRMDatasetBuilder._assert_no_leakage(df)

    def test_db_untouched(self, built):
        """DB mtime/行数不变（§11 M4.1 验收）。"""
        builder, manifest, _ = built
        assert manifest["db_unchanged"] is True
        assert manifest["db_fingerprint_after"] == manifest["db_fingerprint_before"]
        db = Path(builder.cfg["db"]["path"])
        assert db.stat().st_mtime == builder._db_mtime_before

    def test_manifest_fields_complete(self, built):
        _, manifest, _ = built
        for field in ("created_at", "template_version", "template_hash", "tokenizer",
                      "tokenizer_backend", "rendered_tokens_approx", "db_path",
                      "db_fingerprint_before", "db_fingerprint_after", "db_unchanged",
                      "splits_path", "build", "counts", "label_distribution", "dedupe",
                      "skipped", "class_weights", "length_stats", "files"):
            assert field in manifest
        assert manifest["template_hash"] == template_hash()
        assert manifest["class_weights"]["w_pos"] == 1.0
        # §5.2：w_neg = min(cap, n_pos/n_neg)（train split 统计）
        assert manifest["class_weights"]["w_neg"] == pytest.approx(_expected_w_neg(manifest))

    def test_class_weights_cap_pure_function(self):
        """cap 行为单独验证：≈5.56:1 触顶 4（§5.2 预期）。"""
        from prm.labeling import class_weights
        assert class_weights([1] * 9 + [0]) == (1.0, 4.0)          # 9:1 → 触顶
        assert class_weights([1, 1, 1, 0, 0, 0]) == (1.0, 1.0)     # 1:1
        assert class_weights([0, 0]) == (1.0, 0.0)                 # 无正类：公式值 0
        assert class_weights([]) == (1.0, 1.0)                     # 空 split

    def test_parquet_schema_and_messages_struct(self, built):
        import pyarrow.parquet as pq
        builder, _, keys = built
        table = pq.read_table(Path(builder.out_dir) / "train.parquet")
        assert table.schema == SCHEMA  # §5.4 schema 逐字段一致（含类型）
        rows = table.to_pylist()
        row = next(r for r in rows if r["node_key"] == keys["s4"])
        roles = [m["role"] for m in row["messages"]]
        assert roles[0] == "system" and roles[1] == "user" and roles[-1] == "user"
        assistant = row["messages"][2]
        assert assistant["tool_calls"][0]["function"]["name"] == "bash"
        args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
        assert args == {"command": "ls"}  # arguments 以 JSON 字符串存储
        assert row["rendered_tokens"] > 0
        assert row["sample_id"] == f"{row['instance_id']}::{row['node_key']}::{row['step_index']}"

    def test_length_report_and_outputs(self, built):
        builder, manifest, _ = built
        out = Path(builder.out_dir)
        assert (out / "manifest.json").exists()
        assert (out / "build_report.md").exists()
        assert (out / "length_report.md").exists()
        ls = manifest["length_stats"]
        assert set(ls["percentiles"]) == {50, 75, 90, 95, 99}
        assert ls["approx"] is True  # 无 tokenizer → 近似并在报告标注
        assert "3.5 chars/token 近似" in (out / "length_report.md").read_text(encoding="utf-8")
        # U6：人工抽检报告（§13.2 步骤 3 留档）
        spot = out / "spot_check.md"
        assert spot.exists()
        txt = spot.read_text(encoding="utf-8")
        assert "样本抽检报告" in txt and "核对要点" in txt and "```text" in txt

    def test_cli_overwrite_guard(self, env):
        """已存在输出且未加 --overwrite → SystemExit（不重建）。"""
        from prm.build_dataset import main
        cfg, _ = env
        PRMDatasetBuilder(cfg).build()
        argv = ["--db", cfg["db"]["path"], "--splits", cfg["splits"]["path"],
                "--output", cfg["output"]["dir"], "--min-rollouts", "3"]
        with pytest.raises(SystemExit):
            main(argv)


# ---------------------------------------------------------------------------
# 并行路径与配置合并
# ---------------------------------------------------------------------------

class TestWorkersAndConfig:
    def test_workers2_equivalent(self, env, tmp_path):
        """workers=2（多进程）与 workers=1 产出等价（除时间戳/报告外计数一致）。"""
        cfg, _ = env
        single = PRMDatasetBuilder(deep_merge(cfg, {"build": {"workers": 1}}))
        m1 = single.build()
        cfg2 = deep_merge(cfg, {"build": {"workers": 2},
                                "output": {"dir": str(tmp_path / "out2")}})
        dual = PRMDatasetBuilder(cfg2)
        m2 = dual.build()
        assert m1["counts"] == m2["counts"]
        assert m1["dedupe"]["after"] == m2["dedupe"]["after"]

    def test_deep_merge_and_cli_config(self):
        cfg = deep_merge(DEFAULT_CONFIG, {"build": {"min_rollouts": 5, "leaf_chain": None}})
        assert cfg["build"]["min_rollouts"] == 5
        assert cfg["build"]["leaf_chain"] is True  # None 不覆盖
        import argparse
        from prm.build_dataset import parse_args
        args = parse_args(["--db", "x.db", "--min-rollouts", "7", "--no-leaf-chain"])
        merged = deep_merge(cfg, {"db": {"path": args.db},
                                  "build": {"min_rollouts": args.min_rollouts,
                                            "leaf_chain": args.leaf_chain}})
        assert merged["db"]["path"] == "x.db"
        assert merged["build"]["min_rollouts"] == 7
        assert merged["build"]["leaf_chain"] is False
