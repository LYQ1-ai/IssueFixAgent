# SPDX-License-Identifier: BSD-3-Clause

"""prm/raw.py 单测（docs/prm_training_plan.md §10）。

全部离线：迷你 v5 DB（test/prm_fixtures.py 构造，口径与 docs/construction/01
DDL 一致），无 Docker / LLM / GPU / 网络。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prm_fixtures import _DDL, _HEAD, add_tree, make_splits_parquet  # noqa: E402

from prm import raw  # noqa: E402


@pytest.fixture
def mini_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(_DDL)
    add_tree(conn, "instA", root_correct=[1, 1, 1, 0, 0])       # done（root mc=0.6）
    add_tree(conn, "instB", status="failed", root_correct=[1])  # failed
    conn.commit()
    conn.close()
    return db


class TestOpenDbReadonly:
    def test_query_only_rejects_write(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        with pytest.raises(sqlite3.Error):
            conn.execute("DELETE FROM nodes")
        conn.close()

    def test_mode_ro_rejects_creation(self, tmp_path: Path):
        with pytest.raises(sqlite3.Error):
            raw.open_db_readonly(str(tmp_path / "nonexistent.db")).execute("SELECT 1")

    def test_reads_work(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
        conn.close()


class TestLoaders:
    def test_load_tree_status(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        status = raw.load_tree_status(conn)
        assert status == {"instA": "done", "instB": "failed"}

    def test_load_instance_heads_raw_json(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        heads = raw.load_instance_heads(conn)
        assert set(heads) == {"instA", "instB"}
        assert "issue_description" in heads["instA"]  # 原文（不解析）
        assert raw.head_user_content(heads["instA"]).startswith("Consider the issue")

    def test_head_user_content_rejects_bad_head(self):
        # 无 user 消息 → 报错（head 缺失说明树数据不完整，不应静默降级）
        with pytest.raises(ValueError):
            raw.head_user_content([{"role": "system", "content": "s"}])
        # user content 非字符串 → 报错
        with pytest.raises(ValueError):
            raw.head_user_content([{"role": "user", "content": None}])

    def test_mc_recompute_matches_manual(self, mini_db: Path):
        """mc 重算 SQL 结果与手算一致（§10）：root 5 条 correct=[1,1,0,0,0] → 0.6。"""
        conn = raw.open_db_readonly(str(mini_db))
        mc = raw.load_mc_table(conn)
        row = mc[(mc["instance_id"] == "instA") & (mc["node_key"] == "root")].iloc[0]
        assert row["mc"] == pytest.approx(3 / 5)
        assert row["n_rollouts"] == 5
        # 与 nodes.mc_score 缓存无关（fixture 中缓存为 NULL）
        assert {"instance_id", "node_key", "mc", "n_rollouts"} == set(mc.columns)

    def test_iter_nodes_columns_and_order(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        nodes = list(raw.iter_nodes(conn))
        keys = {(n["instance_id"], n["node_key"]) for n in nodes}
        assert ("instA", "root") in keys and ("instB", "root") in keys
        root = next(n for n in nodes if n["instance_id"] == "instA")
        assert root["prefix_json"] == "[]" and root["in_pool"] == 1
        # 只选需要的列（含审计用 mc_score 缓存）
        assert {"instance_id", "node_key", "prefix_json", "visits", "in_pool",
                "mc_score"} == set(nodes[0].keys())
        # 排序确定性
        order = [(n["instance_id"], n["node_key"]) for n in nodes]
        assert order == sorted(order)

    def test_load_root_rollouts(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        rollouts = raw.load_root_rollouts(conn, "instA")
        assert len(rollouts) == 5
        assert [r["rollout_idx"] for r in rollouts] == list(range(5))
        assert sum(r["correct"] for r in rollouts) == 3
        assert rollouts[0]["steps"], "result_json.steps 应已解析"
        assert all(r["exit_status"] == "Submitted" for r in rollouts)

    def test_load_splits_drops_null(self, tmp_path: Path):
        p = make_splits_parquet(tmp_path / "splits.parquet", [
            ("a", "repo1", "train"), ("b", "repo1", None), ("c", "repo2", "test")])
        df = raw.load_splits(str(p))
        assert sorted(df["instance_id"]) == ["a", "c"]
        assert set(df.columns) == {"instance_id", "repo", "prm_split"}

    def test_db_fingerprint(self, mini_db: Path):
        conn = raw.open_db_readonly(str(mini_db))
        fp = raw.db_fingerprint(conn)
        assert fp == {"tree_instances": 2, "nodes": 2, "rollouts": 6}
