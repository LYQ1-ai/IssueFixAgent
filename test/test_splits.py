"""``mcts/splits.py``（PLAN §1.3 三层划分：60% 生成池 + repo 级 80/10/10 + instance 映射）测试。

核心验收点：划分可复现（固定 seed）；**同 repo 不跨 split**（D6 防泄漏）；
prm_split 只出现在生成池 repo 上。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcts.instances import Gold, Instance  # noqa: E402
from mcts.splits import SPLIT_COLUMNS, build_splits, make_splits  # noqa: E402


def make_instances(n_repos: int, per_repo: int = 2) -> list[Instance]:
    insts = []
    for r in range(n_repos):
        repo = f"swesmith/repo{r:02d}.deadbeef"
        for i in range(per_repo):
            insts.append(
                Instance(
                    instance_id=f"repo{r:02d}.deadbeef.func_basic__t{i}",
                    repo=repo,
                    owner="swesmith",
                    name=f"repo{r:02d}",
                    commit8="deadbeef",
                    base_commit=None,
                    problem_statement=f"Fix bug {r}-{i}",
                    patch="diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
                    use_patch=True,
                    gold=Gold(frozenset(["x.py"]), frozenset(), frozenset()),
                    source="train",
                )
            )
    return insts


class TestMakeSplits:
    def test_deterministic(self):
        insts = make_instances(20, 3)
        df1, rep1 = make_splits(insts, seed=42)
        df2, rep2 = make_splits(insts, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
        assert rep1 == rep2

    def test_seed_changes_assignment(self):
        insts = make_instances(20, 2)
        df1, _ = make_splits(insts, seed=1)
        df2, _ = make_splits(insts, seed=2)
        assert not df1["prm_split"].equals(df2["prm_split"])

    def test_no_repo_leakage(self):
        insts = make_instances(20, 3)
        df, _ = make_splits(insts, seed=42)
        grouped = df.groupby("repo")["prm_split"].nunique()
        assert (grouped <= 1).all()  # 同一 repo 的所有实例落入同一 split（或同为 None）

    def test_splits_disjoint(self):
        insts = make_instances(20, 2)
        df, _ = make_splits(insts, seed=42)
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            assert not set(df[df["prm_split"] == a]["instance_id"]) & set(
                df[df["prm_split"] == b]["instance_id"]
            )

    def test_gen_pool_ratio(self):
        insts = make_instances(20, 2)
        df, rep = make_splits(insts, seed=42, gen_pool_ratio=0.6)
        n_gen_repos = df[df["gen_pool"]]["repo"].nunique()
        assert n_gen_repos == 12  # round(20 * 0.6)
        assert rep["gen_pool_repos"] == 12
        assert rep["gen_pool_instances"] == 24
        assert rep["excluded_instances"] == 40 - 24

    def test_prm_split_only_within_gen_pool(self):
        insts = make_instances(20, 2)
        df, _ = make_splits(insts, seed=42)
        assert (df["prm_split"].notna() == df["gen_pool"]).all()

    def test_prm_ratios_approx(self):
        insts = make_instances(50, 1)
        df, rep = make_splits(insts, seed=42)
        pool = rep["gen_pool_repos"]
        assert pool == 30
        # 80/10/10 近似（30 -> 24/3/3）
        assert rep["prm_train_repos"] == 24
        assert rep["prm_dev_repos"] == 3
        assert rep["prm_test_repos"] == 3
        assert rep["prm_train_instances"] + rep["prm_dev_instances"] + rep["prm_test_instances"] == pool

    def test_dev_test_at_least_one_when_possible(self):
        insts = make_instances(10, 1)  # pool = 6 -> 4/1/1
        _, rep = make_splits(insts, seed=42)
        assert rep["prm_dev_repos"] >= 1 and rep["prm_test_repos"] >= 1

    def test_single_repo_all_train(self):
        insts = make_instances(1, 3)
        df, rep = make_splits(insts, seed=42)
        assert rep["gen_pool_repos"] == 1
        assert set(df["prm_split"].dropna().unique()) == {"train"}

    def test_empty_instances(self):
        df, rep = make_splits([], seed=42)
        assert df.empty
        assert rep["n_instances"] == 0 and rep["n_repos"] == 0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            make_splits(make_instances(5), gen_pool_ratio=1.5)
        with pytest.raises(ValueError):
            make_splits(make_instances(5), prm_dev_ratio=-0.1)


class TestBuildSplits:
    def test_build_from_instances_parquet(self, tmp_path):
        # 先造一个 instances.parquet（复用 instances 模块的 to_row 格式）
        insts = make_instances(10, 2)
        df = pd.DataFrame([i.to_row() for i in insts])
        inst_path = tmp_path / "instances.parquet"
        df.to_parquet(inst_path, index=False)
        out = tmp_path / "out"
        report = build_splits(inst_path, output_dir=out, cfg={"split": {"seed": 42}})
        assert report["n_instances"] == 20
        assert report["n_repos"] == 10
        assert report["gen_pool_repos"] == 6
        splits_path = out / "splits.parquet"
        assert splits_path.is_file()
        assert (out / "splits_report.json").is_file()
        assert (out / "splits_report.md").is_file()
        sdf = pd.read_parquet(splits_path)
        assert list(sdf.columns) == SPLIT_COLUMNS
        assert len(sdf) == 20
        # 同 repo 不跨 split
        assert (sdf.groupby("repo")["prm_split"].nunique() <= 1).all()

    def test_missing_instances_parquet_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_splits(tmp_path / "nope.parquet", output_dir=tmp_path / "out")
