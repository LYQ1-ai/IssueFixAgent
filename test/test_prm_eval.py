# SPDX-License-Identifier: BSD-3-Clause

"""prm/metrics.py + eval 汇总单测（docs/prm_training_plan.md §10 test_prm_eval 行）。

覆盖：指标与手算一致 + sklearn 对照（ROC-AUC/PR-AUC/LogLoss）、分桶正确、
best-of-N 选择器与 regret、bootstrap CI、report JSON/MD 字段一致。
prm.metrics 纯 numpy（任意环境可跑）；eval_prm 汇总类测试经 importorskip 门控。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prm import metrics as M


# ---------------------------------------------------------------------------
# 一致性：与手算 / sklearn 对照
# ---------------------------------------------------------------------------

class TestRocAuc:
    def test_matches_manual(self):
        y = [0, 0, 1, 1]
        s = [0.1, 0.4, 0.35, 0.8]
        # 手算：正类 (0.35, 0.8) vs 负类 (0.1, 0.4) → 3/4
        assert M.roc_auc(y, s) == pytest.approx(0.75)

    def test_perfect_and_inverted(self):
        assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
        assert M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_ties_average_rank(self):
        assert M.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)

    def test_single_class_nan(self):
        assert np.isnan(M.roc_auc([1, 1], [0.2, 0.8]))
        assert np.isnan(M.roc_auc([], []))

    def test_matches_sklearn(self):
        sklearn = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        s = np.clip(y * 0.3 + rng.normal(0.5, 0.2, 500), 0, 1)
        assert M.roc_auc(y, s) == pytest.approx(sklearn.roc_auc_score(y, s), abs=1e-10)


class TestPrAuc:
    def test_matches_sklearn(self):
        sklearn = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        s = rng.random(500) + y * 0.25
        assert M.pr_auc(y, s) == pytest.approx(
            sklearn.average_precision_score(y, s), abs=1e-10)

    def test_perfect(self):
        assert M.pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


class TestCalibrationMetrics:
    def test_brier_manual(self):
        y = [1, 0, 1]
        p = [0.8, 0.2, 0.4]
        assert M.brier_score(y, p) == pytest.approx((0.04 + 0.04 + 0.36) / 3)

    def test_ece_manual_perfectly_calibrated(self):
        # 单一非空桶内 mean p == pos_rate（0.5 桶放一对正负）→ ECE 0
        y = [1, 0]
        p = [0.5, 0.5]
        assert M.expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.0, abs=1e-12)

    def test_ece_manual_confidently_wrong(self):
        # 全部预测 1.0 但一半是负类 → ECE = 0.5
        y = [1, 0]
        p = [1.0, 1.0]
        assert M.expected_calibration_error(y, p) == pytest.approx(0.5)

    def test_classification_metrics_manual(self):
        m = M.classification_metrics([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1])
        assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["f1"] == pytest.approx(0.5)
        assert m["accuracy"] == pytest.approx(0.5)

    def test_log_loss_matches_sklearn(self):
        sklearn = pytest.importorskip("sklearn.metrics")
        y = [0, 1, 1, 0]
        p = [0.2, 0.8, 0.6, 0.4]
        assert M.log_loss(y, p) == pytest.approx(sklearn.log_loss(y, p), abs=1e-6)


class TestCorrelations:
    def test_pearson_manual(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert M.pearson(x, y) == pytest.approx(1.0)
        assert M.pearson(x, y[::-1]) == pytest.approx(-1.0)

    def test_spearman_rank_invariant(self):
        x = [10, 20, 30, 40]
        y = [1, 2, 3, 4]
        assert M.spearman(x, y) == pytest.approx(1.0)
        assert M.spearman(x, [100, 200, 300, 400]) == pytest.approx(1.0)

    def test_degenerate_nan(self):
        assert np.isnan(M.pearson([1, 1, 1], [1, 2, 3]))


class TestBootstrap:
    def test_ci_covers_point_estimate(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.6, 0.1, 200)
        lo, hi = M.bootstrap_ci(lambda x: float(np.mean(x)), a, n_boot=500)
        assert lo < a.mean() < hi

    def test_ci_of_difference_excludes_zero_for_strong_effect(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.8, 0.05, 100)   # 强选择器
        b = rng.normal(0.4, 0.05, 100)   # random
        lo, hi = M.bootstrap_ci(lambda x, y: float(np.mean(x - y)), a, b, n_boot=500)
        assert lo > 0  # 差值显著为正


class TestBuckets:
    def test_position_buckets(self):
        assert M.position_bucket(1, 9) == "1"
        assert M.position_bucket(5, 9) == "5"
        assert M.position_bucket(7, 9) == "6-10"
        assert M.position_bucket(12, 20) == "11-15"
        assert M.position_bucket(25, 30) == "21+"

    def test_relative_buckets(self):
        assert M.relative_position_bucket(1, 10) == "[0%,10%]"
        assert M.relative_position_bucket(5, 10) == "[40%,50%]"
        assert M.relative_position_bucket(10, 10) == "[90%,100%]"

    def test_length_buckets(self):
        assert M.context_length_bucket(1000) == "<=4K"
        assert M.context_length_bucket(9000) == "8-12K"
        assert M.context_length_bucket(20000) == ">16K"
        assert M.context_length_bucket(None) == "nan"

    def test_bucket_metrics(self):
        y = [0, 1, 1, 0]
        s = [0.2, 0.8, 0.9, 0.1]
        out = M.bucket_metrics(y, s, [[0, 1], [2, 3]])
        assert out[0]["n"] == 2 and out[0]["auc"] == pytest.approx(1.0)
        assert out[1]["n"] == 2 and out[1]["auc"] == pytest.approx(1.0)
        # 单一类别组 → AUC NaN（不可评估）
        single = M.bucket_metrics(y, s, [[1, 2]])
        assert np.isnan(single[0]["auc"])


# ---------------------------------------------------------------------------
# best-of-N（§9.2 选择器）
# ---------------------------------------------------------------------------

def _mk_rollouts(rewards: list[float], n_steps: list[int],
                 scores: list[float]) -> list[dict]:
    return [{"rollout_idx": i, "reward": r, "correct": int(r > 0), "n_steps": n,
             "agg_mean": s, "agg_min": s - 0.1, "agg_last": s}
            for i, (r, n, s) in enumerate(zip(rewards, n_steps, scores))]


class TestSelectors:
    def test_oracle_max_reward(self):
        cands = _mk_rollouts([0.0, 1.0, 0.5], [10, 12, 8], [0.1, 0.2, 0.9])
        assert M.select_rollout(cands, "oracle")["rollout_idx"] == 1

    def test_shortest(self):
        cands = _mk_rollouts([1.0, 0.0, 1.0], [10, 8, 9], [0.9, 0.1, 0.8])
        assert M.select_rollout(cands, "shortest")["rollout_idx"] == 1

    def test_prm_keys(self):
        cands = _mk_rollouts([0.0, 1.0], [10, 10], [0.2, 0.9])
        assert M.select_rollout(cands, "prm_mean")["rollout_idx"] == 1
        assert M.select_rollout(cands, "prm_min")["rollout_idx"] == 1
        assert M.select_rollout(cands, "prm_last")["rollout_idx"] == 1

    def test_random_with_seed(self):
        cands = _mk_rollouts([1.0, 0.0], [10, 10], [0.5, 0.5])
        r1 = M.select_rollout(cands, "random", rng=np.random.default_rng(42))
        r2 = M.select_rollout(cands, "random", rng=np.random.default_rng(42))
        assert r1["rollout_idx"] == r2["rollout_idx"]

    def test_empty_candidates(self):
        assert M.select_rollout([], "oracle") is None

    def test_unknown_selector_raises(self):
        with pytest.raises(ValueError):
            M.select_rollout(_mk_rollouts([1], [1], [1]), "bogus")


class TestBestOfMetrics:
    def test_oracle_is_upper_bound(self):
        per_instance = [{"instance_id": "a",
                         "rollouts": _mk_rollouts([0, 1, 0], [10, 12, 8],
                                                  [0.2, 0.3, 0.9])}]
        out = M.best_of_metrics(per_instance)
        assert out["oracle"]["selected_reward"] == pytest.approx(1.0)
        assert out["oracle"]["regret"] == pytest.approx(0.0)
        assert out["oracle"]["win_rate"] == pytest.approx(1.0)

    def test_perfect_prm_matches_oracle(self):
        """PRM 分数完全可分（好 rollout 分高）→ PRM-mean = oracle。"""
        per_instance = [{"instance_id": "a",
                         "rollouts": _mk_rollouts([0, 1], [10, 12], [0.2, 0.9])},
                        {"instance_id": "b",
                         "rollouts": _mk_rollouts([1, 0], [8, 15], [0.8, 0.1])}]
        out = M.best_of_metrics(per_instance)
        assert out["prm_mean"]["selected_reward"] == pytest.approx(
            out["oracle"]["selected_reward"])
        assert out["prm_mean"]["regret"] == pytest.approx(0.0)

    def test_regret_nonzero_for_bad_scores(self):
        per_instance = [{"instance_id": "a",
                         "rollouts": _mk_rollouts([0, 1], [10, 12], [0.9, 0.1])}]
        out = M.best_of_metrics(per_instance)
        assert out["prm_mean"]["regret"] == pytest.approx(1.0)  # 选错
        assert out["oracle"]["regret"] == pytest.approx(0.0)

    def test_ci_fields_present(self):
        per_instance = [{"instance_id": f"i{k}",
                         "rollouts": _mk_rollouts([0, 1], [10, 12], [0.9, 0.1 + 0.01 * (k % 3)])}
                        for k in range(20)]
        out = M.best_of_metrics(per_instance, n_boot=200)
        assert "reward_diff_vs_random_ci95" in out["prm_mean"]
        assert len(out["prm_mean"]["reward_diff_vs_random_ci95"]) == 2


class TestAggregates:
    def test_rollout_aggregates_manual(self):
        a = M.rollout_aggregates([0.2, 0.4, 0.9], gamma=0.95)
        assert a["mean"] == pytest.approx((0.2 + 0.4 + 0.9) / 3)
        assert a["min"] == pytest.approx(0.2)
        assert a["last"] == pytest.approx(0.9)
        g = 0.95
        expect = (g ** 0 * 0.2 + g ** 1 * 0.4 + g ** 2 * 0.9) / (1 + g + g ** 2)
        assert a["discounted"] == pytest.approx(expect)

    def test_empty(self):
        a = M.rollout_aggregates([])
        assert np.isnan(a["mean"]) and a["n_steps"] == 0


# ---------------------------------------------------------------------------
# summarize_split / report（torch 门控）
# ---------------------------------------------------------------------------

class TestSummarizeSplit:
    def test_summary_structure_and_values(self):
        pytest.importorskip("torch")
        from prm.eval_prm import summarize_split
        n = 200
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "instance_id": [f"i{i % 10}" for i in range(n)],
            "split": ["dev"] * n,
            "label": rng.random(n),
            "label_binary": (rng.random(n) > 0.4).astype(int),
            "label_source": ["node_mc"] * (n // 2) + ["leaf_chain"] * (n - n // 2),
            "mc_score": rng.random(n),
            "rendered_tokens": rng.integers(1000, 20000, n),
            "step_index": rng.integers(1, 10, n),
            "step_count": [10] * n,
            "score": np.clip(rng.random(n) + 0.2, 0, 1),
            "truncated": [bool(i % 3 == 0) for i in range(n)],
        })
        s = summarize_split(df)
        ov = s["overall"]
        assert 0 <= ov["roc_auc"] <= 1 and 0 <= ov["brier"] <= 1
        assert set(s["buckets"]["label_source"]) >= {"node_mc", "leaf_chain"}
        assert "auc_gap_node_mc_minus_leaf_chain" in s["buckets"]["label_source"]
        assert s["mc_correlation"]["n"] == n // 2
        assert s["calibration_bins"][0]["bin"].startswith("[0.0")
        # 截断与否分桶（U3）：两组齐备 + AUC 差 + 覆盖全部样本
        tr = s["buckets"]["truncated"]
        assert {"truncated", "not_truncated"} <= set(tr)
        assert "auc_gap_truncated_minus_clean" in tr
        assert tr["truncated"]["n"] + tr["not_truncated"]["n"] == n

    def test_truncated_bucket_absent_without_column(self):
        """df 无 truncated 列 → 空桶（兼容旧版 predictions.parquet）。"""
        pytest.importorskip("torch")
        from prm.eval_prm import bucket_block_truncated
        assert bucket_block_truncated(pd.DataFrame({"label_binary": [0, 1],
                                                    "score": [0.2, 0.8]})) == {}

    def test_cap_candidates(self):
        """§9.2「每实例 ≤k 条 root rollouts」：确定性取前 k；k<=0 不限制。"""
        from prm.metrics import cap_candidates
        rs = [{"rollout_idx": i, "reward": i} for i in [3, 0, 4, 1, 2, 6, 5]]
        assert [r["rollout_idx"] for r in cap_candidates(rs, 5)] == [0, 1, 2, 3, 4]
        assert len(cap_candidates(rs, 0)) == len(rs)      # 不限制
        assert len(cap_candidates(rs, 99)) == len(rs)     # k > 总数
        assert (cap_candidates(rs, 5)
                == cap_candidates(list(reversed(rs)), 5))  # 与输入顺序无关

    def test_report_md_contains_blocks(self):
        pytest.importorskip("torch")
        from prm.eval_prm import report_md, summarize_split
        df = pd.DataFrame({
            "sample_id": ["a", "b"], "instance_id": ["i", "i"], "split": ["dev", "dev"],
            "label": [1.0, 0.0], "label_binary": [1, 0], "label_source": ["node_mc", "leaf_chain"],
            "mc_score": [0.8, 0.0], "rendered_tokens": [100, 200],
            "step_index": [1, 1], "step_count": [1, 1], "score": [0.9, 0.1],
        })
        s = summarize_split(df)
        report = {"created_at": "t", "run": "r", "base_model": "m", "splits": {"dev": s}}
        md = report_md(report)
        assert "ROC-AUC" in md and "label_source 分桶" in md and "与 MC 相关性" in md
