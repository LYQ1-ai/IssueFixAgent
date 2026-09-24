# SPDX-License-Identifier: BSD-3-Clause

"""评估指标与分桶（docs/prm_training_plan.md §9 / §10 test_prm_eval）。

纯 numpy 实现（ROC-AUC 用秩统计、ECE 等宽分桶、bootstrap CI），不依赖
sklearn——build/eval 报告在任意带 numpy 的环境可算。约定：正类 =
``label_binary == 1``（继续执行大概率修复成功）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC（秩统计实现，并列取平均秩；Mann-Whitney U 等价形式）。

    全同分数（无区分能力）→ 0.5；标签单一 → NaN（不可评估）。
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")  # 稳定排序 → 并列取平均秩
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """PR-AUC（按阈值扫描的阶梯积分，逐步长 = 该阈值下 recall 增量）。"""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or len(y) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    # 阶梯积分：recall 增量 × 当前 precision（每个正类样本处取该阈值 precision）
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) + recall[0] * precision[0])


def classification_metrics(y_true: Sequence[int], scores: Sequence[float],
                           threshold: float = 0.5) -> dict:
    """Accuracy（默认阈值 0.5）/ P / R / F1（正类）。

    无正类或无预测正类时对应项为 0.0（约定，便于汇总）。
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(scores, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def brier_score(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Brier = mean((p − y)²)（校准主看指标，§9.2）。"""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: Sequence[int], scores: Sequence[float],
                               n_bins: int = 10) -> float:
    """ECE：等宽 10 桶，Σ (n_b/N)·|acc_b − conf_b|（主看指标，§9.2）。"""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def calibration_bins(y_true: Sequence[int], scores: Sequence[float],
                     n_bins: int = 10) -> list[dict]:
    """等宽分桶明细（报告用）：n / mean_score / pos_rate。"""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        bins.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})" if b < n_bins - 1
                   else f"[{edges[b]:.1f},{edges[b + 1]:.1f}]",
            "n": int(mask.sum()),
            "mean_score": float(p[mask].mean()) if mask.any() else float("nan"),
            "pos_rate": float(y[mask].mean()) if mask.any() else float("nan"),
        })
    return bins


def log_loss(y_true: Sequence[int], scores: Sequence[float], eps: float = 1e-7) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    if len(y) == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman = 秩上的 Pearson（并列平均秩）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    return pearson(_average_ranks(x), _average_ranks(y))


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def bootstrap_ci(metric_fn, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """bootstrap 置信区间（§9.2 best-of-5 差值报 95% CI）。

    ``metric_fn(*resampled_arrays) -> float``；NaN 结果跳过。
    """
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    if n == 0:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = [np.asarray(a)[idx] for a in arrays]
        v = metric_fn(*vals)
        if v is not None and np.isfinite(v):
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def bucket_metrics(y_true: Sequence[int], scores: Sequence[float],
                   buckets: Sequence[Sequence[int]]) -> list[dict]:
    """按给定索引分组计算 AUC/n（label_source 分桶、位置分桶等，§9.2）。

    Args:
        buckets: 每组一个样本索引序列（如 ``np.where(group == g)[0]``）。
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    out = []
    for idx in buckets:
        idx = np.asarray(idx, dtype=np.int64)
        sub_y, sub_s = y[idx], s[idx]
        out.append({
            "n": int(len(idx)),
            "pos_rate": float(sub_y.mean()) if len(idx) else float("nan"),
            "mean_score": float(sub_s.mean()) if len(idx) else float("nan"),
            "auc": roc_auc(sub_y, sub_s),
        })
    return out


def position_bucket(step_index: int, step_count: int) -> str:
    """绝对步数分桶（§9.2）：1/2/3/4/5/6-10/11-15/16-20/21+。"""
    if step_index <= 5:
        return str(step_index)
    if step_index <= 10:
        return "6-10"
    if step_index <= 15:
        return "11-15"
    if step_index <= 20:
        return "16-20"
    return "21+"


def relative_position_bucket(step_index: int, step_count: int) -> str:
    """相对位置 10% 分桶（§9.2）：step_index/step_count → [0,10%),…,[90%,100%]。"""
    if step_count <= 0:
        return "nan"
    frac = (step_index - 1) / max(1, step_count)  # 被判定步在 1..count
    b = min(9, int(frac * 10))
    return f"[{b * 10}%,{(b + 1) * 10}%]"


def context_length_bucket(tokens: Optional[int]) -> str:
    """上下文长度分桶（§9.2）：≤4K / 4-8K / 8-12K / 12-16K / >16K。"""
    if tokens is None:
        return "nan"
    if tokens <= 4096:
        return "<=4K"
    if tokens <= 8192:
        return "4-8K"
    if tokens <= 12288:
        return "8-12K"
    if tokens <= 16384:
        return "12-16K"
    return ">16K"


# ---------------------------------------------------------------------------
# best-of-N 选择器（§9.2；纯函数便于离线单测）
# ---------------------------------------------------------------------------

def discounted_aggregate(scores: Sequence[float], gamma: float = 0.95) -> float:
    """轨迹级 discounted 聚合：Σ γ^t·p_t / Σ γ^t（t=0 为最后一步? 约定 t=0 为第一步）。"""
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        return float("nan")
    gammas = gamma ** np.arange(len(s), dtype=np.float64)
    return float((gammas * s).sum() / gammas.sum())


def rollout_aggregates(step_scores: Sequence[float], gamma: float = 0.95) -> dict:
    """一条 rollout 的逐步 PRM 分数 → 聚合（mean/min/last/discounted）。"""
    s = np.asarray(step_scores, dtype=np.float64)
    if len(s) == 0:
        return {"mean": float("nan"), "min": float("nan"), "last": float("nan"),
                "discounted": float("nan"), "n_steps": 0}
    return {
        "mean": float(s.mean()),
        "min": float(s.min()),
        "last": float(s[-1]),
        "discounted": discounted_aggregate(s, gamma),
        "n_steps": int(len(s)),
    }


SELECTORS = ("random", "shortest", "prm_mean", "prm_min", "prm_last", "oracle")


def _agg_value(r: dict, key: str) -> float:
    """读取 PRM 聚合字段；缺失/NaN → -inf（选择时排到最后）。"""
    v = r.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return v if np.isfinite(v) else float("-inf")


def select_rollout(rollouts: list[dict], selector: str, *,
                   rng: Optional[np.random.Generator] = None) -> Optional[dict]:
    """从一条实例的 ≤N 条候选 rollout 中按策略选一条（§9.2 选择器）。

    ``rollouts[i]`` 需含 ``rollout_idx / reward / n_steps`` 与 PRM 聚合字段
    （``agg_mean / agg_min / agg_last``）。空候选 → None。
    """
    if not rollouts:
        return None
    if selector == "random":
        rng = rng or np.random.default_rng(0)
        return rollouts[int(rng.integers(len(rollouts)))]
    if selector == "shortest":
        return min(rollouts, key=lambda r: (r.get("n_steps") or 0, r.get("rollout_idx") or 0))
    if selector == "oracle":
        return max(rollouts, key=lambda r: (_agg_value(r, "reward"),
                                            -int(r.get("rollout_idx") or 0)))
    if selector.startswith("prm_"):
        key = f"agg_{selector.split('_', 1)[1]}"
        return max(rollouts, key=lambda r: (_agg_value(r, key),
                                            -int(r.get("rollout_idx") or 0)))
    raise ValueError(f"未知选择器: {selector}")


def cap_candidates(rollouts: Sequence[dict], k: int,
                   key: str = "rollout_idx") -> list[dict]:
    """按 ``key`` 升序取前 ``k`` 条候选（§9.2「每实例 ≤5 条 root rollouts」）。

    ``k <= 0`` 表示不限制。固定候选数是 best-of-N 配对实验的前提：N 越大，
    ``oracle = max(reward)`` 的期望越高、PRM 选择器的收益也越大，而 ``random``
    的期望与 N 无关 → 各实例 N 不等会把「N 的差异」混进 ``vs random`` 的配对差 CI。
    取前 k 条（而非随机抽样）保证同一份数据重复评估结果一致。
    """
    if k is None or int(k) <= 0:
        return list(rollouts)
    return sorted(rollouts, key=lambda r: r.get(key, 0))[:int(k)]


def best_of_metrics(per_instance: list[dict], selectors: Sequence[str] = SELECTORS,
                    n_boot: int = 1000, seed: int = 42) -> dict:
    """best-of-N 指标（§9.2）：selected_reward / selected_correct / regret / win_rate，
    差值（相对 random 的逐实例配对差）报 bootstrap 95% CI。

    ``per_instance[i] = {"instance_id", "rollouts": [...]}``（rollouts 字段同
    :func:`select_rollout`；oracle 基准 = max reward）。
    """
    rng = np.random.default_rng(seed)
    summary: dict[str, dict] = {}
    per_selector_rewards: dict[str, list[float]] = {}
    per_selector_correct: dict[str, list[float]] = {}
    for sel in selectors:
        rewards, corrects, regrets, wins = [], [], [], []
        for inst in per_instance:
            cands = inst["rollouts"]
            pick = select_rollout(cands, sel, rng=rng)
            if pick is None:
                continue
            r = float(pick.get("reward", 0.0) or 0.0)
            best_r = max(float(c.get("reward", 0.0) or 0.0) for c in cands)
            rewards.append(r)
            corrects.append(float(bool(pick.get("correct"))))
            regrets.append(best_r - r)
            wins.append(1.0 if r >= best_r else 0.0)
        per_selector_rewards[sel] = rewards
        per_selector_correct[sel] = corrects
        summary[sel] = {
            "n_instances": len(rewards),
            "selected_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "selected_correct": float(np.mean(corrects)) if corrects else float("nan"),
            "regret": float(np.mean(regrets)) if regrets else float("nan"),
            "win_rate": float(np.mean(wins)) if wins else float("nan"),
        }
    # 配对差值 CI：selector − random（同实例配对）
    for sel in selectors:
        if sel == "random":
            continue
        a = np.asarray(per_selector_rewards[sel], dtype=np.float64)
        b = np.asarray(per_selector_rewards.get("random", []), dtype=np.float64)
        if len(a) and len(a) == len(b):
            lo, hi = bootstrap_ci(lambda x, y: float(np.mean(x - y)), a, b,
                                  n_boot=n_boot, seed=seed)
            summary[sel]["reward_diff_vs_random_ci95"] = [lo, hi]
        c = np.asarray(per_selector_correct[sel], dtype=np.float64)
        d = np.asarray(per_selector_correct.get("random", []), dtype=np.float64)
        if len(c) and len(c) == len(d):
            lo, hi = bootstrap_ci(lambda x, y: float(np.mean(x - y)), c, d,
                                  n_boot=n_boot, seed=seed)
            summary[sel]["correct_diff_vs_random_ci95"] = [lo, hi]
    return summary


__all__ = [
    "roc_auc", "pr_auc", "classification_metrics", "brier_score",
    "expected_calibration_error", "calibration_bins", "log_loss",
    "pearson", "spearman", "bootstrap_ci", "bucket_metrics",
    "position_bucket", "relative_position_bucket", "context_length_bucket",
    "discounted_aggregate", "rollout_aggregates", "select_rollout",
    "best_of_metrics", "SELECTORS", "cap_candidates",
]
