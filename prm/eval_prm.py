# SPDX-License-Identifier: BSD-3-Clause

"""PRM 评估 CLI（docs/prm_training_plan.md §9.2，M4.4 验收产出）。

评估内容（dev/test，进程内批量前向，无生成）：

1. **总指标**：Accuracy(p>0.5)、P/R/F1、ROC-AUC（选型主指标）、PR-AUC、
   Brier/ECE（主看）、Log loss；
2. **label_source 分桶**：node_mc vs leaf_chain 两桶 AUC 差（主验收项——回填
   推定正样本必须单独盯）；
3. **位置/长度分桶**：相对位置 10% 桶、绝对步数桶、上下文长度桶、截断与否；
4. **与 MC 相关性**：test 真实非 leaf 节点（label_source=node_mc）上
   Pearson/Spearman(PRM p, mc)；
5. **轨迹级**：root rollouts（``--db`` 读 result_json.steps，§4 模板渲染逐步
   打分），聚合 mean/min/last/discounted(γ) 与 correct/reward/submitted 相关；
6. **best-of-5**：每实例 ≤5 条 root rollouts；选择器 random/shortest/
   PRM-mean/PRM-min/PRM-last/oracle；差值报 bootstrap 95% CI。

产出：``outputs/prm/eval_report.{md,json}`` + ``predictions.parquet``
（+ ``traj_predictions.parquet``）。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from prm import metrics as M
from prm.env import describe as describe_env, load_project_env
from prm.data import (PrmParquetDataset, VerdictCollator, collate_or_skip,
                      tensors_from)
from prm.modeling import VerdictScorer, resolve_verdict_ids
from prm.prompts import template_hash

logger = logging.getLogger("prm.eval")


# ---------------------------------------------------------------------------
# 模型加载（run 目录 = train_prm 产物）
# ---------------------------------------------------------------------------

def load_scorer_for_run(run_dir: str, device: str = "cuda",
                        attn_implementation: str = "sdpa") -> tuple[VerdictScorer, dict, VerdictCollator]:
    """从 run 目录恢复可评估模型：run_manifest（verdict ids / lora / max_length /
    template_hash）→ 基座 + LoRA → 权重加载（model.safetensors）。

    ``template_hash`` 不一致 → 抛错（prompt 与模型错配，禁止评估）。
    """
    import torch
    run = Path(run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("template_hash") != template_hash():
        raise RuntimeError("run 的 template_hash 与当前 prompts.py 不一致——重建数据/复训后再评估")
    collator = VerdictCollator(tokenizer_path=manifest["base_model"],
                               max_length=manifest["max_length"])
    ids = resolve_verdict_ids(collator.tokenizer, tuple(manifest["verdict_pair"]))
    scorer = VerdictScorer.from_pretrained(
        manifest["base_model"], ids, torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation, lora=manifest.get("lora"))
    state: dict
    if (run / "model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(str(run / "model.safetensors"))
    else:
        state = torch.load(run / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = scorer.backbone.load_state_dict(state, strict=False)
    if missing:
        logger.warning("加载 run 权重缺失键: %s", list(missing)[:5])
    if unexpected:
        logger.warning("加载 run 权重出现多余键: %s", list(unexpected)[:5])
    scorer.to(device).eval()
    logger.info("已加载 run: %s（verdict=%s, max_length=%s）",
                run_dir, manifest["verdict_pair"], manifest["max_length"])
    return scorer, manifest, collator


# ---------------------------------------------------------------------------
# 样本级评估 + 汇总
# ---------------------------------------------------------------------------

def evaluate_samples(scorer: VerdictScorer, collator: VerdictCollator, ds: PrmParquetDataset,
                     device: str, batch_size: int = 1) -> pd.DataFrame:
    """逐样本前向 → predictions DataFrame（含分桶字段）。"""
    import torch
    scores: list[float] = []
    trunc_flags: list[bool] = []
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = collate_or_skip(collator, items)      # §7.2 预算不足 → 跳过该样本
        if batch is None:
            continue
        t = tensors_from(batch, device=device)
        with torch.no_grad():
            p = scorer.predict_proba(t["input_ids"], t["attention_mask"])
        scores.extend(p.float().cpu().tolist())
        if "truncated" in t:
            trunc_flags.extend(bool(v) for v in t["truncated"].cpu().tolist())
    keys = ("sample_id", "instance_id", "split", "label", "label_binary", "label_source",
            "mc_score", "rendered_tokens", "step_index", "step_count")
    rows = [{k: ds[j][k] for k in keys} for j in range(len(ds))]
    df = pd.DataFrame(rows)
    df["score"] = scores
    if trunc_flags:
        df["truncated"] = trunc_flags
    return df


def summarize_split(df: pd.DataFrame) -> dict:
    """§9.2 块 1–4：总指标 + label_source 分桶 + 位置/长度分桶 + MC 相关性。"""
    y = df["label_binary"].to_numpy(dtype=np.int64)
    p = df["score"].to_numpy(dtype=np.float64)
    overall = {
        "n": int(len(df)),
        "pos_rate": float(y.mean()),
        **M.classification_metrics(y, p),
        "roc_auc": M.roc_auc(y, p),
        "pr_auc": M.pr_auc(y, p),
        "brier": M.brier_score(y, p),
        "ece": M.expected_calibration_error(y, p),
        "log_loss": M.log_loss(y, p),
    }
    # label_source 分桶（主验收项）
    source_buckets = {}
    for src in ("node_mc", "leaf_chain"):
        sub = df[df["label_source"] == src]
        if len(sub):
            sy = sub["label_binary"].to_numpy(dtype=np.int64)
            sp = sub["score"].to_numpy(dtype=np.float64)
            source_buckets[src] = {
                "n": int(len(sub)), "auc": M.roc_auc(sy, sp), "brier": M.brier_score(sy, sp),
            }
    if {"node_mc", "leaf_chain"} <= source_buckets.keys():
        source_buckets["auc_gap_node_mc_minus_leaf_chain"] = (
            source_buckets["node_mc"]["auc"] - source_buckets["leaf_chain"]["auc"])
    # 位置/长度分桶
    buckets = {
        "label_source": source_buckets,
        "position_abs": bucket_block_abs(df),
        "context_length": bucket_block_len(df),
        "truncated": bucket_block_truncated(df),
    }
    # MC 相关性（真实非 leaf 节点 = node_mc）
    sub = df[df["label_source"] == "node_mc"]
    mc_corr = {
        "n": int(len(sub)),
        "pearson": M.pearson(sub["score"], sub["mc_score"].astype(float)),
        "spearman": M.spearman(sub["score"], sub["mc_score"].astype(float)),
    }
    return {"overall": overall, "buckets": buckets, "mc_correlation": mc_corr,
            "calibration_bins": M.calibration_bins(y, p)}


def bucket_block_abs(df: pd.DataFrame) -> dict:
    """绝对步数 / 相对位置分桶（需 step_index/step_count 列）。"""
    out: dict[str, dict] = {}
    if not {"step_index", "step_count"} <= set(df.columns):
        return out
    y = df["label_binary"].to_numpy(dtype=np.int64)
    p = df["score"].to_numpy(dtype=np.float64)
    pos_abs = [M.position_bucket(int(a), int(c))
               for a, c in zip(df["step_index"], df["step_count"])]
    pos_rel = [M.relative_position_bucket(int(a), int(c))
               for a, c in zip(df["step_index"], df["step_count"])]
    for name, keys in (("relative", pos_rel), ("absolute", pos_abs)):
        groups: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        out[name] = {g: M.bucket_metrics(y, p, [idx])[0]
                     for g, idx in sorted(groups.items())}
    return out


def bucket_block_len(df: pd.DataFrame) -> dict:
    y = df["label_binary"].to_numpy(dtype=np.int64)
    p = df["score"].to_numpy(dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(df["rendered_tokens"]):
        groups.setdefault(M.context_length_bucket(int(t) if t is not None else None),
                          []).append(i)
    return {g: M.bucket_metrics(y, p, [idx])[0] for g, idx in sorted(groups.items())}


def bucket_block_truncated(df: pd.DataFrame) -> dict:
    """「是否被 §7.2 截断」分桶（§9.2）。

    回答一个关键问题：**被截断样本的打分质量是否显著更差**——用于区分"长上下文
    本身难"与"截断把语义撕碎"（两者的对策相反：前者要更多长样本，后者要扩
    max_length 或改截断策略）。
    """
    if "truncated" not in df.columns:
        return {}
    y = df["label_binary"].to_numpy(dtype=np.int64)
    p = df["score"].to_numpy(dtype=np.float64)
    out: dict[str, dict] = {}
    for flag, name in ((False, "not_truncated"), (True, "truncated")):
        idx = [i for i, v in enumerate(df["truncated"]) if bool(v) is flag]
        if idx:
            out[name] = M.bucket_metrics(y, p, [idx])[0]
    if {"truncated", "not_truncated"} <= out.keys():
        out["auc_gap_truncated_minus_clean"] = (
            out["truncated"]["auc"] - out["not_truncated"]["auc"])
    return out


# ---------------------------------------------------------------------------
# 轨迹级评估 + best-of-5（§9.2 块 5/6）
# ---------------------------------------------------------------------------

def trajectory_eval(scorer: VerdictScorer, collator: VerdictCollator, conn,
                    instance_ids: list[str], heads: dict[str, str], device: str,
                    batch_size: int = 1, gamma: float = 0.95,
                    max_instances: Optional[int] = None,
                    max_steps_per_rollout: int = 32,
                    k_candidates: int = 5) -> tuple[pd.DataFrame, dict]:
    """root rollouts 逐步打分 → 聚合相关性 + best-of-5。

    返回 ``(traj_df, bestof_summary)``；``traj_df`` 每行 = 一条 root rollout 的聚合。
    """
    import torch
    from prm.raw import head_user_content, load_root_rollouts
    from prm.preprocess import TrajectoryPreprocessor

    pre = TrajectoryPreprocessor()
    iids = instance_ids[:max_instances] if max_instances else instance_ids
    jobs: list[tuple[str, int, int, list[dict]]] = []  # (iid, rollout_idx, step_i, messages)
    meta: dict[tuple[str, int], dict] = {}
    for iid in iids:
        # §9.2「每实例 ≤5 条 root rollouts」：按 rollout_idx 升序取前 k 条（确定性）
        rollouts = M.cap_candidates(load_root_rollouts(conn, iid), k_candidates)
        if not rollouts:
            continue
        head_user = head_user_content(heads[iid]) if iid in heads else None
        if head_user is None:
            continue
        for r in rollouts:
            steps = r["steps"][:max_steps_per_rollout]
            meta[(iid, r["rollout_idx"])] = {
                "reward": r["reward"], "correct": r["correct"], "submitted": r["submitted"],
                "n_steps": r["n_steps"], "exit_status": r["exit_status"],
            }
            for i in range(1, len(steps) + 1):
                jobs.append((iid, r["rollout_idx"], i,
                             pre.build_messages(head_user, pre.parse_steps(steps[:i]))))
    logger.info("trajectory eval: %d 实例 × root rollouts → %d 个前缀前向",
                len(iids), len(jobs))

    # 分批前向
    scores: dict[tuple[str, int], list[float]] = {}
    batch: list[list[dict]] = []
    keys: list[tuple[str, int, int]] = []

    def flush() -> None:
        if not batch:
            return
        items = [{"messages": m, "label": 0.0} for m in batch]
        tensors = collate_or_skip(collator, items)    # §7.2 预算不足 → 整批丢弃并计数
        if tensors is None:
            batch.clear()
            keys.clear()
            return
        t = tensors_from(tensors, device=device)
        with torch.no_grad():
            p = scorer.predict_proba(t["input_ids"], t["attention_mask"])
        for (iid, ri, _si), sc in zip(keys, p.float().cpu().tolist()):
            # jobs 按 (iid, rollout_idx, i) 升序入队 → 同一 rollout 的分数按步序到达
            scores.setdefault((iid, ri), []).append(sc)
        batch.clear()
        keys.clear()

    for iid, ri, si, msgs in jobs:
        batch.append(msgs)
        keys.append((iid, ri, si))
        if len(batch) >= batch_size:
            flush()
    flush()

    rows = []
    for (iid, ri), agg in scores.items():
        m = meta[(iid, ri)]
        a = M.rollout_aggregates(agg, gamma=gamma)
        rows.append({"instance_id": iid, "rollout_idx": ri,
                     "reward": m["reward"], "correct": m["correct"],
                     "submitted": m["submitted"], "n_steps": m["n_steps"],
                     "exit_status": m["exit_status"],
                     "agg_mean": a["mean"], "agg_min": a["min"],
                     "agg_last": a["last"], "agg_discounted": a["discounted"],
                     "scored_steps": a["n_steps"]})
    traj_df = pd.DataFrame(rows)

    # 聚合 vs 结局相关（point-biserial ≈ Pearson）
    corr = {}
    if len(traj_df):
        for agg_key in ("agg_mean", "agg_min", "agg_last", "agg_discounted"):
            corr[agg_key] = {
                "correct": M.pearson(traj_df[agg_key], traj_df["correct"].astype(float)),
                "reward": M.pearson(traj_df[agg_key], traj_df["reward"].astype(float)),
                "submitted": M.pearson(traj_df[agg_key], traj_df["submitted"].astype(float)),
                "spearman_correct": M.spearman(traj_df[agg_key],
                                               traj_df["correct"].astype(float)),
            }
    # best-of-5（PRM 排序键 = agg_mean/min/last 各自测一轮？——按 §9.2 用 mean/min/last
    # 三个选择器共用同一份聚合，这里以 agg_mean 为主排序键）
    per_instance = []
    for iid, grp in traj_df.groupby("instance_id"):
        per_instance.append({"instance_id": iid, "rollouts": grp.to_dict("records")})
    # 候选数分布（审计 §9.2「每实例 ≤k 条」是否满足；N 不等会影响 oracle/regret 基准）
    cand_counts = [len(x["rollouts"]) for x in per_instance]
    cand_hist: dict[str, int] = {}
    for c in cand_counts:
        cand_hist[str(c)] = cand_hist.get(str(c), 0) + 1
    bestof = M.best_of_metrics(per_instance, n_boot=1000) if per_instance else {}
    return traj_df, {
        "correlations": corr,
        "best_of": bestof,
        "k_candidates": int(k_candidates),
        "n_candidates": {
            "n_instances": len(cand_counts),
            "min": min(cand_counts) if cand_counts else 0,
            "max": max(cand_counts) if cand_counts else 0,
            "mean": float(sum(cand_counts) / len(cand_counts)) if cand_counts else 0.0,
            "hist": dict(sorted(cand_hist.items(), key=lambda kv: int(kv[0]))),
        },
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def report_md(report: dict) -> str:
    lines = ["# PRM 评估报告（M4.4）", "",
             f"- 生成时间: {report['created_at']} | run: `{report['run']}`"
             f" | 基座: `{report['base_model']}`", ""]
    for split, s in report["splits"].items():
        ov = s["overall"]
        lines += [
            f"## split = {split}",
            "",
            f"- n={ov['n']}（正类率 {ov['pos_rate']:.4f}）",
            f"- **ROC-AUC {ov['roc_auc']:.4f}**（选型主指标）| PR-AUC {ov['pr_auc']:.4f}",
            f"- Brier {ov['brier']:.4f} | ECE {ov['ece']:.4f}（校准主看）| LogLoss {ov['log_loss']:.4f}",
            f"- Acc {ov['accuracy']:.4f} | P {ov['precision']:.4f} | R {ov['recall']:.4f} | F1 {ov['f1']:.4f}",
            "",
            "### label_source 分桶（主验收项）",
            "",
            "| source | n | AUC | Brier |",
            "| --- | --- | --- | --- |",
        ]
        for src in ("node_mc", "leaf_chain"):
            b = s["buckets"]["label_source"].get(src)
            if b:
                lines.append(f"| {src} | {b['n']} | {b['auc']:.4f} | {b['brier']:.4f} |")
        gap = s["buckets"]["label_source"].get("auc_gap_node_mc_minus_leaf_chain")
        if gap is not None:
            lines += ["", f"- AUC 差（node_mc − leaf_chain）= {gap:.4f}"]
        mc = s["mc_correlation"]
        lines += ["", f"### 与 MC 相关性（node_mc, n={mc['n']}）", "",
                  f"- Pearson {mc['pearson']:.4f} | Spearman {mc['spearman']:.4f}", ""]
        tr = s["buckets"].get("truncated") or {}
        if tr:
            lines += ["### 截断与否分桶（§7.2 截断对打分质量的影响）", "",
                      "| group | n | AUC | Brier |", "| --- | --- | --- | --- |"]
            for name in ("not_truncated", "truncated"):
                b = tr.get(name)
                if b:
                    lines.append(f"| {name} | {b['n']} | {b['auc']:.4f} | {b['brier']:.4f} |")
            gap_t = tr.get("auc_gap_truncated_minus_clean")
            if gap_t is not None:
                lines += ["", f"- AUC 差（truncated − not_truncated）= {gap_t:.4f}"
                              "（显著为负 ⇒ 截断样本质量更差，考虑扩 max_length 或改截断策略）"]
            lines.append("")
    if report.get("trajectory"):
        bo = report["trajectory"]["best_of"]
        nc = report["trajectory"].get("n_candidates") or {}
        lines += ["## 轨迹级 / best-of-5（test root rollouts）", ""]
        if nc:
            lines += [f"- 候选数（每实例 ≤ k={report['trajectory'].get('k_candidates')}）："
                      f"n_instances={nc['n_instances']} min={nc['min']} max={nc['max']} "
                      f"mean={nc['mean']:.2f} | 分布 {nc['hist']}", ""]
        lines += ["| selector | n | selected_reward | CI95(Δvs random) | selected_correct | regret | win_rate |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for sel, m in bo.items():
            ci = m.get("reward_diff_vs_random_ci95")
            ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "-"
            lines.append(f"| {sel} | {m['n_instances']} | {m['selected_reward']:.4f} "
                         f"| {ci_s} | {m['selected_correct']:.4f} | {m['regret']:.4f} "
                         f"| {m['win_rate']:.4f} |")
        corr = report["trajectory"]["correlations"]
        lines += ["", "### 聚合分数 vs 结局相关（Pearson）", "",
                  "| aggregate | correct | reward | submitted |", "| --- | --- | --- | --- |"]
        for k, v in corr.items():
            lines.append(f"| {k} | {v['correct']:.4f} | {v['reward']:.4f} | {v['submitted']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    load_project_env()          # 入口装载 .env（GPU/CUDA 选择），早于 torch/CUDA 初始化
    p = argparse.ArgumentParser(prog="python -m prm.eval_prm", description="PRM 评估（§9.2）")
    p.add_argument("--config", default="config/prm.yaml")
    p.add_argument("--run", required=True, help="run 名（outputs/prm/runs/<name>）")
    p.add_argument("--splits", nargs="*", default=["dev", "test"])
    p.add_argument("--db", default=None, help="轨迹级评估用 DB（默认 config.db.path）")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--traj-max-instances", type=int, default=None,
                   help="轨迹级评估的实例数上限（全量 test 树较慢时可限）")
    p.add_argument("--skip-trajectory", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    logger.info("GPU 选择: %s", describe_env())
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["output"]["dir"])
    run_dir = data_dir / "runs" / args.run
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or cfg.get("eval", {}).get("batch_size", 1)
    if batch_size != 1:
        raise SystemExit(
            f"batch_size 必须为 1（当前 {batch_size}）：Qwen3.5 混合架构对 pad 前缀敏感，"
            "多样本批的 padding 会改变打分；依据见 docs/prm_training_plan.md §14.2")

    scorer, manifest, collator = load_scorer_for_run(str(run_dir), device=device,
                                                     attn_implementation=args.attn)
    report: dict = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run": args.run, "base_model": manifest["base_model"],
        "verdict_pair": manifest["verdict_pair"], "max_length": manifest["max_length"],
        "template_hash": template_hash(), "splits": {}, "collator_stats": collator.stats,
    }
    predictions = []
    for split in args.splits:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            logger.warning("跳过 %s（无 %s）", split, path)
            continue
        ds = PrmParquetDataset(str(path))
        df = evaluate_samples(scorer, collator, ds, device, batch_size)
        report["splits"][split] = summarize_split(df)
        predictions.append(df)
        logger.info("[%s] AUC=%.4f Brier=%.4f", split,
                    report["splits"][split]["overall"]["roc_auc"],
                    report["splits"][split]["overall"]["brier"])

    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(
            data_dir / "predictions.parquet", index=False)

    # 轨迹级 + best-of-5（test）
    if not args.skip_trajectory:
        from prm import raw
        conn = raw.open_db_readonly(args.db or cfg["db"]["path"])
        heads = raw.load_instance_heads(conn)
        test_ids = sorted(set(predictions[-1]["instance_id"])) if predictions else []
        traj_df, traj_summary = trajectory_eval(
            scorer, collator, conn, test_ids, heads, device, batch_size,
            gamma=cfg.get("eval", {}).get("best_of", {}).get("gamma", 0.95),
            k_candidates=cfg.get("eval", {}).get("best_of", {}).get("k", 5),
            max_instances=args.traj_max_instances)
        report["trajectory"] = traj_summary
        if len(traj_df):
            traj_df.to_parquet(data_dir / "traj_predictions.parquet", index=False)
        conn.close()

    (data_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (data_dir / "eval_report.md").write_text(report_md(report), encoding="utf-8")
    logger.info("评估报告已写出: %s", data_dir / "eval_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
