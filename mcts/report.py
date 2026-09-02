# SPDX-License-Identifier: BSD-3-Clause

"""MCTS rollout 数据报告（`outputs/mcts/rollout_report.{json,md}`）。

从 SQLite（``state.db``）聚合本次/累计数据生成的报告，供 PRM 训练数据采集的
验收与观测使用（PLAN §5 M3 验收：吞吐 / 失败率 / 标注统计 / 回报分布）。

- 每实例：状态 / root_mc / rollout 数 / 正确数 / 失败数 / 平均回报；
- 标注统计：best / leaf / add 数量与覆盖实例数；
- 全局统计：rollout 总数 / 失败率 / 平均回报 / 正确率 / 吞吐（来自 pipeline summary）；
- 输出 ``rollout_report.json``（结构化）+ ``rollout_report.md``（人类可读）。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from mcts.store import StateStore

logger = logging.getLogger("mcts.report")


def build_report(
    out_dir: str | Path,
    *,
    summary: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """从 ``out_dir/state.db`` 聚合生成报告 dict。

    Args:
        out_dir: ``outputs/mcts``（含 ``state.db``）。
        summary: ``MCTSPipeline.run()`` 返回的汇总（吞吐/耗时等）；None 表示
            ``--report-only``（只读现有库，无本次运行统计）。
        config: 运行配置快照（seed / trees / n_rollouts / concurrency 等）。
    """
    store = StateStore(out_dir / "state.db")
    try:
        per_instance = store.instance_summaries()
        ann_rows = store.annotation_summaries()
        counts = store.counts()
    finally:
        store.close()

    ann_by_instance: dict[str, dict[str, int]] = {}
    for row in ann_rows:
        ann_by_instance.setdefault(row["instance_id"], {})[row["type"]] = row["count"]

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "pipeline": summary or {},
        "db_counts": counts,
        "statuses": {},
        "rollout_stats": {
            "total": counts["rollouts"],
            "failed": 0,
            "fail_rate": 0.0,
            "correct": 0,
            "correct_rate": 0.0,
            "submitted": 0,
            "submit_rate": 0.0,
            "avg_reward": 0.0,
        },
        "annotation_stats": {"best": 0, "leaf": 0, "add": 0, "instances_annotated": 0},
        "per_instance": [],
    }

    n_inst_annotated = 0
    for inst in per_instance:
        status = inst["status"]
        report["statuses"][status] = report["statuses"].get(status, 0) + 1
        ann = ann_by_instance.get(inst["instance_id"], {})
        if ann:
            n_inst_annotated += 1
        report["per_instance"].append({
            **inst,
            "annotations": ann,
        })

    n = counts["rollouts"]
    if n:
        total_failed = sum(i["n_failed"] for i in per_instance)
        total_correct = sum(i["n_correct"] for i in per_instance)
        total_submitted = sum(i["n_submitted"] for i in per_instance)
        rewards = [i["avg_reward"] * i["n_rollouts"] for i in per_instance
                   if i["n_rollouts"] > 0]
        avg_reward = (sum(rewards) / n) if n else 0.0
        report["rollout_stats"] = {
            "total": n,
            "failed": total_failed,
            "fail_rate": round(total_failed / n, 4),
            "correct": total_correct,
            "correct_rate": round(total_correct / n, 4),
            "submitted": total_submitted,
            "submit_rate": round(total_submitted / n, 4),
            "avg_reward": round(avg_reward, 4),
        }

    for row in ann_rows:
        report["annotation_stats"][row["type"]] = (
            report["annotation_stats"].get(row["type"], 0) + row["count"]
        )
    report["annotation_stats"]["instances_annotated"] = n_inst_annotated

    if summary and "stats" in (summary or {}):
        s = summary["stats"]
        report["pipeline"] = {
            "elapsed_s": s.get("elapsed_s"),
            "throughput_rollouts_min": s.get("throughput_rollouts_min"),
            "avg_steps": s.get("avg_steps"),
            "total_llm_calls": s.get("total_llm_calls"),
            "total_cost": s.get("total_cost"),
            "env": s.get("env"),
            "db": s.get("db"),
        }
    return report


def write_report(out_dir: str | Path, report: dict) -> Path:
    """写 ``rollout_report.json`` + ``rollout_report.md``，返回 json 路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rollout_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rollout_report.md").write_text(_render_md(report), encoding="utf-8")
    return json_path


def _render_md(report: dict) -> str:
    rs = report.get("rollout_stats", {})
    as_ = report.get("annotation_stats", {})
    cfg = report.get("config", {}) or {}
    lines = [
        "# MCTS Rollout 数据报告（PRM 训练数据采集）",
        "",
        f"- 生成时间：{report.get('generated_at')}",
        f"- 配置：seed={cfg.get('seed')} trees={cfg.get('trees') or cfg.get('sample')} "
        f"n_rollouts={cfg.get('n_rollouts')} concurrency={cfg.get('concurrency')}",
        f"- 实例状态：{report.get('statuses')}",
        "",
        "## 全局统计",
        "",
        f"- rollout 总数：**{rs.get('total')}**；失败 **{rs.get('failed')}**（"
        f"失败率 {rs.get('fail_rate')}）；正确 **{rs.get('correct')}**（{rs.get('correct_rate')}）；"
        f"提交 **{rs.get('submitted')}**（提交率 {rs.get('submit_rate')}）",
        f"- 平均回报（submit_locations 结构化定位 F1 加权和）：**{rs.get('avg_reward')}**",
        f"- 标注：best **{as_.get('best')}** / leaf **{as_.get('leaf')}** / add **{as_.get('add')}**"
        f"（覆盖实例 {as_.get('instances_annotated')} 个）",
        "",
        "## 每实例明细",
        "",
        "| instance_id | status | root_mc | rollouts | correct | failed | avg_reward | best/leaf/add |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for inst in report.get("per_instance", []):
        ann = inst.get("annotations", {})
        lines.append(
            f"| {inst['instance_id']} | {inst['status']} | {inst.get('root_mc')} | "
            f"{inst['n_rollouts']} | {inst['n_correct']} | {inst['n_failed']} | "
            f"{inst['avg_reward']:.3f} | {ann.get('best', 0)}/{ann.get('leaf', 0)}/{ann.get('add', 0)} |"
        )
    lines += [
        "",
        "> 结构化数据见 `rollout_report.json`；rollout 详情（轨迹/步序列）在 `state.db`。",
        "",
    ]
    return "\n".join(lines)


__all__ = ["build_report", "write_report"]
