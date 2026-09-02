# SPDX-License-Identifier: BSD-3-Clause

"""MCTS rollout 数据报告（v5，`outputs/mcts/rollout_report.{json,md}`）。

从 SQLite（``state.db``，v5 三表）聚合数据生成报告（PLAN §5 M3 验收口径）：

- 每实例（tree_summaries）：status / n_rounds / root_mc / n_nodes / rollouts /
  correct / consumed / submitted / avg_reward / 派生 leaf 数；
- 全局：rollout 总数 / 正确率 / 消费数 / 提交率 / 平均回报 / 吞吐（pipeline summary）；
- 标注统计（v5 派生口径）：leaf = 非 root ∧ mc==0；best/add 已移除（施工文件 00 D8）；
- 输出 ``rollout_report.json`` + ``rollout_report.md``。
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
            ``--report-only``（只读现有库）。
        config: 运行配置快照。
    """
    store = StateStore(out_dir / "state.db")
    try:
        per_tree = store.tree_summaries()
        leaf_rows = store.leaf_summaries()
        counts = store.counts()
    finally:
        store.close()

    leaf_by_instance: dict[str, int] = {
        r["instance_id"]: r["count"] for r in leaf_rows
    }

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "pipeline": summary or {},
        "db_counts": counts,
        "statuses": {},
        "rollout_stats": {
            "total": counts["rollouts"],
            "correct": 0,
            "correct_rate": 0.0,
            "consumed": 0,
            "consumed_rate": 0.0,
            "submitted": 0,
            "submit_rate": 0.0,
            "avg_reward": 0.0,
        },
        "leaf_stats": {"total": 0, "instances_with_leaf": 0},
        "per_instance": [],
    }

    n = counts["rollouts"]
    total_correct = 0
    total_consumed = 0
    total_submitted = 0
    reward_sum = 0.0
    n_inst_with_leaf = 0
    for inst in per_tree:
        report["statuses"][inst["status"]] = report["statuses"].get(
            inst["status"], 0) + 1
        n_leaf = leaf_by_instance.get(inst["instance_id"], 0)
        if n_leaf:
            n_inst_with_leaf += 1
        total_correct += inst["n_correct"]
        total_consumed += inst["n_consumed"]
        total_submitted += inst["n_submitted"]
        reward_sum += inst["avg_reward"] * inst["n_rollouts"]
        report["per_instance"].append({
            "instance_id": inst["instance_id"],
            "status": inst["status"],
            "n_rounds": inst["n_rounds"],
            "root_mc": inst["root_mc"],
            "n_nodes": inst["n_nodes"],
            "n_rollouts": inst["n_rollouts"],
            "n_correct": inst["n_correct"],
            "n_consumed": inst["n_consumed"],
            "n_submitted": inst["n_submitted"],
            "avg_reward": round(inst["avg_reward"], 4),
            "n_leaf": n_leaf,
        })
    if n:
        report["rollout_stats"] = {
            "total": n,
            "correct": total_correct,
            "correct_rate": round(total_correct / n, 4),
            "consumed": total_consumed,
            "consumed_rate": round(total_consumed / n, 4),
            "submitted": total_submitted,
            "submit_rate": round(total_submitted / n, 4),
            "avg_reward": round(reward_sum / n, 4),
        }
    report["leaf_stats"] = {
        "total": sum(leaf_by_instance.values()),
        "instances_with_leaf": n_inst_with_leaf,
    }

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
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    (out_dir / "rollout_report.md").write_text(_render_md(report),
                                               encoding="utf-8")
    return json_path


def _render_md(report: dict) -> str:
    rs = report.get("rollout_stats", {})
    ls = report.get("leaf_stats", {})
    cfg = report.get("config", {}) or {}
    lines = [
        "# MCTS Rollout 数据报告（PRM 训练数据采集，v5）",
        "",
        f"- 生成时间：{report.get('generated_at')}",
        f"- 配置：seed={cfg.get('seed')} trees={cfg.get('trees') or cfg.get('sample')} "
        f"n_rollouts={cfg.get('n_rollouts')} concurrency={cfg.get('concurrency')}",
        f"- 实例状态：{report.get('statuses')}",
        "",
        "## 全局统计",
        "",
        f"- rollout 总数：**{rs.get('total')}**；正确 **{rs.get('correct')}**"
        f"（{rs.get('correct_rate')}）；已消费 **{rs.get('consumed')}**"
        f"（{rs.get('consumed_rate')}）；提交 **{rs.get('submitted')}**"
        f"（{rs.get('submit_rate')}）",
        f"- 平均回报：**{rs.get('avg_reward')}**（v5 只存成功 rollout，无失败行）",
        f"- leaf（首个错误步负样本，派生 非root∧mc==0）：**{ls.get('total')}**"
        f"（覆盖实例 {ls.get('instances_with_leaf')} 个）",
        "",
        "## 每实例明细",
        "",
        "| instance_id | status | rounds | root_mc | nodes | rollouts | correct |"
        " consumed | avg_reward | leaf |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for inst in report.get("per_instance", []):
        lines.append(
            f"| {inst['instance_id']} | {inst['status']} | {inst['n_rounds']} | "
            f"{inst.get('root_mc')} | {inst['n_nodes']} | {inst['n_rollouts']} | "
            f"{inst['n_correct']} | {inst['n_consumed']} | {inst['avg_reward']} | "
            f"{inst['n_leaf']} |"
        )
    lines += [
        "",
        "> 结构化数据见 `rollout_report.json`；rollout 详情（轨迹/步序列）在 `state.db`。",
        "",
    ]
    return "\n".join(lines)


__all__ = ["build_report", "write_report"]
