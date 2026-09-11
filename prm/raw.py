# SPDX-License-Identifier: BSD-3-Clause

"""原始数据加载纯函数（docs/prm_training_plan.md §2）。

供 :mod:`prm.build_dataset` 与 :mod:`prm.eval_prm` 复用。**硬约束**：
``outputs/batch500/state.db``（~5GB）严格只读——连接一律经
:func:`open_db_readonly`（``mode=ro`` URI + ``PRAGMA query_only=ON``），任何
写入尝试都会被 SQLite 直接拒绝。

不信任 ``nodes.mc_score`` 缓存（§2.1）：标签一律由
:func:`load_mc_table` 从 ``rollouts.correct`` 重算（同时得到 ``n_rollouts``
用于 ``--min-rollouts`` 过滤；缓存仅用于"重算值 vs 缓存差 >1e-6"的异常审计）。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from typing import Optional

import pandas as pd

# MC 重算 SQL（§2 表格）：correct 为事实（失败 rollout 不落库，见 docs/construction/01 §1.3），
# AVG(CAST(correct AS REAL)) 即 mcts/node.py::MCTSNode.compute_mc 的 SQL 等价形式。
_MC_SQL = (
    "SELECT instance_id, node_key, AVG(CAST(correct AS REAL)) AS mc, COUNT(*) AS n "
    "FROM rollouts GROUP BY 1, 2"
)

# 节点遍历 SQL（§2 表格基础上追加 mc_score 列：§2.1-4 要求"重算值与缓存差 >1e-6
# 的节点跳过并列入构建报告"，需要缓存值做审计；其余列保持计划书原文）。
_NODES_SQL = (
    "SELECT instance_id, node_key, prefix_json, visits, in_pool, mc_score "
    "FROM nodes ORDER BY instance_id, node_key"
)


def open_db_readonly(db_path: str) -> sqlite3.Connection:
    """以**只读**方式打开 SQLite：URI ``mode=ro`` + ``PRAGMA query_only=ON``。

    双保险：即使拿到写连接的代码路径误调用 execute 写语句，query_only 也会拒绝。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_tree_status(conn: sqlite3.Connection) -> dict[str, str]:
    """``{instance_id: status}``（status 枚举见 docs/construction/01 §3.1）。

    构建器只用其中 ``done`` 的树（MCTS 引擎收束完成的实例）。
    """
    cur = conn.execute("SELECT instance_id, status FROM tree_instances")
    return {row[0]: row[1] for row in cur.fetchall()}


def load_instance_heads(conn: sqlite3.Connection) -> dict[str, str]:
    """``{instance_id: messages_head_json 原文}``。

    head = ``[system, user]`` 两条消息（注册时渲染写入，此后不变，docs/construction/01
    §3.1）；user 内容含 issue 描述 + agent 工具约定（§4.2 的 ``issue_and_conventions``
    原文来源）。返回**原文**不解析，解析交给调用方（:func:`head_user_content`）。
    """
    cur = conn.execute("SELECT instance_id, messages_head_json FROM tree_instances")
    return {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}


def load_mc_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """从 rollouts 重算 MC：``DataFrame[instance_id, node_key, mc, n_rollouts]``。

    为什么重算（§2.1）：``nodes.mc_score`` 是结算缓存（可能陈旧），标签的权威
    来源必须是"5 次续跑的成败计数"这一原始事实；重算顺带得到 ``n_rollouts``。
    """
    df = pd.read_sql_query(_MC_SQL, conn)
    df = df.rename(columns={"n": "n_rollouts"})
    return df[["instance_id", "node_key", "mc", "n_rollouts"]]


def iter_nodes(conn: sqlite3.Connection) -> Iterator[dict]:
    """逐行遍历节点表（cursor 迭代，不一次性物化 23k 行 × 前缀原文）。

    每行 dict：``instance_id / node_key / prefix_json / visits / in_pool / mc_score``
    （``mc_score`` 为缓存值，仅供 §2.1-4 审计比对，不作标签）。
    """
    cur = conn.execute(_NODES_SQL)
    cols = [d[0] for d in cur.description]
    for row in cur:
        yield dict(zip(cols, row))


def load_root_rollouts(conn: sqlite3.Connection, instance_id: str) -> list[dict]:
    """载入某实例 root 节点的全部 rollout（评估期 best-of-5 / 轨迹级评估用，§9.2）。

    返回 ``[{"rollout_idx", "correct", "reward", "submitted", "exit_status",
    "n_steps", "result_json 原文", "steps 解析后的 list"}]``。**不排除
    ``is_consumed=1``**：best-of-5 的候选就是 root 的 N 次续跑，消费账本只影响
    引擎选择、不影响评估口径（§9.2 best-of-5 注释）。
    """
    cur = conn.execute(
        "SELECT rollout_idx, correct, reward, submitted, exit_status, n_steps, result_json "
        "FROM rollouts WHERE instance_id = ? AND node_key = 'root' ORDER BY rollout_idx",
        (instance_id,),
    )
    out: list[dict] = []
    for rollout_idx, correct, reward, submitted, exit_status, n_steps, result_json in cur:
        result = json.loads(result_json) if result_json else {}
        out.append({
            "rollout_idx": rollout_idx,
            "correct": correct,
            "reward": reward,
            "submitted": submitted,
            "exit_status": exit_status,
            "n_steps": n_steps,
            "result": result,
            "steps": result.get("steps") or [],
        })
    return out


def load_splits(path: str) -> pd.DataFrame:
    """读 ``outputs/mcts/splits.parquet`` → ``DataFrame[instance_id, repo, prm_split]``。

    ``prm_split`` 非空（train/dev/test）的行才是 PRM 可用实例（``gen_pool`` 内
    60% 生成池才有树）；其余行（NaN）丢弃。
    """
    df = pd.read_parquet(path, columns=["instance_id", "repo", "prm_split"])
    df = df[df["prm_split"].notna()].copy()
    df["prm_split"] = df["prm_split"].astype(str)
    return df.reset_index(drop=True)[["instance_id", "repo", "prm_split"]]


def head_user_content(head_json: str | list) -> str:
    """从 messages_head（``[system, user]``）提取 user 内容原文（§4.2 规则 1）。

    取**最后一条** user 消息（head 恒为两条，取尾部即任务描述 + 工具约定）；
    结构异常时抛 ``ValueError``（head 缺失说明树数据不完整，不应静默降级）。
    """
    messages = json.loads(head_json) if isinstance(head_json, str) else list(head_json)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError(f"head user content 非字符串: {type(content)!r}")
            return content
    raise ValueError("messages_head 中无 user 消息")


def db_fingerprint(conn: sqlite3.Connection) -> dict:
    """构建前快照（§11 M4.1 验收：DB mtime/行数不变——行数由此函数提供，
    mtime 由调用方对文件本身取）。只读聚合，不触发表数据。"""
    counts = {}
    for table in ("tree_instances", "nodes", "rollouts"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


__all__ = [
    "open_db_readonly",
    "load_tree_status",
    "load_instance_heads",
    "load_mc_table",
    "iter_nodes",
    "load_root_rollouts",
    "load_splits",
    "head_user_content",
    "db_fingerprint",
]
