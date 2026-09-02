# SPDX-License-Identifier: BSD-3-Clause

"""SQLite checkpoint 存储（v2，PLAN §2.3 / docs/mcts_engine_design.md）。

**角色**：MCTS 树结构与 Rollout 结果的**唯一持久化事实源**（取代 v1 的每 rollout
一个 JSON 文件）：

- **树结构与结果常驻内存**，SQLite 只是崩溃安全的镜像：
  - rollout 完成即 upsert（单条事务，WAL 模式，~ms 级）；
  - 树结构（nodes 表）由 TreeDriver **定期快照**（每次 locate 一轮 / 实例完成时）；
  - 实例状态（done / failed / budget_exhausted）与标注（best/leaf/add）落库；
- **断点恢复**（后续里程碑实现）：``running`` 实例可从 rollouts 表按
  ``node_key``（前缀内容哈希）重建节点继续 select/locate；本模块先保证
  schema 与数据完整。
- 并发：WAL + 单写锁（``threading.Lock``）串行化写入（40–50 worker 并发写
  单条事务，SQLite 可承受）；``check_same_thread=False`` 供线程池调用。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("mcts.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    status      TEXT,               -- running / done / failed / budget_exhausted
    root_mc     REAL,
    messages_head_json TEXT,        -- system+user 头部（probe 前缀消息拼接用）
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    prefix_json TEXT NOT NULL,      -- 前缀步序列化（node_key 内容寻址的还原依据）
    mc_score    REAL,
    visits      INTEGER DEFAULT 0,
    n_rollouts  INTEGER DEFAULT 0,  -- 有效 rollout 数（不含失败）
    rollouts_json TEXT,             -- {idx: bool} 各 rollout 是否已落库
    updated_at  REAL,
    PRIMARY KEY (instance_id, node_key)
);
CREATE TABLE IF NOT EXISTS rollouts (
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    rollout_idx INTEGER NOT NULL,
    result_json TEXT NOT NULL,      -- RolloutResult 序列化（steps + 回报；不含大块轨迹）
    correct     INTEGER,            -- 聚合列（报告/统计用，避免全量 parse JSON）
    reward      REAL,
    submitted   INTEGER,            -- 聚合列：exit_status == "Submitted"（提交协议改造后统计提交率）
    error       TEXT,
    created_at  REAL,
    PRIMARY KEY (instance_id, node_key, rollout_idx)
);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    type        TEXT NOT NULL,      -- best / leaf / add
    mc_score    REAL,
    n_steps     INTEGER,
    written_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_rollouts_node ON rollouts (instance_id, node_key);
CREATE INDEX IF NOT EXISTS idx_nodes_instance ON nodes (instance_id);
CREATE INDEX IF NOT EXISTS idx_annotations_instance ON annotations (instance_id);
"""


class StateStore:
    """SQLite 状态存储（WAL；写操作线程安全）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._write_lock = threading.Lock()
        with self._write_lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """向后兼容：早期库的 rollouts 表缺聚合列时补列（v2.1+ 新增
        correct/reward/error；v2.2 新增 submitted —— 提交协议改造）。"""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(rollouts)").fetchall()}
        for col, decl in (("correct", "INTEGER"), ("reward", "REAL"),
                          ("error", "TEXT"), ("submitted", "INTEGER")):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE rollouts ADD COLUMN {col} {decl}")

    # ------------------------------------------------------------------
    # rollout 结果（完成即落库，崩溃安全的最小粒度）
    # ------------------------------------------------------------------

    def upsert_rollout(self, result: Any) -> None:
        """写入一条 rollout 结果（幂等 upsert；不存大块轨迹）。"""
        payload = result.to_dict(include_trajectory=False)
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO rollouts (instance_id, node_key, rollout_idx, result_json,"
                " correct, reward, submitted, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(instance_id, node_key, rollout_idx)"
                " DO UPDATE SET result_json=excluded.result_json,"
                " correct=excluded.correct, reward=excluded.reward,"
                " submitted=excluded.submitted,"
                " error=excluded.error, created_at=excluded.created_at",
                (result.instance_id, result.node_key, result.rollout_idx,
                 json.dumps(payload, ensure_ascii=False),
                 int(result.correct), float(result.reward),
                 int(result.exit_status == "Submitted"),
                 result.error, time.time()),
            )
            self._conn.commit()

    def load_node_rollouts(
        self, instance_id: str, node_key: str, n: int
    ) -> list[Optional[Any]]:
        """读取某节点已落库的 rollout 结果（缺失槽位为 None，断点续跑复用）。"""
        from mcts.tasks import RolloutResult  # 惰性，避免循环导入

        slots: list[Optional[Any]] = [None] * n
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT rollout_idx, result_json FROM rollouts"
                " WHERE instance_id=? AND node_key=?",
                (instance_id, node_key),
            ).fetchall()
        for idx, result_json in rows:
            try:
                slots[int(idx)] = RolloutResult.from_dict(json.loads(result_json))
            except Exception as e:  # noqa: BLE001 - 坏记录当缺失处理
                logger.warning("bad stored rollout %s/%s/%s: %s",
                               instance_id, node_key, idx, e)
        return slots

    def count_rollouts(self, instance_id: Optional[str] = None) -> int:
        with self._write_lock:
            if instance_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM rollouts WHERE instance_id=?", (instance_id,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # 树结构快照（定期写，内存是权威，DB 是镜像）
    # ------------------------------------------------------------------

    def save_nodes(self, instance_id: str, nodes: list[Any]) -> None:
        """整树节点快照（upsert；覆盖该实例的 nodes 行）。"""
        now = time.time()
        with self._write_lock:
            for node in nodes:
                self._conn.execute(
                    "INSERT INTO nodes (instance_id, node_key, prefix_json, mc_score,"
                    " visits, n_rollouts, rollouts_json, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(instance_id, node_key)"
                    " DO UPDATE SET prefix_json=excluded.prefix_json,"
                    " mc_score=excluded.mc_score, visits=excluded.visits,"
                    " n_rollouts=excluded.n_rollouts, rollouts_json=excluded.rollouts_json,"
                    " updated_at=excluded.updated_at",
                    (
                        instance_id, node.node_key,
                        json.dumps([s.to_json() for s in node.prefix_steps],
                                   ensure_ascii=False),
                        node.mc_score, node.visits, node.n_rollouts,
                        json.dumps({i: True for i, r in enumerate(node.rollouts)
                                    if not getattr(r, "error", None)},
                                   ensure_ascii=False),
                        now,
                    ),
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 标注与实例状态
    # ------------------------------------------------------------------

    def write_annotations(self, instance_id: str, entries: list[dict]) -> None:
        """写入（覆盖）该实例的标注条目（best/leaf/add；同类型可重复）。"""
        now = time.time()
        with self._write_lock:
            self._conn.execute(
                "DELETE FROM annotations WHERE instance_id=?", (instance_id,))
            self._conn.executemany(
                "INSERT INTO annotations (instance_id, node_key, type, mc_score,"
                " n_steps, written_at) VALUES (?, ?, ?, ?, ?, ?)",
                [(instance_id, e.get("node_key", ""), e.get("type", ""),
                  e.get("mc_score"), e.get("n_steps"), now + i * 1e-6)
                 for i, e in enumerate(entries)],
            )
            self._conn.commit()

    def set_instance_status(
        self,
        instance_id: str,
        status: str,
        *,
        root_mc: Optional[float] = None,
        messages_head: Optional[list[dict]] = None,
    ) -> None:
        """更新实例状态（running / done / failed / budget_exhausted）。"""
        head_json = (
            json.dumps(messages_head, ensure_ascii=False)
            if messages_head is not None else None
        )
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO instances (instance_id, status, root_mc,"
                " messages_head_json, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(instance_id)"
                " DO UPDATE SET status=excluded.status, root_mc=excluded.root_mc,"
                " messages_head_json=COALESCE(excluded.messages_head_json,"
                " instances.messages_head_json), updated_at=excluded.updated_at",
                (instance_id, status, root_mc, head_json, time.time()),
            )
            self._conn.commit()

    def get_instance_status(self, instance_id: str) -> Optional[dict]:
        with self._write_lock:
            row = self._conn.execute(
                "SELECT status, root_mc, messages_head_json FROM instances"
                " WHERE instance_id=?", (instance_id,),
            ).fetchone()
        if row is None:
            return None
        status, root_mc, head_json = row
        return {"status": status, "root_mc": root_mc,
                "messages_head": json.loads(head_json) if head_json else None}

    # ------------------------------------------------------------------
    # 聚合（数据报告用）
    # ------------------------------------------------------------------

    def instance_summaries(self) -> list[dict]:
        """每实例聚合：status / root_mc / n_rollouts / n_correct / n_failed /
        n_submitted / avg_reward。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT i.instance_id, i.status, i.root_mc,"
                " COALESCE(COUNT(r.rollout_idx), 0),"
                " COALESCE(SUM(r.correct), 0),"
                " COALESCE(SUM(CASE WHEN r.error IS NOT NULL THEN 1 ELSE 0 END), 0),"
                " COALESCE(SUM(r.submitted), 0),"
                " COALESCE(AVG(r.reward), 0)"
                " FROM instances i LEFT JOIN rollouts r ON r.instance_id = i.instance_id"
                " GROUP BY i.instance_id ORDER BY i.instance_id",
            ).fetchall()
        return [
            {"instance_id": r[0], "status": r[1], "root_mc": r[2],
             "n_rollouts": int(r[3]), "n_correct": int(r[4]),
             "n_failed": int(r[5]), "n_submitted": int(r[6]),
             "avg_reward": float(r[7])}
            for r in rows
        ]

    def annotation_summaries(self) -> list[dict]:
        """每实例每类型标注计数（best / leaf / add）。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT instance_id, type, COUNT(*) FROM annotations"
                " GROUP BY instance_id, type ORDER BY instance_id, type",
            ).fetchall()
        return [{"instance_id": r[0], "type": r[1], "count": int(r[2])} for r in rows]

    # ------------------------------------------------------------------
    # 汇总 / 关闭
    # ------------------------------------------------------------------

    def counts(self) -> dict:
        with self._write_lock:
            n_rollouts = self._conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()[0]
            n_instances = self._conn.execute("SELECT COUNT(*) FROM instances").fetchone()[0]
            n_annotations = self._conn.execute(
                "SELECT COUNT(*) FROM annotations").fetchone()[0]
        return {"instances": int(n_instances), "rollouts": int(n_rollouts),
                "annotations": int(n_annotations)}

    def close(self) -> None:
        with self._write_lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass


__all__ = ["StateStore"]
