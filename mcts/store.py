# SPDX-License-Identifier: BSD-3-Clause

"""SQLite 状态存储（v5，docs/construction/01_database_design.md / 02_store_api.md）。

**v5 设计要点（相对 v2）**：
- 三表：``tree_instances / nodes / rollouts``（删除 v2 的 annotations 表与整树快照语义）；
- **事实与判定分离**：节点行/rollout 行 = 事实，随时落；会话判定（is_consumed /
  visits / n_rounds / in_pool）= :meth:`StateStore.commit_session` 单事务原子提交；
- **会话即事务**：一次 (select+locate) 会话成功结束才提交判定；崩溃中途 =
  判定未提交 = 该 (node, rollout) 未消费 → resume 重选重做（probe 结果经内容寻址复用）；
- **失败不落库**：rollout 硬失败 = 无行 = 缺槽补跑信号；``error`` 列移除；
- **in_pool 只随会话提交**：中断会话的孤儿探针永不入池（防 resume 候选池污染）。
- 并发：WAL + 单写锁（``threading.Lock``）串行化写入；``check_same_thread=False``。
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
CREATE TABLE IF NOT EXISTS tree_instances (
    instance_id        TEXT PRIMARY KEY,
    status             TEXT NOT NULL,           -- not_started / running / budget_exhausted / done / failed
    n_rounds           INTEGER NOT NULL DEFAULT 0,
    messages_head_json TEXT,
    root_mc            REAL,
    updated_at         REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,                  -- sha1(前缀序列化)；root 节点为 'root'
    status      TEXT NOT NULL DEFAULT 'rollout',-- rollout | ready
    prefix_json TEXT NOT NULL,
    mc_score    REAL,
    visits      INTEGER NOT NULL DEFAULT 0,
    in_pool     INTEGER NOT NULL DEFAULT 0,     -- 会话提交时为 expanded 置 1；root 恒 1
    created_at  REAL,
    PRIMARY KEY (instance_id, node_key)
);
CREATE TABLE IF NOT EXISTS rollouts (
    instance_id        TEXT NOT NULL,
    node_key           TEXT NOT NULL,
    rollout_idx        INTEGER NOT NULL,
    is_consumed        INTEGER NOT NULL DEFAULT 0,
    consumed_iteration INTEGER,
    result_json        TEXT NOT NULL,
    n_steps            INTEGER,
    correct            INTEGER,
    reward             REAL,
    submitted          INTEGER,
    replay_drift       INTEGER NOT NULL DEFAULT 0,
    exit_status        TEXT,
    created_at         REAL,
    PRIMARY KEY (instance_id, node_key, rollout_idx)
);
CREATE INDEX IF NOT EXISTS idx_nodes_instance        ON nodes (instance_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_node         ON rollouts (instance_id, node_key);
CREATE INDEX IF NOT EXISTS idx_rollouts_consumed     ON rollouts (instance_id, is_consumed);
"""

_LEGACY_HINT = (
    "检测到旧版(v2)状态库 schema（annotations/instances 表或 rollouts 缺 is_consumed）。"
    "v5 语义与旧库不兼容，请使用 `python -m mcts.run_mcts --fresh`（自动备份旧库后重建）"
    "或手工删除/迁移 outputs/mcts/state.db。"
)


class LegacySchemaError(RuntimeError):
    """旧版 schema 检测到且不自动迁移（语义不可靠还原，见施工文件 00 §6）。"""


def serialize_prefix(prefix_steps: list) -> str:
    return json.dumps(
        [s.to_json() if hasattr(s, "to_json") else s for s in prefix_steps],
        ensure_ascii=False,
    )


def deserialize_prefix(prefix_json: str) -> list:
    from mcts.steps import Step  # 惰性

    raw = json.loads(prefix_json) if prefix_json else []
    out = []
    for s in raw:
        try:
            out.append(Step.from_json(s) if isinstance(s, dict) else s)
        except Exception:  # noqa: BLE001 - 坏前缀按原始 dict 保留
            out.append(s)
    return out


class StateStore:
    """SQLite 状态存储（v5；WAL；写操作线程安全；会话提交原子）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._write_lock = threading.Lock()
        with self._write_lock:
            self._check_legacy()          # 先检测再建表（旧库不被 CREATE IF NOT EXISTS 污染）
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # schema 检测
    # ------------------------------------------------------------------

    def _check_legacy(self) -> None:
        tables = {
            r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if not tables:
            return  # 全新空库
        legacy = bool({'annotations', 'instances'} & tables)
        if 'rollouts' in tables:
            cols = {r[1] for r in self._conn.execute(
                "PRAGMA table_info(rollouts)").fetchall()}
            if 'is_consumed' not in cols:
                legacy = True
        if 'tree_instances' not in tables and 'rollouts' in tables:
            legacy = True
        if legacy:
            raise LegacySchemaError(_LEGACY_HINT)

    # ------------------------------------------------------------------
    # 树（tree_instances）
    # ------------------------------------------------------------------

    def create_tree(self, instance_id: str, messages_head: list[dict]) -> bool:
        """注册实例为 not_started 树；返回是否新建（幂等）。"""
        head_json = json.dumps(messages_head, ensure_ascii=False)
        with self._write_lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO tree_instances"
                " (instance_id, status, n_rounds, messages_head_json, updated_at)"
                " VALUES (?, 'not_started', 0, ?, ?)",
                (instance_id, head_json, time.time()),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def get_tree(self, instance_id: str) -> Optional[dict]:
        with self._write_lock:
            row = self._conn.execute(
                "SELECT status, n_rounds, messages_head_json, root_mc"
                " FROM tree_instances WHERE instance_id=?", (instance_id,),
            ).fetchone()
        if row is None:
            return None
        status, n_rounds, head_json, root_mc = row
        return {
            "status": status,
            "n_rounds": int(n_rounds),
            "messages_head": json.loads(head_json) if head_json else None,
            "root_mc": root_mc,
        }

    def set_tree_status(
        self, instance_id: str, status: str, *, root_mc: Optional[float] = None
    ) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE tree_instances SET status=?, root_mc=COALESCE(?, root_mc),"
                " updated_at=? WHERE instance_id=?",
                (status, root_mc, time.time(), instance_id),
            )
            self._conn.commit()

    def increment_rounds(self, instance_id: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE tree_instances SET n_rounds=n_rounds+1, updated_at=?"
                " WHERE instance_id=?", (time.time(), instance_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 节点（nodes）—— 事实随时落；判定（in_pool）只随会话提交
    # ------------------------------------------------------------------

    def ensure_node(self, instance_id: str, node_key: str, prefix_steps: list) -> bool:
        """节点诞生即插行（status=rollout，in_pool=root?1:0）；返回是否新建。"""
        with self._write_lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO nodes"
                " (instance_id, node_key, status, prefix_json, visits, in_pool, created_at)"
                " VALUES (?, ?, 'rollout', ?, 0, ?, ?)",
                (instance_id, node_key, serialize_prefix(prefix_steps),
                 int(node_key == "root"), time.time()),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def get_node_row(self, instance_id: str, node_key: str) -> Optional[dict]:
        with self._write_lock:
            row = self._conn.execute(
                "SELECT status, prefix_json, mc_score, visits, in_pool"
                " FROM nodes WHERE instance_id=? AND node_key=?",
                (instance_id, node_key),
            ).fetchone()
        if row is None:
            return None
        status, prefix_json, mc, visits, in_pool = row
        return {"status": status, "mc_score": mc, "visits": int(visits),
                "in_pool": int(in_pool), "prefix_steps": deserialize_prefix(prefix_json)}

    def set_node_ready(
        self, instance_id: str, node_key: str, *, mc_score: Optional[float]
    ) -> None:
        """节点一次补跑结算完成：status→ready + mc 缓存。"""
        with self._write_lock:
            self._conn.execute(
                "UPDATE nodes SET status='ready', mc_score=? WHERE instance_id=? AND node_key=?",
                (mc_score, instance_id, node_key),
            )
            self._conn.commit()

    def set_node_in_pool(self, instance_id: str, node_key: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE nodes SET in_pool=1 WHERE instance_id=? AND node_key=?",
                (instance_id, node_key),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # rollout（rollouts）—— 成功即落；失败不落库；消费账本只随会话提交
    # ------------------------------------------------------------------

    def upsert_rollout(self, result: Any) -> None:
        """写入一条成功 rollout（幂等 upsert；不存大块轨迹）。

        ON CONFLICT 只覆盖结果列，**不覆盖 is_consumed/consumed_iteration**
        （消费判定已提交后，同槽位结果重写不得回滚账本）。
        """
        payload = result.to_dict(include_trajectory=False)
        steps = getattr(result, "steps", None) or []
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO rollouts (instance_id, node_key, rollout_idx, result_json,"
                " n_steps, correct, reward, submitted, replay_drift, exit_status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(instance_id, node_key, rollout_idx)"
                " DO UPDATE SET result_json=excluded.result_json,"
                " n_steps=excluded.n_steps, correct=excluded.correct,"
                " reward=excluded.reward, submitted=excluded.submitted,"
                " replay_drift=excluded.replay_drift,"
                " exit_status=excluded.exit_status, created_at=excluded.created_at",
                (result.instance_id, result.node_key, result.rollout_idx,
                 json.dumps(payload, ensure_ascii=False),
                 len(steps), int(result.correct), float(result.reward),
                 int(result.exit_status == "Submitted"),
                 int(getattr(result, "replay_drift", False)),
                 result.exit_status or "", time.time()),
            )
            self._conn.commit()

    def load_node_rollouts(
        self, instance_id: str, node_key: str, n: int
    ) -> list[Optional[Any]]:
        """某节点已落库 rollout 结果（缺失槽位 None；断点续跑复用）。"""
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

    def load_node_rollout_rows(
        self, instance_id: str, node_key: str
    ) -> list[dict]:
        """原始行（含 is_consumed/consumed_iteration），载入用。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT rollout_idx, is_consumed, consumed_iteration, result_json,"
                " n_steps, correct, reward, submitted, replay_drift, exit_status"
                " FROM rollouts WHERE instance_id=? AND node_key=?"
                " ORDER BY rollout_idx",
                (instance_id, node_key),
            ).fetchall()
        return [
            {"rollout_idx": int(r[0]), "is_consumed": int(r[1]),
             "consumed_iteration": r[2], "result_json": r[3],
             "n_steps": r[4], "correct": r[5], "reward": r[6],
             "submitted": r[7], "replay_drift": int(r[8] or 0),
             "exit_status": r[9]}
            for r in rows
        ]

    def count_node_rollouts(self, instance_id: str, node_key: str) -> int:
        """成功 rollout 行数（缺槽判定的真值来源）。"""
        with self._write_lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM rollouts WHERE instance_id=? AND node_key=?",
                (instance_id, node_key),
            ).fetchone()
        return int(row[0])

    def is_rollout_consumed(
        self, instance_id: str, node_key: str, rollout_idx: int
    ) -> bool:
        with self._write_lock:
            row = self._conn.execute(
                "SELECT is_consumed FROM rollouts"
                " WHERE instance_id=? AND node_key=? AND rollout_idx=?",
                (instance_id, node_key, rollout_idx),
            ).fetchone()
        return bool(row and row[0])

    # ------------------------------------------------------------------
    # 会话原子提交（★核心：判定只在此单事务落库）
    # ------------------------------------------------------------------

    def commit_session(
        self,
        instance_id: str,
        node_key: str,
        rollout_idx: int,
        *,
        visits: int,
        increment_rounds: bool = False,
        n_rounds: Optional[int] = None,
        expanded: list[str] | tuple[str, ...] = (),
        consumed_iteration: Optional[int] = None,
    ) -> None:
        """(select+locate) 会话成功结束的原子提交（施工文件 02 §3.4）。

        同一事务内：
          - 该 (node, rollout) 置 is_consumed=1（+consumed_iteration）
          - 所属节点 visits 落为提交后的值
          - 本会话 expanded 探针 in_pool=1（非 root 时）
          - n_rounds+1（increment_rounds=True 时）
        任一步失败整体回滚 —— 崩溃在事务内 = 全部未提交 = 会话可重做。
        """
        if consumed_iteration is None and increment_rounds:
            consumed_iteration = n_rounds
        with self._write_lock:
            try:
                self._conn.execute(
                    "UPDATE rollouts SET is_consumed=1, consumed_iteration=?"
                    " WHERE instance_id=? AND node_key=? AND rollout_idx=?",
                    (consumed_iteration, instance_id, node_key, rollout_idx),
                )
                self._conn.execute(
                    "UPDATE nodes SET visits=? WHERE instance_id=? AND node_key=?",
                    (visits, instance_id, node_key),
                )
                if expanded:
                    self._conn.executemany(
                        "UPDATE nodes SET in_pool=1 WHERE instance_id=? AND node_key=?",
                        [(instance_id, nk) for nk in expanded],
                    )
                if increment_rounds and n_rounds is not None:
                    self._conn.execute(
                        "UPDATE tree_instances SET n_rounds=?, updated_at=?"
                        " WHERE instance_id=?",
                        (n_rounds, time.time(), instance_id),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # 整树载入（resume / 激活）
    # ------------------------------------------------------------------

    def load_tree_state(self, instance_id: str) -> dict:
        """整树载入：{tree, nodes: {node_key: row}, rollouts: {node_key: [row...]}}。

        不做判定（mc 重算/补跑由 TreeDriver 负责）；rollout 行含解析好的
        RolloutResult 与消费账本。
        """
        from mcts.tasks import RolloutResult  # 惰性

        tree = self.get_tree(instance_id)
        with self._write_lock:
            node_rows = self._conn.execute(
                "SELECT node_key, status, prefix_json, mc_score, visits, in_pool"
                " FROM nodes WHERE instance_id=? ORDER BY created_at",
                (instance_id,),
            ).fetchall()
            rollout_rows = self._conn.execute(
                "SELECT node_key, rollout_idx, is_consumed, consumed_iteration,"
                " result_json, n_steps, correct, reward, submitted, replay_drift,"
                " exit_status"
                " FROM rollouts WHERE instance_id=? ORDER BY node_key, rollout_idx",
                (instance_id,),
            ).fetchall()
        nodes: dict[str, dict] = {}
        for nk, status, prefix_json, mc, visits, in_pool in node_rows:
            nodes[nk] = {
                "status": status, "mc_score": mc, "visits": int(visits),
                "in_pool": int(in_pool),
                "prefix_steps": deserialize_prefix(prefix_json),
            }
        rollouts: dict[str, list[dict]] = {}
        for nk, idx, consumed, iter_, result_json, *_rest in rollout_rows:
            try:
                result = RolloutResult.from_dict(json.loads(result_json))
            except Exception as e:  # noqa: BLE001
                logger.warning("bad stored rollout %s/%s/%s: %s",
                               instance_id, nk, idx, e)
                continue
            rollouts.setdefault(nk, []).append(
                {"rollout_idx": int(idx), "is_consumed": bool(consumed),
                 "consumed_iteration": iter_, "result": result}
            )
        return {"tree": tree, "nodes": nodes, "rollouts": rollouts}

    # ------------------------------------------------------------------
    # 聚合（报告用，适配 v5）
    # ------------------------------------------------------------------

    def tree_summaries(self) -> list[dict]:
        """每树聚合：status / n_rounds / root_mc / n_nodes / n_rollouts /
        n_correct / n_consumed / n_submitted / avg_reward。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT t.instance_id, t.status, t.n_rounds, t.root_mc,"
                " COALESCE(n.n_nodes, 0),"
                " COALESCE(r.n_roll, 0), COALESCE(r.n_corr, 0),"
                " COALESCE(r.n_cons, 0), COALESCE(r.n_sub, 0), COALESCE(r.avg_rew, 0)"
                " FROM tree_instances t"
                " LEFT JOIN (SELECT instance_id, COUNT(*) AS n_nodes FROM nodes"
                "            GROUP BY instance_id) n"
                "   ON n.instance_id = t.instance_id"
                " LEFT JOIN (SELECT instance_id, COUNT(*) AS n_roll,"
                "                   SUM(correct) AS n_corr,"
                "                   SUM(is_consumed) AS n_cons,"
                "                   SUM(submitted) AS n_sub,"
                "                   AVG(reward) AS avg_rew"
                "            FROM rollouts GROUP BY instance_id) r"
                "   ON r.instance_id = t.instance_id"
                " ORDER BY t.instance_id",
            ).fetchall()
        return [
            {"instance_id": r[0], "status": r[1], "n_rounds": int(r[2]),
             "root_mc": r[3], "n_nodes": int(r[4]), "n_rollouts": int(r[5]),
             "n_correct": int(r[6]), "n_consumed": int(r[7]),
             "n_submitted": int(r[8]), "avg_reward": float(r[9])}
            for r in rows
        ]

    def leaf_summaries(self) -> list[dict]:
        """每实例 leaf 派生统计：非 root ∧ mc==0（替代旧 annotations 统计）。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT instance_id, COUNT(*) FROM nodes"
                " WHERE node_key != 'root' AND mc_score = 0.0"
                " GROUP BY instance_id ORDER BY instance_id",
            ).fetchall()
        return [{"instance_id": r[0], "count": int(r[1])} for r in rows]

    def count_rollouts(self, instance_id: Optional[str] = None) -> int:
        with self._write_lock:
            if instance_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM rollouts WHERE instance_id=?", (instance_id,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()
        return int(row[0])

    def counts(self) -> dict:
        with self._write_lock:
            n_t = self._conn.execute("SELECT COUNT(*) FROM tree_instances").fetchone()[0]
            n_n = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            n_r = self._conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()[0]
        return {"trees": int(n_t), "nodes": int(n_n), "rollouts": int(n_r)}

    def close(self) -> None:
        with self._write_lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass


__all__ = ["StateStore", "LegacySchemaError",
           "serialize_prefix", "deserialize_prefix"]
