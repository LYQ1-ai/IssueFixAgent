# SPDX-License-Identifier: BSD-3-Clause

"""StateStore v5 单元测试（施工文件 05 §2）。

覆盖：schema/旧库检测、tree/nodes/rollouts CRUD、commit_session 原子性、
upsert 不覆盖消费账本、load_tree_state、聚合、并发写。
"""

import sqlite3
import threading

import pytest

from mcts.store import LegacySchemaError, StateStore


def make_store(tmp_path) -> StateStore:
    return StateStore(tmp_path / "state.db")


class _FakeResult:
    """最小 RolloutResult 兼容对象（真实流程由 RolloutResult 提供）。"""

    def __init__(self, iid="i", node_key="root", idx=0, correct=True,
                 reward=1.0, exit_status="Submitted", steps=None,
                 replay_drift=False):
        self.instance_id = iid
        self.node_key = node_key
        self.rollout_idx = idx
        self.correct = correct
        self.reward = reward
        self.exit_status = exit_status
        self.steps = steps if steps is not None else []
        self.replay_drift = replay_drift
        self.error = None

    def to_dict(self, include_trajectory=True):
        return {
            "instance_id": self.instance_id, "node_key": self.node_key,
            "rollout_idx": self.rollout_idx, "correct": self.correct,
            "reward": self.reward, "steps": self.steps,
            "exit_status": self.exit_status, "error": self.error,
        }


def _setup(s: StateStore, iid="i"):
    s.create_tree(iid, [{"role": "user", "content": "task"}])
    s.ensure_node(iid, "root", [])
    s.ensure_node(iid, "n1", [{"assistant": {"content": "a"}}])


class TestSchema:
    def test_fresh_schema(self, tmp_path):
        s = make_store(tmp_path)
        tables = {r[0] for r in s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"tree_instances", "nodes", "rollouts"} <= tables
        assert "annotations" not in tables and "instances" not in tables
        cols = {r[1] for r in s._conn.execute("PRAGMA table_info(rollouts)")}
        assert {"is_consumed", "consumed_iteration", "result_json"} <= cols
        s.close()

    def test_legacy_detection(self, tmp_path):
        """旧 v2 库（annotations/instances 表）打开即抛 LegacySchemaError。"""
        p = tmp_path / "legacy.db"
        c = sqlite3.connect(str(p))
        c.executescript(
            "CREATE TABLE instances (instance_id TEXT PRIMARY KEY, status TEXT);"
            "CREATE TABLE rollouts (instance_id TEXT, node_key TEXT,"
            " rollout_idx INTEGER, result_json TEXT, error TEXT,"
            " PRIMARY KEY (instance_id,node_key,rollout_idx));"
            "CREATE TABLE annotations (id INTEGER PRIMARY KEY);")
        c.commit()
        c.close()
        with pytest.raises(LegacySchemaError):
            StateStore(p)


class TestTree:
    def test_create_get_idempotent(self, tmp_path):
        s = make_store(tmp_path)
        assert s.create_tree("i1", [{"role": "user", "content": "q"}]) is True
        assert s.create_tree("i1", []) is False
        t = s.get_tree("i1")
        assert t["status"] == "not_started" and t["n_rounds"] == 0
        assert t["messages_head"][0]["content"] == "q"
        assert s.get_tree("missing") is None
        s.close()

    def test_status_and_rounds(self, tmp_path):
        s = make_store(tmp_path)
        s.create_tree("i1", [])
        s.set_tree_status("i1", "running", root_mc=0.4)
        s.increment_rounds("i1")
        t = s.get_tree("i1")
        assert t["status"] == "running" and t["root_mc"] == 0.4
        assert t["n_rounds"] == 1
        s.close()


class TestNodes:
    def test_ensure_node(self, tmp_path):
        s = make_store(tmp_path)
        s.create_tree("i1", [])
        assert s.ensure_node("i1", "root", []) is True
        assert s.ensure_node("i1", "root", []) is False  # 幂等
        assert s.ensure_node("i1", "n1", [{"assistant": {"content": "a"}}]) is True
        row = s.get_node_row("i1", "root")
        assert row["status"] == "rollout" and row["in_pool"] == 1
        assert s.get_node_row("i1", "n1")["in_pool"] == 0
        s.close()

    def test_set_node_ready(self, tmp_path):
        s = make_store(tmp_path)
        s.create_tree("i1", [])
        s.ensure_node("i1", "n1", [{"assistant": {"content": "a"}}])
        s.set_node_ready("i1", "n1", mc_score=0.6)
        row = s.get_node_row("i1", "n1")
        assert row["status"] == "ready" and row["mc_score"] == 0.6
        s.close()


class TestRollouts:
    def test_upsert_load(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        s.upsert_rollout(_FakeResult(idx=2, correct=False, reward=0.0,
                                     exit_status="LimitsExceeded"))
        assert s.count_node_rollouts("i", "root") == 2
        slots = s.load_node_rollouts("i", "root", 3)
        assert slots[0] is not None and slots[1] is None and slots[2] is not None
        assert slots[2].correct is False
        s.close()

    def test_upsert_never_overwrites_consumed(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        s.commit_session("i", "root", 0, visits=1, increment_rounds=False)
        # 同槽位结果重写（防御性）→ is_consumed 保留
        s.upsert_rollout(_FakeResult(idx=0, reward=2.0))
        assert s.is_rollout_consumed("i", "root", 0) is True
        row = s.load_node_rollout_rows("i", "root")[0]
        assert row["is_consumed"] == 1
        s.close()


class TestCommitSession:
    def test_success(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        s.commit_session("i", "root", 0, visits=3, increment_rounds=False,
                         expanded=["n1"])
        assert s.is_rollout_consumed("i", "root", 0) is True
        assert s.get_node_row("i", "root")["visits"] == 3
        assert s.get_node_row("i", "n1")["in_pool"] == 1
        assert s.get_tree("i")["n_rounds"] == 0
        s.close()

    def test_increment_rounds_and_iteration(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.ensure_node("i", "n2", [{"assistant": {"content": "b"}}])
        s.upsert_rollout(_FakeResult(idx=0, node_key="n1"))
        s.commit_session("i", "n1", 0, visits=1, increment_rounds=True,
                         n_rounds=4, expanded=["n2"])
        assert s.get_tree("i")["n_rounds"] == 4
        row = s.load_node_rollout_rows("i", "n1")[0]
        assert row["consumed_iteration"] == 4
        s.close()

    def test_atomic_rollback(self, tmp_path):
        """事务中途失败 → is_consumed/visits/n_rounds/in_pool 全部回滚。"""
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        with pytest.raises(sqlite3.ProgrammingError):
            s.commit_session("i", "root", 0, visits=5, increment_rounds=True,
                             n_rounds=7, expanded=[("bad", "tuple")])
        assert s.is_rollout_consumed("i", "root", 0) is False
        assert s.get_node_row("i", "root")["visits"] == 0
        assert s.get_tree("i")["n_rounds"] == 0
        assert s.get_node_row("i", "n1")["in_pool"] == 0
        s.close()


class TestLoadTreeState:
    def test_roundtrip(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        s.commit_session("i", "root", 0, visits=1, increment_rounds=False,
                         expanded=["n1"])
        st = s.load_tree_state("i")
        assert st["tree"]["status"] == "not_started"
        assert set(st["nodes"]) == {"root", "n1"}
        assert st["nodes"]["root"]["in_pool"] == 1
        assert st["rollouts"]["root"][0]["is_consumed"] is True
        assert st["rollouts"]["root"][0]["result"].reward == 1.0
        s.close()


class TestAggregates:
    def test_summaries(self, tmp_path):
        s = make_store(tmp_path)
        _setup(s)
        s.upsert_rollout(_FakeResult(idx=0))
        s.upsert_rollout(_FakeResult(idx=1, correct=False, reward=0.0))
        s.commit_session("i", "root", 0, visits=1, increment_rounds=False)
        s.set_node_ready("i", "n1", mc_score=0.0)
        sums = s.tree_summaries()
        assert sums[0]["n_rollouts"] == 2
        assert sums[0]["n_correct"] == 1
        assert sums[0]["n_consumed"] == 1
        leaves = s.leaf_summaries()
        assert leaves[0]["count"] == 1  # n1 mc==0 且非 root
        assert s.counts()["rollouts"] == 2
        s.close()


class TestConcurrency:
    def test_parallel_writes(self, tmp_path):
        s = make_store(tmp_path)
        s.create_tree("i", [])
        s.ensure_node("i", "root", [])
        errors = []

        def writer():
            try:
                for j in range(20):
                    s.upsert_rollout(_FakeResult(
                        idx=j, correct=(j % 2 == 0),
                        reward=1.0 if j % 2 == 0 else 0.0))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert s.count_node_rollouts("i", "root") == 20
        s.close()
