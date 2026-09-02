# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.store`` SQLite 持久化测试（纯逻辑，离线可跑）。

覆盖：schema 初始化、rollout upsert/load 往返（不含轨迹）、节点快照、
标注与实例状态、并发写入（worker 线程模拟）。
"""

import json
import threading

from mcts.store import StateStore
from mcts.tasks import RolloutResult

from mcts.tests.helpers import make_instance, make_step


def _result(iid="inst1", node_key="root", idx=0, correct=True, n_steps=2) -> RolloutResult:
    return RolloutResult(
        instance_id=iid, node_key=node_key, rollout_idx=idx,
        reward=1.0 if correct else 0.0, correct=correct,
        steps=[make_step(f"step {j}") for j in range(n_steps)],
        trajectory={"info": {}, "messages": [{"role": "exit"}]},  # 落库时应被剔除
        exit_status="Submitted" if correct else "LimitsExceeded",
        submission="diff --git a/x.py b/x.py" if correct else "",
        n_calls=n_steps, cost=0.0, duration=0.1, replay_drift=False,
        trace_id="trace-1",
    )


class TestRolloutRoundtrip:
    def test_upsert_and_load(self, workdir):
        store = StateStore(workdir / "state.db")
        try:
            store.upsert_rollout(_result())
            slots = store.load_node_rollouts("inst1", "root", 5)
            assert slots[0] is not None
            assert slots[1] is None                       # 未写的槽位为 None
            r = slots[0]
            assert r.correct is True and r.reward == 1.0
            assert len(r.steps) == 2
            assert r.trace_id == "trace-1"
        finally:
            store.close()

    def test_trajectory_not_persisted(self, workdir):
        store = StateStore(workdir / "state.db")
        try:
            store.upsert_rollout(_result())
            with store._write_lock:
                row = store._conn.execute(
                    "SELECT result_json FROM rollouts").fetchone()
            payload = json.loads(row[0])
            assert "trajectory" not in payload             # DB 体积控制
            assert "steps" in payload
        finally:
            store.close()

    def test_upsert_idempotent(self, workdir):
        store = StateStore(workdir / "state.db")
        try:
            store.upsert_rollout(_result(correct=True))
            store.upsert_rollout(_result(correct=False))   # 同 key 覆盖
            assert store.count_rollouts("inst1") == 1
            slots = store.load_node_rollouts("inst1", "root", 5)
            assert slots[0].correct is False
        finally:
            store.close()


class TestNodeSnapshot:
    def test_save_nodes_roundtrip(self, workdir):
        from mcts.node import MCTSNode

        store = StateStore(workdir / "state.db")
        try:
            node = MCTSNode(instance_id="inst1", node_key="n_key",
                            prefix_steps=[make_step("p1"), make_step("p2")])
            node.mc_score = 0.6
            node.visits = 3
            node.add_rollout(_result(correct=True))
            node.add_rollout(_result(correct=False))
            store.save_nodes("inst1", [node])
            with store._write_lock:
                row = store._conn.execute(
                    "SELECT prefix_json, mc_score, visits, n_rollouts, rollouts_json"
                    " FROM nodes WHERE instance_id='inst1' AND node_key='n_key'").fetchone()
            prefix = json.loads(row[0])
            assert len(prefix) == 2
            assert row[1] == 0.6 and row[2] == 3 and row[3] == 2
            # JSON 键为字符串
            assert json.loads(row[4]) == {"0": True, "1": True}
        finally:
            store.close()


class TestAnnotationsAndStatus:
    def test_annotations_write_and_status(self, workdir):
        store = StateStore(workdir / "state.db")
        try:
            store.write_annotations("inst1", [
                {"instance_id": "inst1", "node_key": "a", "type": "best",
                 "mc_score": 0.8, "n_steps": 2},
                {"instance_id": "inst1", "node_key": "b", "type": "leaf",
                 "mc_score": 0.0, "n_steps": 3},
            ])
            store.set_instance_status("inst1", "done", root_mc=0.4,
                                      messages_head=[{"role": "system", "content": "s"}])
            status = store.get_instance_status("inst1")
            assert status["status"] == "done"
            assert status["root_mc"] == 0.4
            assert status["messages_head"] == [{"role": "system", "content": "s"}]
            assert store.counts()["annotations"] == 2
        finally:
            store.close()


class TestConcurrentWrites:
    def test_thread_safe_upserts(self, workdir):
        store = StateStore(workdir / "state.db")
        try:
            n_threads, per_thread = 8, 10
            barrier = threading.Barrier(n_threads)
            errors: list = []

            def worker(tid):
                try:
                    barrier.wait()
                    for i in range(per_thread):
                        store.upsert_rollout(_result(
                            iid=f"inst{tid}", node_key="root", idx=i))
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(t,))
                       for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == []
            assert store.count_rollouts() == n_threads * per_thread
        finally:
            store.close()
