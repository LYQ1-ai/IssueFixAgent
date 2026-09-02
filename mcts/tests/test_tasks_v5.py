# SPDX-License-Identifier: BSD-3-Clause

"""TreeDriver v5 会话式驱动单元测试（施工文件 05 §4/§5）。

覆盖：从零初始化、root 门控、轮数上限、resume（done 跳过 / budget_exhausted 续跑）、
中断会话重做不重复执行、孤儿节点不污染候选池、force rerun 清库、预算熔断。
全部离线（ScriptedExecutor，无 Docker/LLM）。
"""

import asyncio
from pathlib import Path

from mcts.store import StateStore
from mcts.tasks import Budget, RolloutResult, Stats, TaskQueue, TreeDriver, Worker

from mcts.tests.helpers import REARTE_R_SEQUENCES, ScriptedExecutor, make_instance

N = 5


class DriverHarness:
    """每 run 一个 worker + 共享 store/executor 的驱动测试台。"""

    def __init__(self, tmp_path, sequences=REARTE_R_SEQUENCES, *,
                 budget=None, resume=True, max_iterations=20):
        self.db_path = Path(tmp_path) / "state.db"
        self.store = StateStore(self.db_path)
        self.queue = TaskQueue()
        self.stats = Stats()
        self.sem = asyncio.Semaphore(6)
        self.executor = ScriptedExecutor(list(sequences))
        self.budget = budget or Budget()
        self.resume = resume
        self.max_iterations = max_iterations

    async def run_driver(self, instance, *, resume=None, max_iterations=None,
                         budget=None):
        worker = Worker(self.queue, self.executor, self.sem, self.stats)
        wt = asyncio.create_task(worker.run())
        driver = TreeDriver(
            instance,
            queue=self.queue, stats=self.stats,
            budget=budget if budget is not None else self.budget,
            out_dir=self.db_path.parent, store=self.store,
            n_rollouts=N,
            max_iterations=max_iterations if max_iterations is not None
            else self.max_iterations,
            resume=self.resume if resume is None else resume,
            config_file="config/mini_submit.yaml",
        )
        try:
            return await driver.run()
        finally:
            worker.shutdown.set()
            await wt

    def executed_keys(self) -> set:
        return {(k, idx) for k, idx in self.executor.executed}


def _add_fake_rollouts(store, iid, node_key, n=N, correct_flags=None):
    flags = correct_flags if correct_flags is not None else [1] * n
    for i in range(n):
        r = RolloutResult(
            instance_id=iid, node_key=node_key, rollout_idx=i,
            reward=1.0 if flags[i] else 0.0, correct=bool(flags[i]),
            steps=[], exit_status="Submitted" if flags[i] else "LimitsExceeded",
        )
        store.upsert_rollout(r)


def _run(coro):
    return asyncio.run(coro)


class TestDriverFlow:
    def test_from_zero_done(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path)
            res = await h.run_driver(make_instance("i1"))
            assert res["status"] == "done"
            tree = h.store.get_tree("i1")
            assert tree["status"] == "done" and tree["root_mc"] is not None
            assert tree["n_rounds"] >= 1
            root_row = h.store.get_node_row("i1", "root")
            assert root_row["in_pool"] == 1
            assert h.store.count_rollouts("i1") >= N
            st = h.store.load_tree_state("i1")
            assert all(n["mc_score"] is not None
                       for n in st["nodes"].values())
        _run(t())

    def test_root_all_correct_no_search(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, sequences=[([1] * N, [1] * N)])
            res = await h.run_driver(make_instance("i1"))
            assert res["status"] == "done"
            tree = h.store.get_tree("i1")
            assert tree["n_rounds"] == 0 and tree["root_mc"] == 1.0
            assert h.store.count_rollouts("i1") == N
        _run(t())

    def test_rounds_cap(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, max_iterations=2)
            res = await h.run_driver(make_instance("i1"))
            assert res["status"] == "done"
            assert h.store.get_tree("i1")["n_rounds"] == 2
        _run(t())

    def test_done_skipped_on_resume(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, max_iterations=2)
            await h.run_driver(make_instance("i1"))
            before = h.store.counts()
            res2 = await h.run_driver(make_instance("i1"), resume=True)
            assert res2["skipped"] is True and res2["status"] == "done"
            assert h.store.counts() == before
        _run(t())


class TestResume:
    def test_budget_exhausted_then_resume_continues(self, tmp_path):
        async def t():
            h = DriverHarness(
                tmp_path, budget=Budget(max_rollouts_per_instance=N + 2),
                max_iterations=5)
            res1 = await h.run_driver(make_instance("i1"))
            assert res1["status"] == "budget_exhausted"
            n_first = h.store.count_rollouts("i1")
            h2 = DriverHarness(tmp_path, max_iterations=5,
                               budget=Budget(max_rollouts_per_instance=500))
            res2 = await h2.run_driver(make_instance("i1"), resume=True)
            assert res2["status"] == "done"
            assert h2.store.count_rollouts("i1") > n_first
            assert h2.store.get_tree("i1")["n_rounds"] >= 1
        _run(t())

    def test_interrupted_session_redo_never_reruns_completed_probes(
            self, tmp_path):
        async def t():
            h = DriverHarness(
                tmp_path, budget=Budget(max_rollouts_per_instance=N + 2),
                max_iterations=5)
            await h.run_driver(make_instance("i1"))
            executed_a = h.executed_keys()
            h2 = DriverHarness(
                tmp_path, max_iterations=5,
                budget=Budget(max_rollouts_per_instance=500))
            await h2.run_driver(make_instance("i1"), resume=True)
            seen: dict[str, set[int]] = {}
            for k, idx in h2.executed_keys():
                seen.setdefault(k, set()).add(idx)
            for k, idxs in seen.items():
                if all((k, j) in executed_a for j in range(N)):
                    assert idxs <= {j for j in range(N)
                                    if (k, j) not in executed_a}, \
                        f"completed probe {k} fully re-executed: {idxs}"
        _run(t())

    def test_orphan_probe_never_pollutes_pool(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, max_iterations=1)
            await h.run_driver(make_instance("i1"))
            # 注入"中断会话残留"：孤儿节点（mc=0.6，rollout 齐）但 in_pool=0
            h.store.ensure_node("i1", "n_orphan",
                                [{"assistant": {"content": "orphan"}}])
            _add_fake_rollouts(h.store, "i1", "n_orphan",
                               correct_flags=[1, 1, 1, 0, 0])
            h.store.set_node_ready("i1", "n_orphan", mc_score=0.6)

            h2 = DriverHarness(tmp_path, max_iterations=5)
            res = await h2.run_driver(make_instance("i1"), resume=True)
            assert res["status"] == "done"
            row = h2.store.get_node_row("i1", "n_orphan")
            assert row["in_pool"] == 0
            consumed = {r["rollout_idx"] for r in
                        h2.store.load_node_rollout_rows("i1", "n_orphan")
                        if r["is_consumed"]}
            assert consumed == set()
        _run(t())

    def test_force_rerun_clears_instance(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, max_iterations=2)
            await h.run_driver(make_instance("i1"))
            n_before = h.store.count_rollouts("i1")
            assert n_before > 0
            h2 = DriverHarness(tmp_path, max_iterations=2)
            res = await h2.run_driver(make_instance("i1"), resume=False)
            assert res["status"] == "done"
            assert h2.store.count_rollouts("i1") <= n_before
        _run(t())


class TestBudget:
    def test_global_budget_limits_rollouts(self, tmp_path):
        async def t():
            h = DriverHarness(tmp_path, budget=Budget(max_rollouts=6),
                              max_iterations=5)
            res = await h.run_driver(make_instance("i1"))
            assert res["status"] in ("done", "budget_exhausted")
            assert h.store.count_rollouts("i1") <= 6
        _run(t())
