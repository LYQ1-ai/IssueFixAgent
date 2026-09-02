# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.tasks`` 任务驱动高并发引擎测试（FakeExecutor，离线可跑）。

覆盖：队列去重/优先级、节点内 N 次 rollout 并发（受全局并发上限约束）、
ReARTeR docs/01 §11.1–11.9 全流程标注、MC 门控、预算熔断、断点续跑
（实例级跳过 + 节点级 rollout 缓存复用）。
"""

import asyncio
import threading
import time

from mcts.store import StateStore
from mcts.tasks import (
    Budget,
    EnvFactory,
    MCTSPipeline,
    RolloutResult,
    RolloutTask,
    TaskQueue,
)

from mcts.tests.helpers import (
    REARTE_R_SEQUENCES,
    ScriptedExecutor,
    make_instance,
    make_step,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


class CountingExecutor:
    """带活跃计数器的 executor：验证并发语义（活跃数 ≤ 全局上限且 > 1）。"""

    def __init__(self, delay: float = 0.05):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.executed: list[str] = []
        self.delay = delay

    def run(self, task):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.executed.append(task.task_id)
        time.sleep(self.delay)
        steps = [make_step(f"{task.node_key}#{task.rollout_idx}#{j}") for j in range(2)]
        with self.lock:
            self.active -= 1
        return RolloutResult(
            instance_id=task.instance_id, node_key=task.node_key,
            rollout_idx=task.rollout_idx, correct=True, reward=1.0, steps=steps,
            n_calls=2, duration=self.delay,
        )


def run_pipeline(instances, executor, out_dir, **kwargs) -> dict:
    env_factory = EnvFactory(creation_concurrency=2)  # 测试不真正创建容器
    pipeline = MCTSPipeline(
        instances,
        executor_factory=lambda ef: executor,
        env_factory=env_factory,
        out_dir=out_dir,
        max_concurrency=kwargs.pop("max_concurrency", 4),
        n_rollouts=kwargs.pop("n_rollouts", 5),
        budget=kwargs.pop("budget", Budget()),
        resume=kwargs.pop("resume", False),
        seed=kwargs.pop("seed", 1),
        max_iterations=kwargs.pop("max_iterations", 20),
        stats_interval=999.0,
    )
    return asyncio.run(pipeline.run())


def open_store(out_dir) -> StateStore:
    """打开（或新建）该工作目录的 SQLite 存储，读取后关闭。"""
    return StateStore(out_dir / "state.db")


def load_annotations(out_dir, iid="inst1") -> list[dict]:
    store = open_store(out_dir)
    try:
        with store._write_lock:
            rows = store._conn.execute(
                "SELECT node_key, type, mc_score, n_steps FROM annotations"
                " WHERE instance_id=? ORDER BY written_at", (iid,),
            ).fetchall()
        return [{"node_key": r[0], "type": r[1], "mc_score": r[2], "n_steps": r[3]}
                for r in rows]
    finally:
        store.close()


def load_state(out_dir, iid="inst1") -> dict:
    store = open_store(out_dir)
    try:
        return store.get_instance_status(iid) or {}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------


async def _queue_dedup():
    q = TaskQueue()
    t = RolloutTask("i1", "root", 0, "root", 0, {})
    f1 = await q.submit(t)
    f2 = await q.submit(t)
    assert f1 is f2                       # 去重：同一任务返回同一 future
    q.complete(t, RolloutResult("i1", "root", 0))
    f3 = await q.submit(t)
    assert f3 is not f1                   # 完成后可重新提交


async def _queue_priority():
    q = TaskQueue()
    probe = RolloutTask("i1", "k", 0, "probe", 1, {})
    root = RolloutTask("i2", "root", 0, "root", 0, {})
    await q.submit(probe)
    await q.submit(root)
    got = await q.get()
    assert got.task_id == root.task_id    # root 优先级更高


def test_queue_dedup():
    asyncio.run(_queue_dedup())


def test_queue_priority():
    asyncio.run(_queue_priority())


# ---------------------------------------------------------------------------
# 并发语义
# ---------------------------------------------------------------------------


def test_node_rollouts_run_concurrently_within_global_cap(workdir):
    exe = CountingExecutor(delay=0.08)
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir,
                           max_concurrency=3)
    assert summary["stats"]["done"] == 5          # 根节点 5 次 rollout
    assert len(exe.executed) == 5
    assert 2 <= exe.max_active <= 3               # 并发确实发生且受全局上限约束


def test_multiple_trees_run_concurrently(workdir):
    exe = CountingExecutor(delay=0.05)
    insts = [make_instance(f"inst{i}") for i in range(4)]
    summary = run_pipeline(insts, executor=exe, out_dir=workdir,
                           max_concurrency=4, n_rollouts=2)
    # 4 棵树 × 2 root rollout（全对 → 无树搜索）
    assert summary["stats"]["done"] == 8
    assert exe.max_active >= 3                    # 多树并行
    assert summary["statuses"] == {"done": 4}


# ---------------------------------------------------------------------------
# ReARTeR 数值示例全流程（M2 验收核心）
# ---------------------------------------------------------------------------


def test_rearter_full_flow_via_pipeline(workdir):
    # max_iterations=4：正好覆盖 docs/01 §11.1–11.6 的 7 个节点与 5 条 best 标注
    exe = ScriptedExecutor(REARTE_R_SEQUENCES)
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir,
                           max_iterations=4)
    assert summary["instances"][0]["status"] == "done"
    assert summary["instances"][0]["root_mc"] == 0.4
    # root 5 + 6 个探测节点 × 5 = 35 次 rollout
    assert summary["stats"]["done"] == 35
    assert len(exe.by_key) == 7                   # root + n1..n6

    ann = load_annotations(workdir)
    bests = [e for e in ann if e["type"] == "best"]
    leaves = [e for e in ann if e["type"] == "leaf"]
    adds = [e for e in ann if e["type"] == "add"]

    # 标注顺序（对齐 docs/01 §11.2–11.6）：root(无标注) → n1 ×3 → n5 ×2
    b1, b2, b3, b4, b5 = (bests[i]["node_key"] for i in range(5))
    assert b1 == b2 == b3                         # n1 连续三次被选（同前缀多 rollout 佐证）
    assert b4 == b5 != b1                         # 探索项把 n5 推上榜首
    assert bests[0]["mc_score"] == 0.8
    assert bests[3]["mc_score"] == 0.6

    mc_of_add = {e["node_key"]: e["mc_score"] for e in adds}
    assert set(mc_of_add.values()) == {0.8, 0.2, 0.6}   # n1 / n2 / n5
    assert len(leaves) == 2
    assert all(l["mc_score"] == 0.0 for l in leaves)    # n4 / n6（首个错误步负样本）
    assert load_state(workdir)["status"] == "done"


# ---------------------------------------------------------------------------
# MC 门控
# ---------------------------------------------------------------------------


def test_gate_skips_search_when_all_correct(workdir):
    exe = ScriptedExecutor([([1] * 5, [1] * 5)])
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir)
    assert summary["stats"]["done"] == 5
    assert load_annotations(workdir) == []       # 全对 → 无标注


def test_gate_skips_search_when_all_wrong(workdir):
    exe = ScriptedExecutor([([0] * 5, [1] * 5)])
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir)
    assert summary["stats"]["done"] == 5
    assert load_annotations(workdir) == []       # 全错 → 无标注


def test_head_rebuilt_from_instance_when_trajectory_missing(workdir):
    """root rollout 轨迹缺失（模拟 resume 后 trajectory=None）→ head 由
    config 提示词 + 实例输入重建（build_messages_head）→ probe 正常扩展。

    修复 2026-08-31：head 不再依赖轨迹 / 历史落库；任何实例可还原头部。
    """
    exe = ScriptedExecutor(REARTE_R_SEQUENCES, include_trajectory=False)
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir,
                           max_iterations=4)
    inst = summary["instances"][0]
    assert inst["status"] == "done"
    assert inst["root_mc"] == 0.4                 # root MC 保留
    assert summary["stats"]["done"] == 35         # root 5 + 6 个探测节点 × 5
    assert len(exe.by_key) == 7                   # root + n1..n6
    assert len(load_annotations(workdir)) > 0     # 正常产出标注


class _OnceFailingExecutor:
    """第 1 次调用返回 error（模拟连接抖动），之后全部成功。"""

    def __init__(self):
        self.calls = 0
        self.failed_first = True

    def run(self, task):
        self.calls += 1
        if self.failed_first:
            self.failed_first = False
            return RolloutResult(
                instance_id=task.instance_id, node_key=task.node_key,
                rollout_idx=task.rollout_idx,
                error="Connection error (simulated)",
            )
        steps = [make_step(f"{task.node_key}#{task.rollout_idx}#0")]
        return RolloutResult(
            instance_id=task.instance_id, node_key=task.node_key,
            rollout_idx=task.rollout_idx, correct=True, reward=1.0, steps=steps,
            trajectory={"messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "u"}],
                        "trajectory_format": "mini-swe-agent-1.1"},
        )


def test_failed_rollout_retried_and_not_persisted_when_still_failing(workdir):
    """失败 rollout：运行中自动重跑一次；重跑成功则落库凑满 N；
    重跑仍失败则不落库（视为没跑过，resume 槽位为 None 再补跑）。"""
    # 场景 1：第一次失败、重跑成功 → 首跑就凑满 5 条，DB 无失败记录
    exe1 = _OnceFailingExecutor()
    summary1 = run_pipeline([make_instance()], executor=exe1, out_dir=workdir)
    store = open_store(workdir)
    try:
        n_after_first = store.count_rollouts("inst1")
    finally:
        store.close()
    assert n_after_first == 5                     # 失败槽位被树级重跑补齐
    assert exe1.calls == 6                        # 5 次初始 + 1 次树级重跑
    assert summary1["instances"][0]["status"] == "done"


def test_failed_rollout_never_persisted_when_always_failing(workdir):
    """重跑仍失败 → 不落库（DB 无失败记录）；树因无有效 rollout 标记 failed。"""
    class _AlwaysFail:
        def __init__(self):
            self.calls = 0

        def run(self, task):
            self.calls += 1
            return RolloutResult(
                instance_id=task.instance_id, node_key=task.node_key,
                rollout_idx=task.rollout_idx, error="always failing",
            )

    exe = _AlwaysFail()
    summary = run_pipeline([make_instance()], executor=exe, out_dir=workdir)
    assert summary["instances"][0]["status"] == "failed"
    store = open_store(workdir)
    try:
        assert store.count_rollouts("inst1") == 0  # 失败结果从未落库
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 预算熔断
# ---------------------------------------------------------------------------


def test_global_budget_partial_submission(workdir):
    insts = [make_instance(f"inst{i}") for i in range(2)]
    exe = CountingExecutor(delay=0)
    budget = Budget(max_rollouts=3)
    summary = run_pipeline(insts, executor=exe, out_dir=workdir, budget=budget)
    assert summary["stats"]["submitted"] == 3
    assert len(exe.executed) == 3
    # 第一个实例拿到 3 次（全对 → done）；第二个实例 0 配额 → budget_exhausted
    assert summary["statuses"] == {"done": 1, "budget_exhausted": 1}


def test_per_instance_budget(workdir):
    insts = [make_instance(f"inst{i}") for i in range(2)]
    exe = CountingExecutor(delay=0)
    budget = Budget(max_rollouts_per_instance=3)
    summary = run_pipeline(insts, executor=exe, out_dir=workdir, budget=budget)
    assert summary["stats"]["submitted"] == 6     # 每实例 3 次
    assert summary["statuses"] == {"done": 2}


# ---------------------------------------------------------------------------
# 断点续跑
# ---------------------------------------------------------------------------


def test_resume_skips_done_instances(workdir):
    seq = [([1] * 5, [1] * 5)]
    run_pipeline([make_instance()], executor=ScriptedExecutor(seq),
                 out_dir=workdir, resume=False)

    exe2 = ScriptedExecutor(seq)
    summary = run_pipeline([make_instance()], executor=exe2, out_dir=workdir,
                           resume=True)
    assert summary["instances"][0]["skipped"] is True
    assert exe2.executed == []                    # 已完成实例零重跑


def test_resume_reuses_cached_node_rollouts(workdir):
    run_pipeline([make_instance()], executor=ScriptedExecutor(REARTE_R_SEQUENCES),
                 out_dir=workdir, resume=False, max_iterations=4)
    # 模拟中断：实例状态置回 running（未 done），rollout 缓存保留在 SQLite
    store = open_store(workdir)
    try:
        store.set_instance_status("inst1", "running")
    finally:
        store.close()

    exe2 = ScriptedExecutor(REARTE_R_SEQUENCES)
    summary = run_pipeline([make_instance()], executor=exe2, out_dir=workdir,
                           resume=True, max_iterations=4)
    assert summary["instances"][0]["status"] == "done"
    assert exe2.executed == []                    # 全部节点 rollout 从 SQLite 复用
    assert len(load_annotations(workdir)) > 0    # 标注重新落库


# ---------------------------------------------------------------------------
# SQLite 持久化
# ---------------------------------------------------------------------------


def test_sqlite_persistence_roundtrip(workdir):
    run_pipeline([make_instance()], executor=ScriptedExecutor(REARTE_R_SEQUENCES),
                 out_dir=workdir, resume=False, max_iterations=4)
    store = open_store(workdir)
    try:
        assert store.count_rollouts("inst1") == 35
        # 节点级读取（断点续跑复用依据）：root 与探测节点的 rollout 都能还原
        root_slots = store.load_node_rollouts("inst1", "root", 5)
        assert all(r is not None and r.correct in (True, False) for r in root_slots)
        assert store.get_instance_status("inst1")["status"] == "done"
        # result_json 不含大块轨迹（DB 体积控制）
        with store._write_lock:
            row = store._conn.execute(
                "SELECT result_json FROM rollouts WHERE instance_id='inst1'"
                " AND node_key='root' AND rollout_idx=0").fetchone()
        import json as _json

        payload = _json.loads(row[0])
        assert "trajectory" not in payload
        assert "steps" in payload and payload["reward"] in (0.0, 1.0)
    finally:
        store.close()
