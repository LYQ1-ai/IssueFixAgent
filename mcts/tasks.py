# SPDX-License-Identifier: BSD-3-Clause

"""任务驱动高并发 MCTS 引擎（PLAN §2.3–2.4 / 设计文档 docs/mcts_engine_design.md）。

**核心思想（任务驱动，v2）**：

- **RolloutTask** 是最小调度单元，payload 自包含（instance / 前缀步 / 前缀消息 /
  温度）—— 树逻辑（TreeDriver）与执行细节（Worker + Executor）完全解耦；
- **TaskQueue**：``asyncio.PriorityQueue`` + future 注册表；``submit(task) -> Future``
  按 task_id 去重（幂等 —— 断点续跑 / 并发扩展的安全保证）；
- **Worker 池**：每个 worker 一个 asyncio Task，``to_thread`` 执行阻塞的容器 +
  Agent 调用；全局信号量限流（``max_concurrency``）；
- **TreeDriver**（每实例一个）：root rollout × N → MC 门控 → select/locate 二分
  （每次 locate 试探 = 该节点 N 次 rollout **并发**提交并等待）；
- **EnvFactory**：容器**创建即用、用完即销毁**（不复用、无 repo+commit key、
  不限制总创建次数），只限**并发创建数**（``creation_concurrency`` 信号量）；
- **SQLite 持久化**（``mcts.store.StateStore``）：树结构与 rollout 结果**常驻内存**，
  完成即落库（rollout 单条事务）+ 定期快照（树节点）+ 实例状态/标注 —— 崩溃安全的
  断点恢复事实源（恢复代码后续里程碑实现）；
- **可靠性**：重试降级（失败 rollout 不计入 N）、预算熔断、Stats 可观测。

本模块**不 import ``agent`` / ``minisweagent``**（executor 由 :mod:`mcts.executor`
提供并在 pipeline 构造时注入 —— 单测用 FakeExecutor 完全离线跑通全流程）。
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from mcts.locate import annotation_entry, locate_error
from mcts.node import MCTSNode, select_best_node
from mcts.steps import Step, messages_for_prefix, prefix_node_key, steps_to_messages

logger = logging.getLogger("mcts.tasks")


def messages_head_from_rollouts(rollouts: list) -> list[dict]:
    """从 root rollout 列表取第一个含完整头部（system+user）的轨迹头部。

    失败 rollout（``trajectory=None`` / 空 messages）跳过；全部不可用返回 []。
    修复 2026-08-31：原实现只看 ``rollouts[0]``，若第 0 个 root rollout 失败
    （如 sglang 重启期的 Connection error），probe 的 ``prefix_messages`` 会缺
    system+user → 请求 messages 无 user → sglang 400 "No user query found"。
    """
    for rollout in rollouts or []:
        traj = getattr(rollout, "trajectory", None)
        if traj and traj.get("messages"):
            head = messages_for_prefix(traj["messages"], 0)
            if head:
                return [dict(m) for m in head]
    return []


def resolve_messages_head(rollouts: list, stored_head: Optional[list] = None) -> list[dict]:
    """解析 probe 前缀的 system+user 头部（修复 2026-08-31 resume 缺陷）。

    优先从 rollout 轨迹计算（首跑路径）；计算不出（如 resume 续跑时 rollout 从
    DB 加载、``trajectory=None``）则回退到落库的 ``instances.messages_head_json``；
    仍无则返回 []（probe 由 run_probe 兜底拒绝，不发坏请求）。
    """
    head = messages_head_from_rollouts(rollouts)
    if head:
        return head
    if stored_head:
        return [dict(m) for m in stored_head]
    return []


def build_messages_head(instance: Any, config_file: str = "config/mini_submit.yaml") -> list[dict]:
    """从 config 提示词 + 实例原始输入直接构造 system+user 头部（不依赖轨迹）。

    修复 2026-08-31：head 不应依赖 root rollout 轨迹（resume 后 trajectory 丢失、
    历史实例 head 未落库都曾导致 head 缺失）——头部本应由**固定提示词 + 任务描述**
    决定，与 rollout 轨迹无关：

    - ``system`` = ``config/mini_submit.yaml`` 的 ``agent.system_template`` 渲染
      （``{{system}}/{{release}}/{{version}}/{{machine}}`` 用 ``platform.uname()``
      提供，与容器内 ``DockerEnvironment.get_template_vars`` 同源）；
    - ``user`` = ``agent.instance_template`` 渲染（``{{task}}`` = 实例
      ``problem_statement``，即原始 issue 描述）。

    这样任何实例（含历史 head 缺失的）都能稳定还原对话头部。
    """
    import platform  # noqa: PLC0415 - 惰性

    from jinja2 import StrictUndefined, Template  # noqa: PLC0415
    from minisweagent.config import get_config_from_spec  # noqa: PLC0415

    cfg = get_config_from_spec(config_file) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    system_t = Template(agent_cfg.get("system_template", "") or "",
                        undefined=StrictUndefined)
    instance_t = Template(agent_cfg.get("instance_template", "") or "",
                          undefined=StrictUndefined)
    env_vars = platform.uname()._asdict()  # system/release/version/machine
    system = system_t.render(**env_vars)
    user = instance_t.render(task=getattr(instance, "problem_statement", ""))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 任务 / 结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutTask:
    """一次 rollout 的最小调度单元。"""

    instance_id: str
    node_key: str
    rollout_idx: int
    kind: str                      # "root" | "probe"
    priority: int                  # 0 = root（最高），1 = probe
    payload: dict                  # {"instance", "prefix_steps", "prefix_messages", "temperature"}

    @property
    def task_id(self) -> str:
        return f"{self.instance_id}:{self.node_key}:{self.rollout_idx}"


@dataclass
class RolloutResult:
    """一次 rollout 的执行结果（含树逻辑需要的步序列与终态回报）。"""

    instance_id: str
    node_key: str
    rollout_idx: int
    reward: float = 0.0
    correct: bool = False
    steps: list = field(default_factory=list)          # list[Step]（续跑步）
    trajectory: Optional[dict] = None                  # 轻量轨迹（去 response 大块）
    exit_status: str = ""
    submission: str = ""
    n_calls: int = 0
    cost: float = 0.0
    duration: float = 0.0
    replay_drift: bool = False
    trace_id: Optional[str] = None
    reward_details: Optional[dict] = None   # 分层判定细节（depth 分布/M/N/C/匹配对等，分析用）
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return bool(self.error)

    def to_dict(self, include_trajectory: bool = True) -> dict:
        """序列化。``include_trajectory=False`` 供 SQLite 落库（轨迹可从
        steps + submission/exit 字段推导，不存大块消息，控制 DB 体积）。"""
        d = {
            "instance_id": self.instance_id,
            "node_key": self.node_key,
            "rollout_idx": self.rollout_idx,
            "reward": self.reward,
            "correct": self.correct,
            "steps": [s.to_json() if isinstance(s, Step) else s for s in self.steps],
            "exit_status": self.exit_status,
            "submission": self.submission,
            "n_calls": self.n_calls,
            "cost": self.cost,
            "duration": self.duration,
            "replay_drift": self.replay_drift,
            "trace_id": self.trace_id,
            "reward_details": self.reward_details,
            "error": self.error,
        }
        if include_trajectory:
            d["trajectory"] = self.trajectory
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RolloutResult":
        return cls(
            instance_id=d.get("instance_id", ""),
            node_key=d.get("node_key", ""),
            rollout_idx=int(d.get("rollout_idx", 0)),
            reward=float(d.get("reward", 0.0)),
            correct=bool(d.get("correct", False)),
            steps=[Step.from_json(s) for s in d.get("steps", [])],
            trajectory=d.get("trajectory"),
            exit_status=d.get("exit_status", ""),
            submission=d.get("submission", ""),
            n_calls=int(d.get("n_calls", 0)),
            cost=float(d.get("cost", 0.0)),
            duration=float(d.get("duration", 0.0)),
            replay_drift=bool(d.get("replay_drift", False)),
            trace_id=d.get("trace_id"),
            reward_details=d.get("reward_details"),
            error=d.get("error"),
        )


class RolloutExecutor(Protocol):
    """执行一次 rollout（阻塞、线程池内调用）的协议。"""

    def run(self, task: RolloutTask) -> RolloutResult: ...


# ---------------------------------------------------------------------------
# 任务队列（asyncio 优先级队列 + future 注册表 + 去重）
# ---------------------------------------------------------------------------


class TaskQueue:
    """``submit(task) -> Future``；同 task_id 去重（幂等）。"""

    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._futures: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._seq = 0

    async def submit(self, task: RolloutTask) -> asyncio.Future:
        """提交任务；已存在未完成的任务返回同一 future（去重，不重复执行）。"""
        async with self._lock:
            fut = self._futures.get(task.task_id)
            if fut is not None and not fut.done():
                return fut
            fut = asyncio.get_running_loop().create_future()
            self._futures[task.task_id] = fut
            self._seq += 1
            await self._q.put((task.priority, self._seq, task))
            return fut

    async def get(self) -> RolloutTask:
        _prio, _seq, task = await self._q.get()
        return task

    def complete(self, task: RolloutTask, result: RolloutResult) -> None:
        fut = self._futures.get(task.task_id)
        if fut is not None and not fut.done():
            fut.set_result(result)

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    async def join(self) -> None:
        await self._q.join()


# ---------------------------------------------------------------------------
# 统计 / 预算
# ---------------------------------------------------------------------------


class Stats:
    """跨线程的进度统计（worker 线程写、主循环读）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.submitted = 0
        self.done = 0
        self.failed = 0
        self.n_calls = 0
        self.cost = 0.0
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self.start = time.time()

    def on_submit(self, n: int = 1) -> None:
        with self._lock:
            self.submitted += n

    def on_done(self, result: RolloutResult) -> None:
        with self._lock:
            self.done += 1
            if result.failed:
                self.failed += 1
            else:
                self.rewards.append(result.reward)
                self.steps.append(len(result.steps))
                self.n_calls += result.n_calls
                self.cost += result.cost

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = max(time.time() - self.start, 1e-6)
            n = len(self.rewards)
            return {
                "elapsed_s": round(time.time() - self.start, 1),
                "submitted": self.submitted,
                "done": self.done,
                "failed": self.failed,
                "fail_rate": round(self.failed / max(self.done, 1), 4),
                "throughput_rollouts_min": round(self.done / elapsed * 60, 2),
                "avg_reward": round(sum(self.rewards) / n, 4) if n else 0.0,
                "avg_steps": round(sum(self.steps) / n, 2) if n else 0.0,
                "total_llm_calls": self.n_calls,
                "total_cost": round(self.cost, 4),
            }


class BudgetExhausted(Exception):
    """预算耗尽：树停止扩展新节点（保留已生成数据）。"""


class Budget:
    """全局 / 每实例 rollout 数预算 + LLM 调用 / 成本软预算。

    :meth:`allowed_count` 返回当前还能提交的 rollout 数（受全局与每实例上限约束），
    树驱动按此**部分提交**（预算将尽时节点只跑允许数量的 rollout，MC 照常计算）——
    比"一次性拒绝整批"更平滑，保证预算边界处不浪费已产生数据。
    """

    def __init__(
        self,
        *,
        max_rollouts: int = 0,
        max_rollouts_per_instance: int = 0,
        max_llm_calls: int = 0,
        max_cost: float = 0.0,
    ) -> None:
        self.max_rollouts = max_rollouts
        self.max_rollouts_per_instance = max_rollouts_per_instance
        self.max_llm_calls = max_llm_calls
        self.max_cost = max_cost
        self._per_instance: dict[str, int] = {}

    def allowed_count(self, instance_id: str, stats: Stats, n: int) -> int:
        """当前还能提交的 rollout 数（≤ n）；0 = 预算耗尽。"""
        allowed = n
        if self.max_rollouts:
            allowed = min(allowed, self.max_rollouts - stats.submitted)
        if self.max_rollouts_per_instance:
            allowed = min(allowed, self.max_rollouts_per_instance
                          - self._per_instance.get(instance_id, 0))
        # LLM 调用 / 成本为软预算（无法预知单次调用数）：已超即不再提交
        if self.max_llm_calls and stats.n_calls >= self.max_llm_calls:
            return 0
        if self.max_cost and stats.cost >= self.max_cost:
            return 0
        return max(0, allowed)

    def record_submit(self, instance_id: str, n: int = 1) -> None:
        self._per_instance[instance_id] = self._per_instance.get(instance_id, 0) + n


# ---------------------------------------------------------------------------
# 容器工厂（v2：创建即用、用完即销毁；不复用、无 repo+commit key、不限制总创建次数）
# ---------------------------------------------------------------------------


class EnvFactory:
    """执行环境容器工厂：只负责**容器创建与环境准备**，销毁交回调用方。

    - ``create_env(instance)``：每次调用都新建一个独立容器（EnvManager v2 语义），
      容器名含 uuid 天然互不冲突；不做任何池化 / 复用 / key 分组；
    - ``destroy_env(container)``：``docker rm -f`` 幂等删除（executor 的 finally）；
    - **并发创建节流**：``creation_concurrency`` 信号量只限同时创建的容器数
      （默认 8，防 40–50 路 worker 同时 docker run + unzip 的瞬时 IO 风暴），
      **不限制总创建次数**；
    - ``exec(container, cmd)``：容器内执行命令（rollout 后取 patch 用）。
    """

    def __init__(
        self,
        manager: Any = None,
        *,
        creation_concurrency: int = 8,
        env_manager_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.mgr = manager
        self._mgr_factory = env_manager_factory or self._default_manager_factory
        self._create_sem = threading.Semaphore(max(1, int(creation_concurrency)))
        self.created = 0

    @staticmethod
    def _default_manager_factory() -> Any:
        from agent.init_env import EnvManager  # 惰性：纯逻辑测试不拉起 agent

        return EnvManager()

    def _get_manager(self) -> Any:
        if self.mgr is None:
            self.mgr = self._mgr_factory()
        return self.mgr

    def create_env(self, instance: Any) -> str:
        """按实例创建**全新**容器并完成环境准备（bug 状态），返回容器名。"""
        from mcts.env import instance_env_params  # 惰性

        params = instance_env_params(instance)
        with self._create_sem:  # 只限并发创建数，不限总次数
            container = self._get_manager().get_env(
                params.repo, params.commit, patch=params.patch)
        self.created += 1
        return container

    def destroy_env(self, container: str) -> None:
        """销毁容器（docker rm -f，幂等；rollout 完成后调用）。"""
        self._get_manager().release_env(container)

    def exec(self, container: str, command: str, *, input_text: Optional[str] = None) -> str:
        """在容器内执行命令（取 agent 改动 patch 等）。"""
        r = self._get_manager()._exec(container, command, input_text=input_text)
        return r.stdout or ""

    def stats(self) -> dict:
        return {"created": self.created,
                "creation_concurrency": self._create_sem._value if hasattr(
                    self._create_sem, "_value") else 0}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class Worker:
    """消费队列的 worker：``to_thread`` 执行阻塞 rollout + 全局信号量限流。"""

    def __init__(
        self,
        queue: TaskQueue,
        executor: RolloutExecutor,
        semaphore: asyncio.Semaphore,
        stats: Stats,
        *,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> None:
        self.queue = queue
        self.executor = executor
        self.semaphore = semaphore
        self.stats = stats
        self.retries = retries
        self.backoff = backoff
        self.shutdown = asyncio.Event()

    async def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue
            result: Optional[RolloutResult] = None
            try:
                async with self.semaphore:
                    result = await self._execute_with_retry(task)
            except Exception as e:  # noqa: BLE001 - 兜底：任何异常都产出失败结果
                logger.error("rollout %s crashed: %s", task.task_id, e)
                result = RolloutResult(
                    instance_id=task.instance_id, node_key=task.node_key,
                    rollout_idx=task.rollout_idx, error=f"worker crash: {e}",
                )
            finally:
                if result is not None:
                    self.queue.complete(task, result)
                    self.stats.on_done(result)
                self.queue.task_done()

    async def _execute_with_retry(self, task: RolloutTask) -> RolloutResult:
        attempts = 1 + self.retries
        for attempt in range(attempts):
            try:
                return await asyncio.to_thread(self.executor.run, task)
            except Exception as e:  # noqa: BLE001 - 重试后仍失败由调用方兜底
                logger.warning("rollout %s attempt %d/%d failed: %s",
                               task.task_id, attempt + 1, attempts, e)
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.backoff ** attempt)
        return RolloutResult(instance_id=task.instance_id, node_key=task.node_key,
                             rollout_idx=task.rollout_idx, error="retries exhausted")


# ---------------------------------------------------------------------------
# 树驱动（每实例一棵树）
# ---------------------------------------------------------------------------


class TreeDriver:
    """单实例的 MCTS 全流程：root rollout → MC 门控 → select/locate 标注循环。

    与 ReARTeR ``gen_data_Step_RFT.py`` + ``module.py::process_annotations`` 对应，
    但 rollout 经任务队列并发执行（probe 节点 N 次 rollout 同时进行）。

    **树结构与 rollout 结果常驻内存**；SQLite（``StateStore``）作为崩溃安全镜像：
    rollout 完成即 upsert、树节点定期快照、实例状态与标注落库（断点恢复用）。
    """

    def __init__(
        self,
        instance: Any,
        *,
        queue: TaskQueue,
        stats: Stats,
        budget: Budget,
        out_dir: Path,
        store: Optional["StateStore"] = None,
        n_rollouts: int = 5,
        temperature_range: tuple[float, float] = (0.7, 1.0),
        select_kwargs: Optional[dict] = None,
        max_iterations: int = 20,
        resume: bool = True,
        config_file: str = "config/mini_submit.yaml",
    ) -> None:
        from mcts.store import StateStore  # 惰性

        self.instance = instance
        self.iid = instance.instance_id
        self.queue = queue
        self.stats = stats
        self.budget = budget
        self.out_dir = out_dir
        self.store = store or StateStore(out_dir / "state.db")
        self.n_rollouts = n_rollouts
        self.temperature_range = temperature_range
        self.select_kwargs = select_kwargs or {}
        self.max_iterations = max_iterations
        self.resume = resume
        self.config_file = config_file
        self.nodes: dict[str, MCTSNode] = {}      # node_key → node（内容寻址）
        self.leaves: list[MCTSNode] = []
        self.expanded: list[MCTSNode] = []        # 扩展节点（add 标注 = 这些节点）
        self.best_entries: list[dict] = []
        self.status = "running"
        self._messages_head: list[dict] = []       # system+user 头部（probe 消息拼接用）
        self._tree_dirty = False                   # 树快照待写标记

    # ------------------------------------------------------------------
    # 运行入口
    # ------------------------------------------------------------------

    async def run(self) -> dict:
        state = self.store.get_instance_status(self.iid)
        if self.resume and state and state.get("status") in ("done", "failed"):
            logger.info("%s: skip (status=%s)", self.iid, state["status"])
            return {"instance_id": self.iid, "status": state["status"], "skipped": True}
        self.store.set_instance_status(self.iid, "running")
        try:
            root = self._get_or_create_node([])
            await self._perform_rollouts(root, kind="root")
            # 头部优先从 root 轨迹计算；resume 续跑时 rollout 从 DB 加载
            # （trajectory=None）算不出 → 回退到落库的 messages_head_json →
            # 最后用「config 提示词 + 实例原始输入」直接重建（任何实例可还原，
            # 不再依赖轨迹 / 历史落库，见 build_messages_head）。
            self._messages_head = (
                resolve_messages_head(
                    root.rollouts,
                    stored_head=(state or {}).get("messages_head"),
                )
                or build_messages_head(self.instance, self.config_file)
            )
            root.mc_score = root.compute_mc()
            n_correct = root.correct_count()
            logger.info("%s: root MC=%.2f (%d/%d)", self.iid, root.mc_score,
                        n_correct, root.n_rollouts)
            if root.n_rollouts == 0:
                # 根节点全部 rollout 失败（无有效数据）→ 实例标记 failed
                self.status = "failed"
                self._write_state()
                return {"instance_id": self.iid, "status": "failed",
                        "root_mc": root.mc_score, "n_nodes": len(self.nodes),
                        "reason": "no valid root rollouts"}
            if root.gated:
                if self._messages_head:
                    await self.process_annotations(root)
                else:
                    # 头部（system+user）不可用（历史实例 head 未落库 / 首跑
                    # root 全部失败）：probe 无法构造合法对话（sglang 400
                    # "No user query found"）——跳过扩展，保留 root 数据。
                    logger.warning(
                        "%s: root MC=%.2f but system+user head unavailable; "
                        "skipping probe expansion", self.iid, root.mc_score)
            self.status = "done"
            self._write_annotations()
            self._write_state()
            return {"instance_id": self.iid, "status": "done",
                    "root_mc": root.mc_score, "n_nodes": len(self.nodes),
                    "n_best": len(self.best_entries), "n_leaves": len(self.leaves)}
        except BudgetExhausted:
            logger.info("%s: budget exhausted; partial data kept", self.iid)
            self.status = "budget_exhausted"
            self._write_annotations()
            self._write_state()
            return {"instance_id": self.iid, "status": "budget_exhausted",
                    "n_nodes": len(self.nodes)}
        except Exception as e:  # noqa: BLE001 - 单实例失败不拖垮整体
            logger.exception("%s: tree failed: %s", self.iid, e)
            self.status = "failed"
            self._write_state()
            return {"instance_id": self.iid, "status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # 节点 / rollout
    # ------------------------------------------------------------------

    def _get_or_create_node(self, prefix_steps: list[Step]) -> MCTSNode:
        key = prefix_node_key(prefix_steps)
        node = self.nodes.get(key)
        if node is None:
            node = MCTSNode(instance_id=self.iid, node_key=key,
                            prefix_steps=list(prefix_steps))
            self.nodes[key] = node
        return node

    async def _perform_rollouts(self, node: MCTSNode, kind: str) -> None:
        """确保节点有 N 次有效 rollout：磁盘缓存复用 + 缺失并发提交 + 等待。

        - 节点已有 N 次有效 rollout 且 mc 已算 → 直接返回（内容寻址节点可能被
          不同父节点再次探测，复用已有结果，不重复计算）；
        - resume=True 时先加载磁盘缓存（断点续跑不重跑已完成节点）；
        - 预算将尽时按 ``Budget.allowed_count`` 部分提交。
        """
        if node.n_rollouts >= self.n_rollouts and node.mc_score is not None:
            return
        # 兜底：probe 需要 system+user 头部拼接合法对话；head 缺失时
        # （历史实例 head 未落库）不发任务，避免 sglang 400 / 成批失败。
        if kind == "probe" and not self._messages_head:
            logger.warning("%s: probe skipped (head unavailable) node=%s",
                           self.iid, node.node_key)
            return
        existing = self._load_cached(node.node_key) if self.resume else None
        if existing is None:
            existing = [None] * self.n_rollouts
        missing = [i for i, r in enumerate(existing) if r is None]
        if missing:
            allowed = self.budget.allowed_count(self.iid, self.stats, len(missing))
            if allowed <= 0:
                raise BudgetExhausted
            missing = missing[:allowed]  # 预算将尽时部分提交（节点只跑允许数量的 rollout）
            self.budget.record_submit(self.iid, len(missing))
            self.stats.on_submit(len(missing))
            temperature = random.uniform(*self.temperature_range)
            prefix_messages = None
            if kind == "probe" and node.prefix_steps:
                prefix_messages = list(self._messages_head) + steps_to_messages(node.prefix_steps)
            payload = {
                "instance": self.instance,
                "prefix_steps": list(node.prefix_steps),
                "prefix_messages": prefix_messages,
                "temperature": temperature,
            }
            priority = 0 if kind == "root" else 1
            futs = [
                await self.queue.submit(RolloutTask(
                    instance_id=self.iid, node_key=node.node_key, rollout_idx=i,
                    kind=kind, priority=priority, payload=payload,
                ))
                for i in missing
            ]
            results = await asyncio.gather(*futs)
            for i, res in zip(missing, results):
                res = res if isinstance(res, RolloutResult) else RolloutResult(
                    instance_id=self.iid, node_key=node.node_key, rollout_idx=i,
                    error=f"gather returned {type(res).__name__}",
                )
                if res.error is not None:
                    # 运行中失败自动重跑一次（worker_retries 已兜底多次，树级
                    # 再重试 1 次；修复 2026-08-31：避免失败槽位残留导致有效
                    # N 减少、MC 失真，且树 done 后 resume 不再补跑）。
                    res = await self._retry_rollout(
                        task_id=(self.iid, node.node_key, i, kind, priority),
                        payload=payload,
                    ) or res
                existing[i] = res
                # 失败 rollout **不落库**（视为没跑过）：resume 时该槽位为
                # None → 重新提交重跑（修复 2026-08-31：失败结果落库会被
                # resume 当作"已有"而永久固化，节点有效 N 减少）。
                if res.error is None:
                    self.store.upsert_rollout(res)
        node.set_rollouts([r for r in existing if r is not None])
        node.mc_score = node.compute_mc()
        self._tree_dirty = True

    async def _retry_rollout(self, task_id: tuple, payload: dict) -> Optional[RolloutResult]:
        """失败 rollout 重跑一次（同一 payload）；仍失败返回 None（调用方保留
        原失败结果，不落库、resume 时槽位为 None 再重跑）。"""
        try:
            task = RolloutTask(
                instance_id=task_id[0], node_key=task_id[1], rollout_idx=task_id[2],
                kind=task_id[3], priority=task_id[4], payload=payload,
            )
            retry = await self.queue.submit(task)
            res = await retry
            return res if isinstance(res, RolloutResult) else None
        except Exception as e:  # noqa: BLE001 - 重跑异常按失败处理
            logger.warning("%s: retry failed: %s", task_id[0], e)
            return None

    def _load_cached(self, node_key: str) -> Optional[list[Optional[RolloutResult]]]:
        """从 SQLite 加载该节点已有的 rollout 结果（断点续跑复用）。"""
        slots = self.store.load_node_rollouts(self.iid, node_key, self.n_rollouts)
        if slots is None or all(r is None for r in slots):
            return None
        return slots

    # ------------------------------------------------------------------
    # 树搜索标注（ReARTeR process_annotations 移植）
    # ------------------------------------------------------------------

    async def process_annotations(self, root: MCTSNode) -> None:
        nodes = [root]
        iteration = 0
        while True:
            node, idx, qu = select_best_node(nodes, **self.select_kwargs)
            if node is None:
                break
            if node.prefix_steps:
                self.best_entries.append({
                    "instance_id": self.iid,
                    "node_key": node.node_key,
                    "mc_score": node.mc_score,
                    "n_steps": len(node.prefix_steps),
                    "type": "best",
                })
                iteration += 1
                if iteration > self.max_iterations:
                    logger.info("%s: reached max iterations (%d)", self.iid, self.max_iterations)
                    break
            node.increment_visits()
            rollout = node.rollouts[idx]
            expanded, new_leaves = await self.locate_error(node, rollout)
            for n in expanded:
                if n not in nodes:
                    nodes.append(n)
                    self.expanded.append(n)
            self.leaves.extend(new_leaves)
            # 每轮选择-定位后快照一次树结构（内存是权威，DB 是镜像）
            self.store.save_nodes(self.iid, list(self.nodes.values()))
            self._tree_dirty = False

    async def locate_error(
        self, node: MCTSNode, rollout: RolloutResult
    ) -> tuple[list[MCTSNode], list[MCTSNode]]:
        """二分定位首个错误步（委托 :func:`mcts.locate.locate_error`）。"""
        return await locate_error(
            node, rollout,
            get_node=self._get_or_create_node,
            perform_rollouts=lambda n: self._perform_rollouts(n, kind="probe"),
        )

    # ------------------------------------------------------------------
    # 落盘（SQLite）
    # ------------------------------------------------------------------

    def _write_annotations(self) -> None:
        entries = list(self.best_entries)
        entries += [annotation_entry(leaf, "leaf") for leaf in self.leaves]
        # add = 全部扩展节点（对齐 ReARTeR：全对探测节点 n3、leaf 节点不入 add）
        entries += [annotation_entry(n, "add") for n in self.expanded]
        self.store.write_annotations(self.iid, entries)

    def _write_state(self) -> None:
        if self._tree_dirty:
            self.store.save_nodes(self.iid, list(self.nodes.values()))
            self._tree_dirty = False
        self.store.set_instance_status(
            self.iid, self.status,
            root_mc=(self.nodes.get("root").mc_score
                     if self.nodes.get("root") else None),
            messages_head=self._messages_head or None,
        )


# ---------------------------------------------------------------------------
# 管道（编排：worker + 树驱动 + 预算 + 统计）
# ---------------------------------------------------------------------------


class MCTSPipeline:
    """高并发 MCTS 管道：多棵树并发 + 节点内多次 rollout 并发。

    - 树驱动（TreeDriver）只负责**提交任务并等待结果**（解耦）；
    - 任务队列 + Worker + 全局信号量负责并发执行（asyncio + to_thread）；
    - 容器生命周期归 EnvFactory（创建即用、用完即销毁），与树逻辑完全解耦；
    - SQLite（StateStore）作为唯一持久化事实源（rollout 完成即落、树定期快照）。
    """

    def __init__(
        self,
        instances: list[Any],
        *,
        executor_factory: Callable[[Any], RolloutExecutor],
        env_factory: Optional["EnvFactory"] = None,
        store: Optional["StateStore"] = None,
        out_dir: str | Path = "outputs/mcts",
        max_concurrency: int = 8,
        n_rollouts: int = 5,
        temperature_range: tuple[float, float] = (0.7, 1.0),
        select_kwargs: Optional[dict] = None,
        max_iterations: int = 20,
        budget: Optional[Budget] = None,
        resume: bool = True,
        worker_retries: int = 2,
        stats_interval: float = 30.0,
        seed: Optional[int] = None,
        config_file: str = "config/mini_submit.yaml",
    ) -> None:
        from mcts.store import StateStore  # 惰性

        self.instances = list(instances)
        self.executor_factory = executor_factory
        self.env_factory = env_factory or EnvFactory()
        self.out_dir = Path(out_dir)
        self.store = store or StateStore(self.out_dir / "state.db")
        self.max_concurrency = max_concurrency
        self.n_rollouts = n_rollouts
        self.temperature_range = temperature_range
        self.select_kwargs = select_kwargs or {}
        self.max_iterations = max_iterations
        self.budget = budget or Budget()
        self.resume = resume
        self.worker_retries = worker_retries
        self.stats_interval = stats_interval
        self.config_file = config_file
        if seed is not None:
            random.seed(seed)
        self.stats = Stats()

    async def run(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        queue = TaskQueue()
        sem = asyncio.Semaphore(self.max_concurrency)
        executor = self.executor_factory(self.env_factory)
        workers = [
            Worker(queue, executor, sem, self.stats, retries=self.worker_retries)
            for _ in range(max(1, self.max_concurrency))
        ]
        worker_tasks = [asyncio.create_task(w.run()) for w in workers]
        logger_task = asyncio.create_task(self._periodic_stats())

        results: list[dict] = []
        db_counts: dict = {}
        try:
            drivers = [
                asyncio.create_task(TreeDriver(
                    inst, queue=queue, stats=self.stats, budget=self.budget,
                    out_dir=self.out_dir, store=self.store,
                    n_rollouts=self.n_rollouts,
                    temperature_range=self.temperature_range,
                    select_kwargs=self.select_kwargs,
                    max_iterations=self.max_iterations, resume=self.resume,
                    config_file=self.config_file,
                ).run())
                for inst in self.instances
            ]
            results = await asyncio.gather(*drivers)
        finally:
            logger_task.cancel()
            for w in workers:
                w.shutdown.set()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            db_counts = self.store.counts()   # 关闭前统计
            self.store.close()

        snapshot = self.stats.snapshot()
        snapshot["env"] = self.env_factory.stats()
        snapshot["db"] = db_counts
        statuses: dict[str, int] = {}
        for r in results:
            statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        logger.info("pipeline done: %s; statuses=%s", snapshot, statuses)
        return {"stats": snapshot, "statuses": statuses, "instances": results}

    async def _periodic_stats(self) -> None:
        while True:
            await asyncio.sleep(self.stats_interval)
            logger.info("progress: %s; env=%s; db=%s",
                        self.stats.snapshot(), self.env_factory.stats(),
                        self.store.counts())


__all__ = [
    "RolloutTask",
    "RolloutResult",
    "RolloutExecutor",
    "TaskQueue",
    "Stats",
    "Budget",
    "BudgetExhausted",
    "EnvFactory",
    "Worker",
    "TreeDriver",
    "MCTSPipeline",
]
