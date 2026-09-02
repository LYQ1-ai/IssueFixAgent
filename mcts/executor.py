# SPDX-License-Identifier: BSD-3-Clause

"""生产 rollout 执行器：容器创建 → 回放/自由跑 → 终态回报 → 容器销毁。

组合 :mod:`mcts.replay`（D3 前缀续跑）、:mod:`mcts.reward`（**结构化判定**：
submit_locations 提交的 locations vs gold 定位 F1）、``mcts.tasks.EnvFactory``
（创建即用、用完即销毁），实现 ``mcts.tasks.RolloutExecutor`` 协议 —— 阻塞、
在线程池内调用；结果持久化由 ``mcts.tasks.TreeDriver`` 统一写入 SQLite。

提交协议（PLAN §2.2 改造）：模型走 ``agent.submit_model.SubmitLocationsModel``
（``tools=[bash, submit_locations]``），agent 走 ``agent.submit_agent.SubmitAgent``
（submit 工具触发 ``Submitted`` + 轮次耗尽催收）；bash 魔法串提交默认禁用
（``magic_submit=False``）。判定为完全结构化：无有效提交 → reward 0。

可选 Phoenix 追踪：开启时为每次 rollout 注入独立 trace 上下文（线程池共享进程，
``setup_phoenix_tracing`` 仅需执行一次），trace_id 记入结果。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

from mcts.llm import build_model_config
from mcts.replay import ReplayRunner
from mcts.reward import reward_from_trajectory_exit
from mcts.steps import extract_exit, split_steps
from mcts.tasks import EnvFactory, RolloutResult, RolloutTask

logger = logging.getLogger("mcts.executor")

# 每次 rollout 的采样温度范围（ReARTeR：rollout 多样性来源）
DEFAULT_TEMPERATURE_RANGE = (0.7, 1.0)


class AgentRolloutExecutor:
    """真实环境 rollout 执行器（docker 容器 + mini-swe-agent + vLLM）。

    **提交协议（2026-08-27 起）**：Agent 通过 ``submit_locations`` 工具提交
    结构化定位结果（PLAN §2.2 提交协议改造）；判定为**完全结构化** —— 仅
    ``exit_status == "Submitted"`` 且 submission 为合法 locations JSON 才计
    reward（``mcts.reward.reward_from_trajectory_exit``），无有效提交 → 0。

    **奖励（2026-08-28 起，docs/reward_design.md 综合方案）**：默认
    ``reward_mode="layered"`` —— 层级路径一对一匹配 + Soft-F1
    （``Reward = 2C/(M+N)``）+ Lᵢ 缺失折算 + τ=0.6 二值化（+ strict_multi_gate）。
    连续 reward 与分层细节（``reward_details``）一并写入 rollout 结果落库。
    """

    def __init__(
        self,
        env_factory: EnvFactory,
        *,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config_file: str = "config/mini_submit.yaml",
        step_limit: int = 20,
        cost_limit: float = 3.0,
        reward_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        reward_threshold: float = 0.6,
        reward_mode: str = "layered",
        depth_weights: Optional[dict] = None,
        strict_multi_gate: bool = True,
        missing_level_norm: bool = True,
        overpromise_penalty: float = 0.0,
        duplicate_penalty: float = 0.0,
        temperature_range: tuple[float, float] = DEFAULT_TEMPERATURE_RANGE,
        workdir: str = "/repo",
        phoenix_tracing: Optional[dict] = None,
        replay_require_match: bool = False,
        model_class: str = "agent.submit_model.SubmitLocationsModel",
        agent_class: Any = "agent.submit_agent.SubmitAgent",
        magic_submit: bool = False,
    ):
        self.env_factory = env_factory
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.config_file = config_file
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.reward_weights = reward_weights
        self.reward_threshold = reward_threshold
        self.reward_mode = reward_mode
        self.depth_weights = depth_weights
        self.strict_multi_gate = strict_multi_gate
        self.missing_level_norm = missing_level_norm
        self.overpromise_penalty = overpromise_penalty
        self.duplicate_penalty = duplicate_penalty
        self.temperature_range = temperature_range
        self.workdir = workdir
        self.phoenix_tracing = phoenix_tracing
        self.replay_require_match = replay_require_match
        self.model_class = model_class
        self.agent_class = agent_class
        self.magic_submit = magic_submit
        self._trace_ready = False

    # ------------------------------------------------------------------
    # RolloutExecutor 协议
    # ------------------------------------------------------------------

    def run(self, task: RolloutTask) -> RolloutResult:
        payload = task.payload
        instance = payload["instance"]
        container: Optional[str] = None
        t0 = time.monotonic()
        try:
            # 创建即用（每次 rollout 一个全新容器，bug 状态）
            container = self.env_factory.create_env(instance)
            temperature = payload.get("temperature") or random.uniform(*self.temperature_range)
            model_cfg = build_model_config(
                self.model_name, self.base_url, self.api_key, temperature,
                model_class=self.model_class,
            )
            runner = ReplayRunner(
                model_name=self.model_name,
                model_config=model_cfg,
                config_file=self.config_file,
                step_limit=self.step_limit,
                cost_limit=self.cost_limit,
                env_cfg={"cwd": self.workdir, "magic_submit": self.magic_submit},
                agent_class=self.agent_class,
                replay_require_match=self.replay_require_match,
            )
            trace_id = self._begin_trace()
            try:
                if task.kind == "root":
                    trajectory = runner.run_free(instance, container)
                    drift = False
                else:
                    trajectory, drift = runner.run_probe(
                        instance, container,
                        prefix_messages=payload["prefix_messages"],
                        prefix_steps=payload["prefix_steps"],
                    )
            finally:
                self._end_trace()

            full_steps = split_steps(trajectory.get("messages", []))
            prefix_len = len(payload.get("prefix_steps", []))
            steps = full_steps[prefix_len:]  # 续跑步（相对节点前缀）

            exit_status, submission = extract_exit(trajectory.get("messages", []))
            # 完全结构化判定：无有效提交 → reward 0（对齐 CodeScout）；
            # layered 模式（默认）：Soft-F1 + 深度权重 + Lᵢ 折算 + τ 二值化
            reward, correct, details = reward_from_trajectory_exit(
                exit_status, submission, instance.gold,
                threshold=self.reward_threshold,
                mode=self.reward_mode,
                weights=self.reward_weights,
                depth_weights=self.depth_weights,
                strict_multi_gate=self.strict_multi_gate,
                missing_level_norm=self.missing_level_norm,
                overpromise_penalty=self.overpromise_penalty,
                duplicate_penalty=self.duplicate_penalty,
            )
            info = trajectory.get("info") or {}
            model_stats = info.get("model_stats") or {}
            result = RolloutResult(
                instance_id=task.instance_id,
                node_key=task.node_key,
                rollout_idx=task.rollout_idx,
                reward=reward,
                correct=correct,
                steps=steps,
                trajectory=self._light_trajectory(trajectory),
                exit_status=exit_status,
                submission=submission,
                n_calls=int(model_stats.get("api_calls", 0)),
                cost=float(model_stats.get("instance_cost", 0.0)),
                duration=time.monotonic() - t0,
                replay_drift=drift,
                trace_id=trace_id,
                reward_details=details,
            )
            logger.debug("rollout done %s reward=%.3f correct=%s steps=%d drift=%s",
                         task.task_id, reward, correct, len(steps), drift)
            return result
        except Exception as e:  # noqa: BLE001 - 失败结果不中断管道（落库由 TreeDriver 负责）
            logger.exception("rollout %s failed: %s", task.task_id, e)
            return RolloutResult(
                instance_id=task.instance_id, node_key=task.node_key,
                rollout_idx=task.rollout_idx,
                error=f"{type(e).__name__}: {e}",
                duration=time.monotonic() - t0,
            )
        finally:
            if container is not None:
                try:
                    self.env_factory.destroy_env(container)  # 用完即销毁（成功/失败都删）
                except Exception as e:  # pragma: no cover - 归还失败不影响结果
                    logger.warning("container release failed %s: %s", container, e)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _light_trajectory(self, trajectory: dict) -> dict:
        """精简轨迹：丢弃 assistant 消息里的大块 ``extra.response``，保留
        actions / raw_output / returncode / exit 信息（PRM 数据与调试够用）。"""
        info = trajectory.get("info") or {}
        msgs: list[dict] = []
        for m in trajectory.get("messages", []):
            extra = m.get("extra") or {}
            keep = {}
            for key in ("actions", "raw_output", "returncode", "exit_status",
                        "submission", "exception_info"):
                if key in extra:
                    keep[key] = extra[key]
            msg = {k: m.get(k) for k in ("role", "content", "tool_call_id") if k in m}
            if keep:
                msg["extra"] = keep
            msgs.append(msg)
        return {"info": info, "messages": msgs,
                "trajectory_format": trajectory.get("trajectory_format")}

    # ------------------------------------------------------------------
    # Phoenix 追踪（可选）
    # ------------------------------------------------------------------

    def _begin_trace(self) -> Optional[str]:
        if not self.phoenix_tracing:
            return None
        try:
            from agent.tracing import (  # 惰性导入
                end_trace_context, setup_phoenix_tracing, start_trace_context,
            )

            if not self._trace_ready:
                setup_phoenix_tracing(
                    endpoint=self.phoenix_tracing.get("endpoint"),
                    project_name=self.phoenix_tracing.get("project_name"),
                )
                self._trace_ready = True
            self._trace_token, trace_id = start_trace_context(
                trace_id=self.phoenix_tracing.get("trace_id"))
            return trace_id
        except Exception as e:  # 追踪失败不阻断 rollout
            logger.warning("phoenix tracing disabled: %s", e)
            return None

    def _end_trace(self) -> None:
        token = getattr(self, "_trace_token", None)
        if token:
            try:
                from agent.tracing import end_trace_context

                end_trace_context(token)
            except Exception:  # pragma: no cover - 撤销失败不影响主流程
                pass
            self._trace_token = None


__all__ = ["AgentRolloutExecutor", "DEFAULT_TEMPERATURE_RANGE"]
