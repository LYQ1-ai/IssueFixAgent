# SPDX-License-Identifier: BSD-3-Clause

"""D3 前缀续跑（probe 机制）：回放前缀 + 自由续跑（PLAN §2.3 D3 / §2.4）。

实现思路：**不修改 mini-swe-agent 源码**，利用 DefaultAgent 的可注入消息：

- **回放（确定性还原容器状态）**：把前缀各步的 bash action 按序在容器里重放
  （``AttachContainerEnvironment.execute``），与记录的 observation
  （``tool`` 消息的 ``extra.raw_output`` / ``returncode``）比对 —— 一致 ⇒ 容器状态
  还原到"已走完前缀"；不一致 ⇒ 标记 ``replay_drift``（可配置
  ``replay_require_match=True`` 硬失败，PLAN §6 风险对策）；
- **自由续跑**：``DefaultAgent.messages`` 预填"system + user + 前缀消息"完整对话，
  然后复刻 ``DefaultAgent.run()`` 的循环逐 ``step()`` 直到 exit —— 模型从前缀
  之后继续自由生成，无需重新生成前缀（对齐 ReARTeR 把 ``partial_answer`` 拼进
  prompt 的语义；此处前缀包含文件系统状态，故必须真重放动作）。

``run_free``（根节点）直接复用 :class:`agent.base_agent.RepoAgent`（含 Phoenix 追踪
支持），与既有 demo 路径一致。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("mcts.replay")


class ReplayRunner:
    """在指定容器内执行一次 rollout（自由跑 / 前缀回放 + 自由续跑）。

    提交协议（PLAN §2.2 改造）：``agent_class`` 默认
    ``agent.submit_agent.SubmitAgent``（submit_locations 工具提交 + 轮次耗尽
    催收）；``magic_submit=False`` 时环境层禁用 bash 魔法串提交。
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_config: Optional[dict] = None,
        config_file: str = "config/mini_submit.yaml",
        step_limit: int = 20,
        cost_limit: float = 3.0,
        max_consecutive_format_errors: int = 3,
        wall_time_limit_seconds: int = 0,
        env_cfg: Optional[dict] = None,
        agent_class: Any = "agent.submit_agent.SubmitAgent",
        replay_require_match: bool = False,
    ):
        self.model_name = model_name
        self.model_config = dict(model_config or {})
        self.config_file = config_file
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.max_consecutive_format_errors = max_consecutive_format_errors
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.env_cfg = dict(env_cfg or {})
        self.agent_class = agent_class
        self.replay_require_match = replay_require_match

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def run_free(self, instance: Any, container: str) -> dict:
        """根节点 rollout：从零自由跑到终态，返回完整 serialize 轨迹。

        复用 ``RepoAgent``（与 demo.py 完全相同的执行路径），容器由调用方提供
        （``container=`` 传入 ⇒ run() 内部不创建不释放）。
        """
        from agent.base_agent import RepoAgent  # 惰性：纯逻辑测试不拉起 agent
        from agent.submit_agent import resolve_agent_class  # noqa: F401

        agent = RepoAgent(
            instance.repo,
            instance.base_commit,
            task=instance.problem_statement,
            patch=instance.patch if instance.use_patch else None,
            container=container,
            model_name=self.model_name,
            model_config=self.model_config,
            config_file=self.config_file,
            agent_config={"step_limit": self.step_limit},
            agent_class=resolve_agent_class(self.agent_class),
            env_config=dict(self.env_cfg),
            keep_container=True,  # 生命周期归调用方（EnvFactory：用完即销毁）
        )
        result = agent.run()
        trajectory = result.get("trajectory") or {}
        trajectory.setdefault("info", {})["trace_id"] = result.get("trace_id")
        return trajectory

    def run_probe(
        self,
        instance: Any,
        container: str,
        *,
        prefix_messages: list[dict],
        prefix_steps: list[Any],
    ) -> tuple[dict, bool]:
        """probe rollout：回放前缀动作还原容器状态 → 预填对话 → 自由续跑。

        Args:
            prefix_messages: ``system + user + 前缀步消息`` 的完整对话（probe 预填）。
            prefix_steps: ``list[Step]`` —— 需要重放的前缀步（含 actions 与期望输出）。

        Returns:
            ``(trajectory, drift)``；``drift`` 为 True 表示至少一个前缀 action 的
            重放输出与记录不一致（容器状态可能漂移，MC 估计噪声，仍可用）。
        """
        # 兜底：prefix 必须含 user 消息（system+user 头部 + 步消息）。
        # 修复 2026-08-31：root 头部缺失时（如 root rollout 失败）构造出的
        # prefix_messages 无 user → sglang 400 "No user query found in messages"
        # 且 litellm 16s/60s 退避重试刷屏——这里直接拒绝，避免发坏请求。
        if not any(m.get("role") == "user" for m in (prefix_messages or [])):
            raise ValueError(
                "prefix_messages missing a user message (root head unavailable); "
                "refusing to send malformed request to the LLM"
            )
        from minisweagent.models import get_model  # noqa: E402

        from agent.base_agent import AttachContainerEnvironment  # noqa: E402
        from agent.submit_agent import resolve_agent_class  # noqa: E402

        env = AttachContainerEnvironment(container_name=container, **self.env_cfg)
        drift = self._replay_prefix(env, prefix_steps)
        remaining_steps = max(0, self.step_limit - len(prefix_steps))
        agent_cfg: dict[str, Any] = {
            "system_template": "",
            "instance_template": "",
            "step_limit": remaining_steps,
            "cost_limit": self.cost_limit,
            "max_consecutive_format_errors": self.max_consecutive_format_errors,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
        }
        model = get_model(self.model_name, config=dict(self.model_config))
        agent = resolve_agent_class(self.agent_class)(model, env, **agent_cfg)
        agent.messages = [dict(m) for m in prefix_messages]  # 预填完整对话（D3）
        trajectory = self._run_agent_steps(agent)
        return trajectory, drift

    # ------------------------------------------------------------------
    # 内部：回放 + 自由续跑循环
    # ------------------------------------------------------------------

    def _replay_prefix(self, env: Any, prefix_steps: list[Any]) -> bool:
        """按序重放前缀动作并校验 observation；返回是否发生漂移。"""
        from minisweagent.exceptions import InterruptAgentFlow  # noqa: E402

        drift = False
        for step in prefix_steps:
            for action in step.actions:
                expected = self._expected_observation(step, action)
                try:
                    out = env.execute(action)
                except InterruptAgentFlow:
                    logger.warning("replay interrupted by terminal command: %s",
                                   action.get("command", "")[:80])
                    drift = True
                    continue
                if expected is not None and (
                    out.get("output", "") != expected.get("raw_output", "")
                    or out.get("returncode") != expected.get("returncode")
                ):
                    drift = True
                    logger.warning(
                        "replay drift: cmd=%r rc=%s vs recorded rc=%s (len %d vs %d)",
                        action.get("command", "")[:80], out.get("returncode"),
                        expected.get("returncode"),
                        len(out.get("output", "")), len(expected.get("raw_output", "")),
                    )
                    if self.replay_require_match:
                        raise RuntimeError("replay observation mismatch (strict mode)")
        return drift

    @staticmethod
    def _expected_observation(step: Any, action: dict) -> Optional[dict]:
        """从步的 tail 里按 tool_call_id 找该 action 的记录 observation。"""
        tid = action.get("tool_call_id")
        for msg in step.tail:
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tid:
                return msg.get("extra") or {}
        return None

    def _run_agent_steps(self, agent: Any) -> dict:
        """跑完剩余步（消息已预填、不渲染初始模板），返回 serialize 轨迹。

        优先用 agent 的 ``_agent_loop()``（``SubmitAgent``：含提交催收逻辑 ——
        轮次耗尽 / 提前结束但未提交时注入 user 催收消息给最后一次机会）；
        ``DefaultAgent`` 无 ``_agent_loop`` 时回退到官方循环复刻（FormatError
        恢复 / RepeatedFormatError 出口 / InterruptAgentFlow 出口）。
        """
        loop = getattr(agent, "_agent_loop", None)
        if loop is not None:
            loop()
            return agent.serialize()

        from minisweagent.exceptions import FormatError, InterruptAgentFlow  # noqa: E402

        while True:
            try:
                agent.step()
                agent.n_consecutive_format_errors = 0
            except FormatError as e:
                agent.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                agent.n_consecutive_format_errors += 1
                if 0 < agent.config.max_consecutive_format_errors <= agent.n_consecutive_format_errors:
                    agent.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    agent.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                agent.add_messages(*e.messages)
            except Exception as e:  # noqa: BLE001 - 与官方 run() 一致：记录后抛出
                agent.handle_uncaught_exception(e)
                raise
            if agent.messages[-1].get("role") == "exit":
                break
        return agent.serialize()


__all__ = ["ReplayRunner"]
