# SPDX-License-Identifier: BSD-3-Clause

"""结果提交 Agent（继承 ``DefaultAgent``，不修改 mini-swe-agent 源码）。

两处扩展：

1. **submit 动作拦截（任务结束标志）**：``execute_actions`` 检测到
   ``submit_locations`` 动作时不再执行任何命令，直接构造 submission
   （locations 的 JSON）并抛 ``Submitted``（``InterruptAgentFlow``）—— 轨迹以
   ``exit`` 消息收尾，``exit_status="Submitted"``、``submission=JSON``，与官方
   ``run()`` 的捕获路径完全一致（``mcts/steps.py::extract_exit`` 无需改动）。
   ``submit_locations`` 与 bash 同批调用 → ``FormatError``（模型可纠正重试）。

2. **提交催收**：任何"未提交就结束"的路径（轮次上限 / 时间上限 / 连续格式错误
   RepeatedFormatError）都会先被**催收一次**：移除 exit 消息、注入一条 user
   消息要求立即调用 ``submit_locations``、放行下一轮（grace）；grace 轮之后
   仍未提交才真正结束。保证"到达轮次上限或提前结束但未输出最终结果时，发送
   user message 要求通过结果提交工具输出结果"（PLAN §2.2 提交协议改造）。

循环逻辑抽为 :meth:`_agent_loop`：``run()`` 在渲染初始 system/user 消息后调用；
``mcts.replay.ReplayRunner`` 的 probe 续跑复用同一循环（消息已预填、不渲染模板）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    TimeExceeded,
)

from agent.submit_tool import (
    SUBMIT_MUST_BE_ALONE_MSG,
    SUBMIT_REMINDER,
    SUBMIT_TOOL_NAME,
)

logger = logging.getLogger("agent.submit_agent")


class SubmitAgent(DefaultAgent):
    """默认提交方式为 ``submit_locations`` 工具的 agent（带轮次耗尽催收）。"""

    def __init__(self, model: Any, env: Any, *, config_class: type = AgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.submitted = False            # 是否已通过 submit_locations 提交
        self._reminder_sent = False       # 是否已催收过一次
        self._grace_after_reminder = 0    # 催收后放行的轮次配额

    # ------------------------------------------------------------------
    # 提交：submit_locations 动作 → 任务结束（新的任务结束标志）
    # ------------------------------------------------------------------

    def execute_actions(self, message: dict) -> list[dict]:
        """执行动作：bash 照旧；submit_locations 触发提交并结束任务。"""
        actions = message.get("extra", {}).get("actions", [])
        submit_actions = [a for a in actions if a.get("tool") == SUBMIT_TOOL_NAME]
        bash_actions = [a for a in actions if a.get("tool") != SUBMIT_TOOL_NAME]
        if submit_actions and bash_actions:
            raise FormatError(
                {
                    "role": "user",
                    "content": SUBMIT_MUST_BE_ALONE_MSG,
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        if submit_actions:
            self.submitted = True
            # 规范化键（模型层 parse_submit_locations 已保证 file/class_name/
            # function_name 齐全；此处兜底，防解析层被绕过时输出不一致）
            submission = json.dumps(
                [
                    {
                        "file": loc.get("file", ""),
                        "class_name": loc.get("class_name"),
                        "function_name": loc.get("function_name"),
                    }
                    for loc in submit_actions[0].get("locations", [])
                ],
                ensure_ascii=False,
            )
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return super().execute_actions(message)

    # ------------------------------------------------------------------
    # 轮次上限 / 时间上限：催收后放行一轮（grace），否则与官方一致
    # ------------------------------------------------------------------

    def query(self) -> dict:
        """复刻官方 query()，但 limit 命中且处于催收 grace 时放行本轮。"""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            if self._grace_after_reminder > 0:
                self._grace_after_reminder -= 1
            else:
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": "LimitsExceeded",
                        "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                    }
                )
        if 0 < self.config.wall_time_limit_seconds <= self._age_seconds():
            if self._grace_after_reminder > 0:
                self._grace_after_reminder -= 1
            else:
                raise TimeExceeded(
                    {
                        "role": "exit",
                        "content": "TimeExceeded",
                        "extra": {"exit_status": "TimeExceeded", "submission": ""},
                    }
                )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def _age_seconds(self) -> int:
        """距 agent 启动的秒数（wall_time 检查；保持与官方同一基准）。"""
        import time

        return int(time.time() - self._start_time)

    # ------------------------------------------------------------------
    # 主循环：复刻官方 run() + 未提交即结束时的催收
    # ------------------------------------------------------------------

    def run(self, task: str = "", **kwargs) -> dict:
        """与官方 ``DefaultAgent.run()`` 语义一致，仅提交方式与催收不同。"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render_template(self.config.system_template),
            ),
            self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            ),
        )
        self._agent_loop()
        return self.messages[-1].get("extra", {})

    def _agent_loop(self) -> None:
        """step 循环（probe 续跑复用：消息已预填时不渲染初始模板）。

        与官方 ``run()`` 循环一致（FormatError 恢复 / RepeatedFormatError 出口 /
        InterruptAgentFlow 出口 / 异常出口），额外在"exit 且未提交"时催收一次。
        """
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                # 该调用已计费，query() 未记账
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:  # noqa: BLE001 - 与官方 run() 一致：记录后抛出
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                if self._maybe_remind():
                    continue
                break

    def _maybe_remind(self) -> bool:
        """exit 且未提交 → 移除 exit、注入催收 user 消息、放行一轮；返回是否继续。"""
        last = self.messages[-1]
        extra = last.get("extra", {}) or {}
        if self.submitted or extra.get("exit_status") == "Submitted":
            return False  # 已提交：正常结束
        if self._reminder_sent:
            return False  # 已催收过一次：真正结束
        self._reminder_sent = True
        self.messages.pop()  # 移除 exit 消息
        self.add_messages(
            self.model.format_message(role="user", content=SUBMIT_REMINDER)
        )
        self._grace_after_reminder += 1  # 放行下一轮（绕过 limit 检查）
        logger.info("submit reminder injected (n_calls=%d)", self.n_calls)
        return True


def resolve_agent_class(spec: Any) -> type:
    """把 ``"module.path.ClassName"`` 或类对象解析为 agent 类。"""
    if isinstance(spec, type):
        return spec
    if not isinstance(spec, str) or "." not in spec:
        raise ValueError(f"invalid agent_class spec: {spec!r}")
    module_name, class_name = spec.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


__all__ = ["SubmitAgent", "resolve_agent_class"]
