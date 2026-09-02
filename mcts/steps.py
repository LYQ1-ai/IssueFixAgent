# SPDX-License-Identifier: BSD-3-Clause

"""轨迹 → 步（step）序列解析（PLAN §2.2 / D1）。

**步（step）定义**：一步 = agent 一次 ``step()``（一次 model query + 随后的工具执行），
对应轨迹 ``trajectory["messages"]`` 中一条 **assistant** 消息及其后续 **tool** /
FormatError **user** 反馈消息（直到下一条 assistant / exit 为止）。

本模块是纯解析逻辑（不 import ``agent`` / ``minisweagent``）：

- :func:`split_steps`：``messages`` → ``list[Step]``（assistant + tail）；
- :func:`extract_exit`：取出 ``exit_status`` / ``submission``（提交协议：
  ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` 后的内容即 submission）；
- :func:`messages_for_prefix`：取"前 k 步"对应的**原始消息序列**（probe 续跑时
  预填进 DefaultAgent 的完整对话，D3 前缀回放的关键）；
- :func:`prefix_node_key`：按前缀**内容**计算节点 key（``"root"`` 为空前缀；
  否则 sha1(各步内容序列化)）—— 同一前缀内容共享同一节点与 rollout（去重）。

轨迹格式为 mini-swe-agent ``DefaultAgent.serialize()`` 输出（``trajectory_format:
mini-swe-agent-1.1``），消息角色：``system`` / ``user`` / ``assistant``（``extra.actions``
为 bash 工具调用） / ``tool``（``extra.raw_output`` / ``extra.returncode``） / ``exit``
（``extra.exit_status`` / ``extra.submission``）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_EXIT = "exit"

MAGIC_SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

TERMINAL_ROLES = (ROLE_EXIT,)


@dataclass
class Step:
    """一步决策：assistant 消息 + 其后续 tool / FormatError user 消息。

    ``tail`` 保持消息顺序，重放时原样重放 assistant 的 actions、用 tool 消息里的
    ``raw_output`` 做 observation 一致性校验。
    """

    assistant: dict
    tail: list[dict] = field(default_factory=list)

    @property
    def content(self) -> str:
        return str(self.assistant.get("content", ""))

    @property
    def actions(self) -> list[dict]:
        """该步的 bash 工具调用（``[{"command": ..., "tool_call_id": ...}]``）。"""
        return list(self.assistant.get("extra", {}).get("actions", []))

    @property
    def commands(self) -> list[str]:
        return [a.get("command", "") for a in self.actions]

    def to_json(self) -> dict:
        """完整往返（断点续跑 / node_key 稳定）：assistant 消息与 tail 原样保留。"""
        return {"assistant": self.assistant, "tail": list(self.tail)}

    @classmethod
    def from_json(cls, d: dict) -> "Step":
        return cls(assistant=dict(d.get("assistant", {})),
                   tail=list(d.get("tail", [])))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Step({len(self.commands)} actions: {self.commands[:2]}...)"


def split_steps(messages: list[dict]) -> list[Step]:
    """把轨迹消息序列切成步序列（assistant 起始，到下一 assistant / exit 为止）。

    - ``system`` / 开头的 ``user``（任务描述）不属于任何步，忽略；
    - ``exit`` 是终态标记，不进入步序列（见 :func:`extract_exit`）；
    - FormatError 反馈（``role=user`` 且带 ``extra.interrupt_type``）归入前一步的 tail
      —— 它是对话的一部分，但不算一次决策步。
    """
    steps: list[Step] = []
    current: Optional[Step] = None
    for msg in messages:
        role = msg.get("role")
        if role == ROLE_ASSISTANT:
            current = Step(assistant=msg)
            steps.append(current)
        elif role in (ROLE_TOOL, ROLE_USER):
            if current is not None:
                current.tail.append(msg)
            # 无当前步的 user 消息（任务描述前缀）忽略
        elif role == ROLE_EXIT:
            break
        # system 忽略
    return steps


def extract_exit(messages: list[dict]) -> tuple[str, str]:
    """从消息序列末尾的 exit 消息取 ``(exit_status, submission)``。

    找不到 exit 消息返回 ``("", "")``（异常中断的轨迹）。submission = 提交魔法串
    之后的内容（mini-swe-agent 的提交协议，见 ``DockerEnvironment._check_finished``）。
    """
    for msg in reversed(messages):
        if msg.get("role") == ROLE_EXIT:
            extra = msg.get("extra", {}) or {}
            return str(extra.get("exit_status", "")), str(extra.get("submission", ""))
    return "", ""


def messages_for_prefix(messages: list[dict], k: int) -> list[dict]:
    """返回前 ``k`` 步对应的**原始消息序列**（含第 k 步的 observation）。

    ``k=0``：只保留开头的 system + user（任务描述）消息（无任何决策步）；
    ``k>=len(steps)``：返回全部消息（去掉尾部 exit）。返回值直接用于 probe 续跑时
    预填 ``DefaultAgent.messages``（D3）。
    """
    if k <= 0:
        out: list[dict] = []
        for msg in messages:
            if msg.get("role") == ROLE_ASSISTANT:
                break
            out.append(msg)
        return out
    assistant_idx = [i for i, m in enumerate(messages) if m.get("role") == ROLE_ASSISTANT]
    if not assistant_idx:
        return []
    end = assistant_idx[k] if k < len(assistant_idx) else len(messages)
    while end > 0 and messages[end - 1].get("role") == ROLE_EXIT:
        end -= 1
    return messages[:end]


def prefix_node_key(prefix_steps: list[Step] | list[dict]) -> str:
    """前缀节点的稳定 key：空前缀 = ``"root"``；否则按内容哈希。

    内容寻址 ⇒ 同一棵树里不同 rollout 产生的相同前缀 probe 共享同一节点与
    rollout（不重复计算 MC），也是断点续跑时在磁盘上定位缓存文件的依据。
    """
    if not prefix_steps:
        return "root"
    h = hashlib.sha1()
    for s in prefix_steps:
        d = s.to_json() if isinstance(s, Step) else s
        h.update(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return "n_" + h.hexdigest()[:16]


def steps_to_messages(steps: list[Step]) -> list[dict]:
    """把步序列还原成消息序列（probe 前缀消息由 :func:`messages_for_prefix` 提供，
    本函数用于持久化 / 测试。"""
    out: list[dict] = []
    for s in steps:
        out.append(s.assistant)
        out.extend(s.tail)
    return out


__all__ = [
    "Step",
    "split_steps",
    "extract_exit",
    "messages_for_prefix",
    "prefix_node_key",
    "steps_to_messages",
    "MAGIC_SUBMIT",
]
