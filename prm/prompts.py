# SPDX-License-Identifier: BSD-3-Clause

"""PRM prompt 模板常量（docs/prm_training_plan.md §4，版本化）。

**改动规则**：任何常量的文本改动都必须递增 :data:`TEMPLATE_VERSION` 并重建
parquet 数据集（build_dataset 的 manifest 记录 ``template_hash``，训练/评估
加载时校验一致，防止 prompt 与模型错配）。

渲染结构（§4.1）::

    [system]              PRM 角色定义（SYSTEM_PRM_V1）
    [user]                任务上下文 + 轨迹说明（USER_CONTEXT_V1，引用 head user 原文）
    [assistant/tool ...]  规范化后的轨迹前缀（含被判定步）
    [user]                末轮判定指令（USER_INSTRUCTION_V1）
    ── apply_chat_template(add_generation_prompt=True, enable_thinking=False) ──
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n   ← 打分位置 = 这之后的最后 token（F3）

**模板行为备注（Qwen3.5 chat_template 实测）**：assistant 消息的
``reasoning_content`` 只有在其位于"最后一条真实 user 查询"**之后**时才渲染为
``<think>...</think>``（模板以反向扫描定位 last_query_index）。v1 布局的末轮
判定指令是最后一条 user 查询，因此轨迹中的 assistant 思维链不进入渲染文本——
但 ``reasoning_content`` 仍完整保留在 canonical messages / parquet 中（§3.1，
作为后续 rationale 升级方案的原料）。M4.0 probe 若 AUC≈0.5 排查时把这一行为
列为候选因素。

规则（§4.2）：

1. ``{issue_and_conventions}`` = head user 内容**原文**（含 agent 的工具约定，无损）；
2. **严禁 gold 信息**（patch / gold locations）进入任何 prompt；
3. ``TRUNCATION_MARKER`` 仅在训练期截断（prm/data.py）插入，不属于构建期模板。
"""

from __future__ import annotations

import hashlib

TEMPLATE_VERSION = "v1"

SYSTEM_PRM_V1 = (
    "You are a process reward model (PRM) for a software engineering agent. "
    "You will read a GitHub issue and the agent's execution trajectory "
    "(assistant messages contain the agent's reasoning and a bash tool call; "
    "tool messages contain the command output). "
    "Judge whether the agent is on track to fix the issue from its current state. "
    "Answer with exactly one word: Correct or Incorrect."
)

USER_CONTEXT_V1 = (
    "You are reviewing a coding agent's attempt to fix the GitHub issue below.\n\n"
    "<task_context>\n{issue_and_conventions}\n</task_context>\n\n"
    "The following conversation is the agent's trajectory so far."
)

USER_INSTRUCTION_V1 = (
    "The trajectory above ends with the agent's latest action and its observation. "
    "Based on the current repository state, judge whether continuing from here is "
    "likely to fix the issue. Respond with exactly one word: Correct or Incorrect."
)

# 训练期整步删除后插入的占位标记（§7.2；不属于构建期模板，但随 template_hash 一起
# 版本化，保证 manifest 可复现）。
TRUNCATION_MARKER = "[earlier steps omitted to fit the PRM context window]"

# 纳入 template_hash 的全部常量（顺序固定；新增常量必须登记在此处）
_HASHED_CONSTANTS = (
    TEMPLATE_VERSION,
    SYSTEM_PRM_V1,
    USER_CONTEXT_V1,
    USER_INSTRUCTION_V1,
    TRUNCATION_MARKER,
)


def template_hash() -> str:
    """sha1(全部 prompt 常量拼接)——manifest / run config 记录用（§4.2 规则 3）。"""
    h = hashlib.sha1()
    for c in _HASHED_CONSTANTS:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")  # 常量边界分隔，防拼接歧义
    return h.hexdigest()


__all__ = [
    "TEMPLATE_VERSION",
    "SYSTEM_PRM_V1",
    "USER_CONTEXT_V1",
    "USER_INSTRUCTION_V1",
    "TRUNCATION_MARKER",
    "template_hash",
]
