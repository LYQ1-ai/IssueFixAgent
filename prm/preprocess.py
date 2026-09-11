# SPDX-License-Identifier: BSD-3-Clause

"""轨迹预处理：步解析 → 消息规范化 → PRM prompt 组装 → 渲染计长
（docs/prm_training_plan.md §3 / §4）。

所有"原始数据 → 可渲染 messages"的逻辑集中在
:class:`TrajectoryPreprocessor` 一个类（可多进程实例化：tokenizer 惰性加载，
每个 worker 进程各自加载一次）。

规范化依据 F5 实测：assistant 100% 带 ``tool_calls``（命令在
``function.arguments``，JSON **字符串**）、100% 带 ``reasoning_content``、99%
content 近空；tool 消息 content 已含 ``{"returncode":..,"output":..}``；另有
``function_call`` / ``provider_specific_fields`` / ``extra`` 冗余字段。

**arguments 必须解析为 dict**：Qwen3.5 chat template 以
``tool_call.arguments|items`` 渲染参数（jinja2 ``items`` 过滤器对字符串抛
TypeError，实测），而 DB 里的 ``arguments`` 是 JSON 字符串——规范化时
``json.loads`` 为 mapping，解析失败才保留原样（防御性）。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from mcts.steps import Step

from prm.prompts import SYSTEM_PRM_V1, USER_CONTEXT_V1, USER_INSTRUCTION_V1

logger = logging.getLogger("prm.preprocess")

# 无 tokenizer 时的 chars/token 近似（§3 rendered_tokens；报告需标注 approx）
APPROX_CHARS_PER_TOKEN = 3.5

# 渲染计长走 chat template 的关键字参数（F3：enable_thinking=False 渲染完整空
# think 块 —— 打分位置 = 生成提示之后的最后 token）
CHAT_TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": False}


def _canon_tool_calls(tool_calls) -> Optional[list[dict]]:
    """tool_calls 白名单：``[{id, type, function{name, arguments(dict)}}]``。

    丢弃 litellm 的 ``index`` 字段；``function.arguments`` JSON 字符串 → dict
    （chat template 渲染要求，见模块 docstring）；非法条目跳过。
    """
    if not tool_calls:
        return None
    out: list[dict] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            continue  # 无函数名的残缺调用不进 prompt
        arguments = fn.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                logger.debug("arguments JSON 解析失败，保留原样: %.80s", arguments)
        if not isinstance(arguments, dict):
            arguments = arguments if arguments is not None else {}
        out.append({
            "id": tc.get("id"),
            "type": tc.get("type", "function"),
            "function": {"name": name, "arguments": arguments},
        })
    return out or None


def canonical_assistant(msg: dict) -> dict:
    """assistant 消息白名单（§3.1）：留 role/content/reasoning_content/tool_calls，
    删 ``function_call`` / ``provider_specific_fields`` / ``extra`` / tool_calls 内 ``index``。"""
    content = msg.get("content")
    out: dict = {"role": "assistant", "content": content if content is not None else ""}
    reasoning = msg.get("reasoning_content")
    if reasoning:  # None / 空串不保留空键，减小 parquet 体积
        out["reasoning_content"] = reasoning
    tool_calls = _canon_tool_calls(msg.get("tool_calls"))
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def canonical_tool(msg: dict) -> dict:
    """tool 消息白名单（§3.1）：留 role/content/tool_call_id，删 ``extra``
    （raw_output 等冗余；content 已含 returncode+output 全文）。"""
    content = msg.get("content")
    return {
        "role": "tool",
        "content": content if content is not None else "",
        "tool_call_id": msg.get("tool_call_id"),
    }


def canonical_user(msg: dict) -> dict:
    """user 反馈（tail 中的 FormatError user 消息）白名单（§3.1）：留 role/content。"""
    content = msg.get("content")
    return {"role": "user", "content": content if content is not None else ""}


class TrajectoryPreprocessor:
    """步解析 / 消息规范化 / prompt 组装 / 渲染计长（§3 类规格）。

    Args:
        tokenizer_path: 模型 tokenizer 目录（如
            ``/media/shared_e/models/Qwen3.5-4B``）。``None`` 或 transformers
            不可用时，:meth:`rendered_tokens` 走 3.5 chars/token 近似
            （§1.2：CodeAgentRL 环境无 transformers，报告标注 approx）。
    """

    def __init__(self, tokenizer_path: Optional[str] = None):
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None           # 惰性加载（transformers AutoTokenizer）
        self._tokenizer_failed = False   # 加载失败只报一次

    # ------------------------------------------------------------------
    # tokenizer（惰性、可多进程）
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """transformers tokenizer（惰性）；不可用返回 ``None``。"""
        if self._tokenizer is None and not self._tokenizer_failed and self.tokenizer_path:
            try:
                from transformers import AutoTokenizer  # 惰性导入（CodeAgentRL 环境无 transformers）
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path, trust_remote_code=True)
            except Exception as e:  # pragma: no cover - 依赖环境
                self._tokenizer_failed = True
                logger.warning("tokenizer 加载失败（%s），rendered_tokens 退化为 %.1f chars/token 近似",
                               e, APPROX_CHARS_PER_TOKEN)
        return self._tokenizer

    @property
    def tokenizer_backend(self) -> Optional[str]:
        return "transformers" if self.tokenizer is not None else None

    # ------------------------------------------------------------------
    # 步解析（§3 parse_steps：直接用 mcts/steps.py，不重写解析）
    # ------------------------------------------------------------------

    def parse_steps(self, prefix_json: str | list) -> list[Step]:
        """``prefix_json``（JSON 数组文本或已解析 list）→ ``[Step]``。

        每个元素 = ``Step.to_json()`` 形态的 ``{"assistant": ..., "tail": [...]}``
        （``mcts/steps.py::Step.from_json`` 原样解析，保证与引擎口径一致）。
        """
        if isinstance(prefix_json, str):
            prefix_json = json.loads(prefix_json)
        return [Step.from_json(d) for d in prefix_json]

    # ------------------------------------------------------------------
    # 消息规范化（§3 canonical_messages）
    # ------------------------------------------------------------------

    def canonical_messages(self, steps: list[Step]) -> list[dict]:
        """步序列 → 规范化消息序列（assistant + tail 展开，顺序保持）。

        tail 中的 tool 消息与 FormatError user 反馈按各自白名单处理（§3.1）。
        """
        out: list[dict] = []
        for step in steps:
            out.append(canonical_assistant(step.assistant))
            for msg in step.tail:
                role = msg.get("role")
                if role == "tool":
                    out.append(canonical_tool(msg))
                elif role == "user":
                    out.append(canonical_user(msg))
                # 其它角色（异常数据）不进 PRM prompt
        return out

    # ------------------------------------------------------------------
    # PRM prompt 组装（§4 build_messages）
    # ------------------------------------------------------------------

    def build_messages(self, head_user: str, steps: list[Step]) -> list[dict]:
        """组装 PRM 全 prompt：``[system, user(context), *trajectory, user(instruction)]``。

        - ``head_user`` = head user 内容**原文**（含工具约定，无损，§4.2 规则 1）；
        - 被判定步 = ``steps`` 最后一步（§3.2，标签语义 = "执行完这一步后继续能否成功"）。
        """
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PRM_V1},
            {"role": "user", "content": USER_CONTEXT_V1.format(issue_and_conventions=head_user)},
        ]
        messages.extend(self.canonical_messages(steps))
        messages.append({"role": "user", "content": USER_INSTRUCTION_V1})
        return messages

    # ------------------------------------------------------------------
    # 渲染计长（§3 rendered_tokens）
    # ------------------------------------------------------------------

    def render_text(self, messages: list[dict]) -> str:
        """chat template 渲染文本（``add_generation_prompt=True, enable_thinking=False``）。

        需要 transformers tokenizer；不可用时抛 ``RuntimeError``（计长入口
        :meth:`rendered_tokens` 自行降级，其它调用方显式失败）。
        """
        tok = self.tokenizer
        if tok is None:
            raise RuntimeError("render_text 需要 transformers tokenizer（tokenizer_path 未配置或加载失败）")
        return tok.apply_chat_template(messages, tokenize=False, **CHAT_TEMPLATE_KWARGS)

    def rendered_tokens(self, messages: list[dict]) -> int:
        """渲染后 tokenize 计数（含 nothink 生成提示，F3）。

        有 tokenizer：``apply_chat_template(tokenize=True)`` 的 id 数（与训练期
        collator 看到的序列一致）。无 tokenizer：3.5 chars/token 近似（§1.2——
        CodeAgentRL 环境跑 build_dataset 的核准路径），**必须在报告标注 approx**
        （manifest 的 ``tokenizer_backend=null`` 即标记）。
        """
        tok = self.tokenizer
        if tok is not None:
            ids = tok.apply_chat_template(messages, tokenize=True, **CHAT_TEMPLATE_KWARGS)
            if hasattr(ids, "input_ids"):     # transformers 5.x 返回 BatchEncoding
                ids = ids.input_ids
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return int(len(ids))
        text = self._approx_text(messages)
        return max(1, round(len(text) / APPROX_CHARS_PER_TOKEN))

    @staticmethod
    def _approx_text(messages: list[dict]) -> str:
        """近似计长的文本序列化：role + content + reasoning + tool_calls 参数全文。"""
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m.get('role')}: {m.get('content') or ''}")
            if m.get("reasoning_content"):
                parts.append(str(m["reasoning_content"]))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append(f"{fn.get('name')}: {json.dumps(fn.get('arguments'), ensure_ascii=False)}")
        return "\n".join(parts)


__all__ = [
    "TrajectoryPreprocessor",
    "canonical_assistant",
    "canonical_tool",
    "canonical_user",
    "APPROX_CHARS_PER_TOKEN",
    "CHAT_TEMPLATE_KWARGS",
]
