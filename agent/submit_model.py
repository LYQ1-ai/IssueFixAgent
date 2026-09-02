# SPDX-License-Identifier: BSD-3-Clause

"""支持结果提交工具的模型（继承 ``LitellmModel``，不修改 mini-swe-agent 源码）。

默认的 ``LitellmModel._query`` 硬编码 ``tools=[BASH_TOOL]``，且
``parse_toolcall_actions`` 只认 ``bash``。本类只做两处覆写：

- ``_query``：传给 litellm 的工具列表变为 ``[BASH_TOOL, submit_locations]``；
- ``_parse_actions``：新增 ``submit_locations`` 动作的解析与校验（复用官方
  bash 解析），格式非法抛 ``FormatError``（agent 收到反馈后可纠正重试）；
  ``submit_locations`` 必须**单独**调用（不与 bash 并行、不得重复调用），
  对齐 CodeScout 的 sanity check（finish 工具恰好调用一次）。

注入方式：``mcts.llm.build_model_config(..., model_class="agent.submit_model.
SubmitLocationsModel")`` → ``minisweagent.models.get_model`` 按 ``model_class``
import 路径解析（``get_model_class`` 支持完整模块路径），根 rollout（RepoAgent）
与 probe 回放（ReplayRunner）两条路径都经同一 model_config 构造，无需改
mini-swe-agent 任何源码。
"""

from __future__ import annotations

import logging
from typing import Any

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    parse_toolcall_actions,
)

from agent.submit_tool import (
    SUBMIT_MUST_BE_ALONE_MSG,
    SUBMIT_TOOL,
    SUBMIT_TOOL_NAME,
    parse_submit_locations,
)

logger = logging.getLogger("agent.submit_model")

_KNOWN_TOOLS = ("bash", SUBMIT_TOOL_NAME)


class SubmitLocationsModel(LitellmModel):
    """``tools=[bash, submit_locations]`` 的 litellm 模型包装。"""

    def _query(self, messages: list[dict], **kwargs):
        import litellm  # 惰性导入（对齐父类）

        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL, SUBMIT_TOOL],
            **(self.config.model_kwargs | kwargs),
        )

    def _parse_actions(self, response) -> list[dict]:
        """解析工具调用：bash 走官方逻辑；submit_locations 走严格校验。

        规则（对齐 CodeScout sanity check）：
        - 响应必须至少一个工具调用（官方语义：每个响应必须包含工具调用）；
        - 未知工具 → FormatError；
        - ``submit_locations`` 必须恰好一次、且单独调用（不与 bash 同批）。
        """
        tool_calls = response.choices[0].message.tool_calls or []
        template_kwargs = {"finish_reason": response.choices[0].finish_reason}
        if not tool_calls:
            raise FormatError(
                {
                    "role": "user",
                    "content": (
                        "No tool calls found in the response. Every response MUST "
                        "include at least one tool call: use the 'bash' tool to "
                        "explore the repository, or call 'submit_locations' (alone) "
                        "to submit your final result."
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        bash_calls = [tc for tc in tool_calls if tc.function.name == "bash"]
        submit_calls = [tc for tc in tool_calls if tc.function.name == SUBMIT_TOOL_NAME]
        unknown = [tc for tc in tool_calls if tc.function.name not in _KNOWN_TOOLS]
        if unknown:
            names = ", ".join(sorted({tc.function.name for tc in unknown}))
            raise FormatError(
                {
                    "role": "user",
                    "content": f"Unknown tool '{names}'. Available tools: bash, {SUBMIT_TOOL_NAME}.",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        if submit_calls:
            if bash_calls or len(submit_calls) != 1:
                raise FormatError(
                    {
                        "role": "user",
                        "content": SUBMIT_MUST_BE_ALONE_MSG,
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            try:
                return [parse_submit_locations(submit_calls[0])]
            except ValueError as e:
                raise FormatError(
                    {
                        "role": "user",
                        "content": f"Invalid submit_locations call: {e}",
                        "extra": {"interrupt_type": "FormatError"},
                    }
                ) from e
        return parse_toolcall_actions(
            bash_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs=template_kwargs,
        )


__all__ = ["SubmitLocationsModel"]
