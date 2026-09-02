# SPDX-License-Identifier: BSD-3-Clause

"""结果提交工具（仿 CodeScout ``localization_finish``，PLAN §2.2 提交协议改造）。

Agent 通过**工具调用**输出根因位置（不再依赖 bash 魔法串）：一次 ``submit_locations``
调用 = 提交最终定位结果 = 任务结束（``exit_status="Submitted"``，
``submission=locations 的 JSON``）。

本模块只包含**纯逻辑**（不 import minisweagent / agent 运行时）：

- :data:`SUBMIT_TOOL`：OpenAI function-calling schema，与 bash 工具一起传给
  litellm（``tools=[BASH_TOOL, SUBMIT_TOOL]``）；
- :func:`parse_submit_locations`：agent 侧**严格**校验并规范化一次 submit 调用
  （file 必填、路径相对仓库根、禁止重复条目、禁止空列表），格式非法抛
  ``ValueError``（由 model 层转 FormatError 反馈给 agent 重试）；
- :func:`SUBMIT_REMINDER`：轮次耗尽 / 提前结束时插入的 user 催收消息（要求 agent
  立即用 submit_locations 输出结果）。
- 判定侧的宽松解析（``mcts.reward.locations_from_submission``）与填充规则说明见
  :data:`SUBMIT_TOOL_DESCRIPTION`（对齐 CodeScout ``TOOL_DESCRIPTION`` 的规则
  1–4 与 IMPORTANT 1–5）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

SUBMIT_TOOL_NAME = "submit_locations"

SUBMIT_TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_TOOL_NAME,
        "description": SUBMIT_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "description": (
                        "List of code locations to modify. Each location must have "
                        "'file' (required, relative to repo root, no leading './'), "
                        "and optionally 'class_name' and 'function_name'. Omit "
                        "class_name for changes to imports/global variables/global "
                        "functions; omit function_name for file- or class-level edits."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Path to the file relative to the repository root (required).",
                            },
                            "class_name": {
                                "type": "string",
                                "description": "Class name (optional).",
                            },
                            "function_name": {
                                "type": "string",
                                "description": "Function/method name (optional).",
                            },
                        },
                        "required": ["file"],
                    },
                }
            },
            "required": ["locations"],
        },
    },
}

# 轮次耗尽 / 提前结束但未提交时的 user 催收消息（SubmitAgent 注入，给最后一次机会）
SUBMIT_REMINDER = (
    "The task is about to end without a final submission. You must now call the "
    "`submit_locations` tool exactly once (alone, without any bash tool call) to "
    "output your final list of code locations. This is your last turn; if you do "
    "not submit, the task will end and your result will be scored as empty."
)

# submit_locations 与 bash 同批调用、或重复调用 submit 时的 FormatError 反馈
SUBMIT_MUST_BE_ALONE_MSG = (
    "The submit_locations tool must be called exactly once, alone in a single "
    "response (do not combine it with bash tool calls, and do not call it more "
    "than once). Call submit_locations as the only tool call of your final response."
)


def parse_submit_locations(tool_call: Any) -> dict:
    """严格解析并校验一次 ``submit_locations`` 工具调用。

    Args:
        tool_call: OpenAI 风格 tool call 对象（``.function.arguments`` / ``.id``）。

    Returns:
        规范化的动作 dict：``{"tool": "submit_locations", "locations": [...],
        "tool_call_id": ...}``，其中每条 location 为
        ``{"file": str, "class_name": str|None, "function_name": str|None}``。

    Raises:
        ValueError: 格式非法（参数解析失败 / 缺 locations / 空列表 / 缺 file /
            重复条目 / 字段类型错误）—— 由 model 层转成 FormatError 反馈给 agent。
    """
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception as e:  # noqa: BLE001 - 统一按格式错误处理
        raise ValueError(f"Error parsing submit_locations arguments: {e}.") from e
    if not isinstance(args, dict) or "locations" not in args:
        raise ValueError(
            "submit_locations requires a single 'locations' argument (a list of "
            "code locations, each with a required 'file' path)."
        )
    locations = args["locations"]
    if not isinstance(locations, list):
        raise ValueError("'locations' must be a list.")
    if not locations:
        raise ValueError("'locations' must contain at least one code location.")
    out: list[dict] = []
    seen: set[tuple] = set()
    for item in locations:
        if not isinstance(item, dict):
            raise ValueError(
                "Each location must be an object with a required 'file' string and "
                "optional 'class_name' / 'function_name' strings."
            )
        file = item.get("file")
        if not isinstance(file, str) or not file.strip():
            raise ValueError("Each location must have a non-empty 'file' path.")
        file = file.strip()
        class_name = item.get("class_name")
        function_name = item.get("function_name")
        if class_name is not None and not isinstance(class_name, str):
            raise ValueError("'class_name' must be a string or null.")
        if function_name is not None and not isinstance(function_name, str):
            raise ValueError("'function_name' must be a string or null.")
        key = (file, class_name, function_name)
        if key in seen:
            raise ValueError(
                "Duplicate location entries (same file / class / function) are not "
                "allowed. Merge them into a single entry."
            )
        seen.add(key)
        out.append({
            "file": file,
            "class_name": class_name,
            "function_name": function_name,
        })
    return {"tool": SUBMIT_TOOL_NAME, "locations": out, "tool_call_id": tool_call.id}


__all__ = [
    "SUBMIT_TOOL_NAME",
    "SUBMIT_TOOL",
    "SUBMIT_TOOL_DESCRIPTION",
    "SUBMIT_REMINDER",
    "SUBMIT_MUST_BE_ALONE_MSG",
    "parse_submit_locations",
]
