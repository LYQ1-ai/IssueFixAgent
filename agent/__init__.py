"""Agent 模块：执行环境管理与只读 shell 工具。

- ``init_env``: 执行环境（Docker 容器）管理器 —— 按 (仓库, commit) 创建/缓存/
  复用容器，容器内 clone 仓库并 checkout 指定 commit；负责全部环境生命周期。
- ``shell_tool``: 只读 shell 工具 —— 只做命令级只读校验并在给定环境容器内
  执行 ``docker exec``，不涉及任何环境创建/管理（环境由 init_env 提供）。
- ``base_agent``: 基础 agent 实现 —— 在 init_env 获取的指定容器中运行
  mini-swe-agent（不修改其源码）：执行前获取容器、执行完成后释放。
- ``tracing``: Phoenix 追踪接入（可选）—— 为每次 Agent 执行注入 trace_id
  并上报 litellm 调用轨迹；可按 trace_id 导出执行轨迹（MCTS 采样数据）。
- ``submit_tool`` / ``submit_model`` / ``submit_agent``: 结果提交工具
  （``submit_locations``，仿 CodeScout localization_finish）—— 工具 schema /
  支持该工具的模型（tools=[bash, submit_locations]）/ 提交即结束 + 轮次耗尽
  催收的 agent（PLAN §2.2 提交协议改造）。
"""

from agent.base_agent import (
    AttachContainerConfig,
    AttachContainerEnvironment,
    RepoAgent,
    main as base_agent_main,
)
from agent.init_env import (
    EnvConfig,
    EnvManager,
    get_env,
    release_env,
)
from agent.shell_tool import (
    ReadOnlyViolation,
    ShellTool,
    ShellToolConfig,
    validate_readonly,
)
from agent.submit_agent import SubmitAgent, resolve_agent_class
from agent.submit_model import SubmitLocationsModel
from agent.submit_tool import (
    SUBMIT_REMINDER,
    SUBMIT_TOOL,
    SUBMIT_TOOL_DESCRIPTION,
    SUBMIT_TOOL_NAME,
    parse_submit_locations,
)
from agent.tracing import (
    end_trace_context,
    export_trace,
    save_trace_json,
    setup_phoenix_tracing,
    start_trace_context,
)

__all__ = [
    # init_env: 环境管理
    "EnvConfig",
    "EnvManager",
    "get_env",
    "release_env",
    # shell_tool: 只读 shell 工具
    "ReadOnlyViolation",
    "ShellTool",
    "ShellToolConfig",
    "validate_readonly",
    # base_agent: 在指定容器中运行 mini-swe-agent
    "AttachContainerConfig",
    "AttachContainerEnvironment",
    "RepoAgent",
    "base_agent_main",
    # submit_*: 结果提交工具（PLAN §2.2 提交协议改造）
    "SubmitAgent",
    "resolve_agent_class",
    "SubmitLocationsModel",
    "SUBMIT_TOOL_NAME",
    "SUBMIT_TOOL",
    "SUBMIT_TOOL_DESCRIPTION",
    "SUBMIT_REMINDER",
    "parse_submit_locations",
    # tracing: Phoenix 追踪（可选）
    "setup_phoenix_tracing",
    "start_trace_context",
    "end_trace_context",
    "export_trace",
    "save_trace_json",
]
