# SPDX-License-Identifier: BSD-3-Clause

"""Phoenix 追踪接入（可选）：为 Agent 每次执行提供 OTLP 轨迹上报与 trace 导出。

设计（与 :mod:`agent.base_agent` 的 :class:`RepoAgent` 配合使用）::

    # 1. 进程内一次性注册 phoenix tracer 并插桩 litellm（幂等）
    setup_phoenix_tracing(endpoint="http://localhost:4317", project_name="codeagentrl")

    # 2. 每次任务执行：注入本次运行的 trace 上下文
    #    （期间所有 litellm 调用 —— mini-swe-agent 的每个模型请求 —— 共享同一 trace_id）
    token, trace_id = start_trace_context()      # 不传 trace_id 则自动生成
    try:
        ...  # 运行 Agent
    finally:
        end_trace_context(token)

    # 3. 按 trace_id 从 phoenix 服务端导出本次执行轨迹（执行轨迹 = MCTS 采样的数据源）
    data = export_trace(trace_id, project_name="codeagentrl")
    save_trace_json(data, "traces/xxx.json")

依赖（CodeAgentRL 环境已装）：``arize-phoenix-otel``、``openinference-instrumentation-litellm``、
``opentelemetry-*``。全部在 :func:`setup_phoenix_tracing` 内**惰性导入**，
未启用追踪时本模块不产生任何开销、也不引入任何运行期依赖。

环境变量：
    PHOENIX_COLLECTOR_ENDPOINT   OTLP collector 地址（默认 http://localhost:4317，gRPC）
    PHOENIX_PROJECT              Phoenix 项目名（默认 codeagentrl）
    PHOENIX_REST_URL             Phoenix REST API 基址，导出 trace 用（默认 http://localhost:6006）
    PHOENIX_API_KEY              服务端启用认证时的 API key（可选）
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("agent.tracing")

# 进程内状态：register / instrument 每个进程只执行一次（幂等）
_state: dict[str, bool] = {"registered": False, "instrumented": False}


# ---------------------------------------------------------------------------
# 启用追踪（进程内一次性）
# ---------------------------------------------------------------------------


def setup_phoenix_tracing(
    *,
    endpoint: Optional[str] = None,
    project_name: Optional[str] = None,
) -> bool:
    """注册 Phoenix tracer 并插桩 litellm（进程内幂等）。

    首次调用执行 ``phoenix.otel.register`` + ``LiteLLMInstrumentor().instrument()``；
    之后同一进程内的调用直接复用。endpoint / project_name 只在首次生效。

    Args:
        endpoint: OTLP collector 地址；缺省读 ``PHOENIX_COLLECTOR_ENDPOINT``，
            再缺省 ``http://localhost:4317``（Phoenix 默认 OTLP gRPC 端口）。
        project_name: Phoenix 项目名；缺省读 ``PHOENIX_PROJECT``，再缺省 ``codeagentrl``。

    Returns:
        True（成功启用）。

    Raises:
        ImportError: 未安装 arize-phoenix-otel / openinference-instrumentation-litellm。
    """
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from phoenix.otel import register

    endpoint = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
    project_name = project_name or os.getenv("PHOENIX_PROJECT", "codeagentrl")

    tracer_provider = None
    if not _state["registered"]:
        tracer_provider = register(project_name=project_name, endpoint=endpoint)
        _state["registered"] = True
    if not _state["instrumented"]:
        # tracer_provider 为 None 时使用 register 设置的全局 provider
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        _state["instrumented"] = True
    logger.info("Phoenix tracing enabled (endpoint=%s, project=%s)", endpoint, project_name)
    return True


# ---------------------------------------------------------------------------
# 每次执行的 trace 上下文（trace_id 的生成与注入）
# ---------------------------------------------------------------------------


def start_trace_context(trace_id: Optional[str] = None) -> tuple[Any, str]:
    """为一次 Agent 执行注入确定的 trace 上下文，返回 ``(token, trace_id)``。

    - trace_id 缺省时自动生成 32 位十六进制（OTel trace id 格式）；
    - 注入后，本次运行期间（同一线程）创建的所有 span —— 包括 mini-swe-agent
      每个 ``litellm.completion`` 调用 —— 都归属该 trace；
    - 执行结束必须调用 :func:`end_trace_context` 撤销（token 为 OTel context token）；
    - 支持外部传入 trace_id（例如 MCTS 采样需要把轨迹与采样节点关联时）。

    Args:
        trace_id: 可选的 32 位十六进制 trace id；缺省自动生成。

    Raises:
        ValueError: trace_id 非法（长度或字符）。
    """
    from opentelemetry import context as context_api
    from opentelemetry import trace as trace_api
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    trace_id = (trace_id or secrets.token_hex(16)).lower()
    if len(trace_id) != 32 or any(c not in "0123456789abcdef" for c in trace_id):
        raise ValueError(f"trace_id must be 32 lowercase hex chars, got: {trace_id!r}")
    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(secrets.token_hex(8), 16),
        is_remote=True,  # 远程父上下文：子 span 直接继承该 trace_id 且保持采样
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    ctx = trace_api.set_span_in_context(NonRecordingSpan(span_context))
    token = context_api.attach(ctx)
    return token, trace_id


def end_trace_context(token: Any) -> None:
    """撤销 :func:`start_trace_context` 注入的上下文。"""
    from opentelemetry import context as context_api

    context_api.detach(token)


# ---------------------------------------------------------------------------
# trace 导出（执行轨迹的保存 / 读取方式，可作 MCTS 采样数据）
# ---------------------------------------------------------------------------


def _rest_base_url(base_url: Optional[str]) -> str:
    return (base_url or os.getenv("PHOENIX_REST_URL", "http://localhost:6006")).rstrip("/")


def export_trace(
    trace_id: str,
    *,
    project_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: int = 1000,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """从 phoenix 服务端按 trace_id 导出该次执行的全部 span（执行轨迹）。

    调用 Phoenix REST 接口：``GET /v1/projects/{project}/spans?trace_id=...``，
    自动翻页（每页上限 1000）。

    Returns:
        字典 ``{"trace_id", "project", "base_url", "spans": [...]}``。spans 每项含
        ``name`` / ``context``（trace_id, span_id）/ ``parent_id`` / ``attributes`` /
        ``start_time`` / ``end_time`` / ``status_code`` / ``events``，可按
        ``parent_id`` 重建 span 树 —— 即 Agent 一次任务执行的完整轨迹，
        可直接 ``save_trace_json`` 落盘作为后续 MCTS 采样的数据。

    Raises:
        RuntimeError: 服务端不可达或返回错误（含 HTTP 状态码）。
    """
    project = project_name or os.getenv("PHOENIX_PROJECT", "codeagentrl")
    base = _rest_base_url(base_url)
    query = urllib.parse.urlencode({"trace_id": trace_id, "limit": min(int(limit), 1000)})
    path = f"/v1/projects/{urllib.parse.quote(project)}/spans?{query}"
    headers: dict[str, str] = {}
    key = api_key or os.getenv("PHOENIX_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    spans: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        url = base + path + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Failed to export trace {trace_id} from {base}: HTTP {e.code} {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to export trace {trace_id} from {base}: {e.reason}"
            ) from e
        spans.extend(payload.get("data") or [])
        cursor = payload.get("next_cursor")
        if not cursor:
            break
    return {"trace_id": trace_id, "project": project, "base_url": base, "spans": spans}


def save_trace_json(trace: dict[str, Any], path: Union[str, os.PathLike]) -> str:
    """把 :func:`export_trace` 的结果写入 JSON 文件（MCTS 采样轨迹的保存方式）。

    父目录不存在时自动创建（如 ``outputs/traces/<trace_id>.json``）。
    """
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    logger.info("Trace saved to %s", path)
    return path


__all__ = [
    "setup_phoenix_tracing",
    "start_trace_context",
    "end_trace_context",
    "export_trace",
    "save_trace_json",
]
