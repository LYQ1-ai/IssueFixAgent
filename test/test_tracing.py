"""``agent/tracing.py``（Phoenix 追踪接入）测试。

用法（在项目根目录 /home/lyq/PycharmProjects/CodeAgentRL 下执行）::

    python -m pytest test/test_tracing.py -v

说明：
- trace 上下文注入 / 导出逻辑为纯单元测试（无需 Docker / Phoenix 服务端）；
- 涉及 ``phoenix.otel.register`` / ``LiteLLMInstrumentor`` 的用例在未安装
  arize-phoenix-otel 时自动跳过；
- 注意：启用追踪的用例会注册全局 TracerProvider 并插桩 litellm（进程内幂等），
  测试结束时通过 uninstrument 恢复，不影响其他测试文件。
"""

import json
import sys
from pathlib import Path

import pytest

# 保证 ``agent`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tracing import (  # noqa: E402
    end_trace_context,
    export_trace,
    save_trace_json,
    setup_phoenix_tracing,
    start_trace_context,
)

_TRACE_ID = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"


# ----------------------------------------------------------------------
# trace 上下文注入
# ----------------------------------------------------------------------


def test_start_trace_context_injects_trace_id():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    token, trace_id = start_trace_context()
    try:
        span = tracer.start_span("hello")
        span.end()
    finally:
        end_trace_context(token)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert format(spans[0].get_span_context().trace_id, "032x") == trace_id


def test_start_trace_context_with_explicit_trace_id():
    token, trace_id = start_trace_context(trace_id=_TRACE_ID)
    try:
        assert trace_id == _TRACE_ID
    finally:
        end_trace_context(token)


def test_start_trace_context_rejects_invalid_trace_id():
    with pytest.raises(ValueError):
        start_trace_context(trace_id="not-hex")
    with pytest.raises(ValueError):
        start_trace_context(trace_id="ab" * 20)  # 长度非法


def test_end_trace_context_restores_context():
    from opentelemetry import context as context_api

    token, _ = start_trace_context()
    end_trace_context(token)
    assert context_api.get_current() is not None  # 可正常继续使用上下文


# ----------------------------------------------------------------------
# trace 导出（REST 调用，mock urllib）
# ----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_export_trace_builds_url_and_paginates(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        if "cursor=" not in req.full_url:
            return _FakeResponse({"data": [{"name": "span-a"}], "next_cursor": "Span:2"})
        return _FakeResponse({"data": [{"name": "span-b"}], "next_cursor": None})

    monkeypatch.setattr("agent.tracing.urllib.request.urlopen", fake_urlopen)
    out = export_trace(_TRACE_ID, project_name="myproj", base_url="http://px:6006")
    assert len(calls) == 2
    assert calls[0].startswith("http://px:6006/v1/projects/myproj/spans?")
    assert f"trace_id={_TRACE_ID}" in calls[0]
    assert "limit=1000" in calls[0]
    assert "cursor=" in calls[1]
    assert out["spans"] == [{"name": "span-a"}, {"name": "span-b"}]
    assert out["project"] == "myproj"


def test_export_trace_http_error(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr("agent.tracing.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        export_trace(_TRACE_ID, project_name="myproj", base_url="http://px:6006")


def test_save_trace_json(tmp_path):
    out_path = save_trace_json(
        {"trace_id": _TRACE_ID, "spans": [{"name": "x"}]},
        tmp_path / "trace.json",
    )
    saved = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert saved["trace_id"] == _TRACE_ID


def test_save_trace_json_creates_parent_dirs(tmp_path):
    # 父目录不存在时应自动创建（如 outputs/traces/<trace_id>.json）
    out_path = save_trace_json(
        {"trace_id": _TRACE_ID, "spans": []},
        tmp_path / "a" / "b" / f"{_TRACE_ID}.json",
    )
    assert Path(out_path).is_file()
    assert json.loads(Path(out_path).read_text(encoding="utf-8"))["trace_id"] == _TRACE_ID


# ----------------------------------------------------------------------
# RepoAgent 集成（run() 返回 trace_id）
# ----------------------------------------------------------------------


class _DummyEnv:
    def __init__(self, *args, **kwargs):
        pass


class _DummyAgent:
    def __init__(self, model, env, **kwargs):
        self.cost = 0.0
        self.n_calls = 0

    def run(self, task, **kwargs):
        return {"exit_status": 0, "submission": "done"}

    def serialize(self):
        return {"trajectory": []}


@pytest.fixture
def fake_run_env(monkeypatch):
    """把 run() 的容器/Agent 环节替换为假实现，专注测追踪逻辑。"""
    import agent.base_agent as ba

    monkeypatch.setattr(ba, "get_env", lambda *a, **k: "fake-container")
    monkeypatch.setattr(ba, "release_env", lambda *a, **k: None)
    monkeypatch.setattr(ba, "AttachContainerEnvironment", _DummyEnv)
    monkeypatch.setattr(ba, "DefaultAgent", _DummyAgent)
    monkeypatch.setattr(ba, "get_model", lambda *a, **k: object())
    return ba


def test_repo_agent_returns_trace_id_when_enabled(fake_run_env):
    pytest.importorskip("phoenix.otel")
    pytest.importorskip("openinference.instrumentation.litellm")

    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    agent = fake_run_env.RepoAgent(
        "owner/repo",
        phoenix_tracing={"trace_id": _TRACE_ID, "project_name": "testproj"},
    )
    try:
        result = agent.run()
        assert result["trace_id"] == _TRACE_ID
        assert result["tracing"]["project_name"] == "testproj"
        assert result["tracing"]["base_url"] == "http://localhost:6006"
    finally:
        try:
            LiteLLMInstrumentor().uninstrument()
        except Exception:  # pragma: no cover
            pass


def test_repo_agent_trace_id_none_when_disabled(fake_run_env):
    agent = fake_run_env.RepoAgent("owner/repo")
    result = agent.run()
    assert result["trace_id"] is None
    assert result["tracing"] is None


def test_setup_phoenix_tracing_is_idempotent():
    pytest.importorskip("phoenix.otel")
    pytest.importorskip("openinference.instrumentation.litellm")

    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    try:
        assert setup_phoenix_tracing(project_name="p1") is True
        assert setup_phoenix_tracing(project_name="p2") is True  # 幂等：不重复注册
    finally:
        try:
            LiteLLMInstrumentor().uninstrument()
        except Exception:  # pragma: no cover
            pass
