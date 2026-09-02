# SPDX-License-Identifier: BSD-3-Clause

"""``scripts/run_sglang_qwen4b.sh`` 服务的**手动验证**测试（sglang Qwen3.5-4B）。

用法（先按脚本启动服务，再设置 ``SGLANG_BASE_URL`` 显式开启本组测试）::

    # 1) 启动服务（前台或后台均可）
    bash scripts/run_sglang_qwen4b.sh            # 或 bash scripts/run_sglang_qwen4b.sh --background
    # 2) 运行本组测试（未设置 SGLANG_BASE_URL 时全部跳过）
    SGLANG_BASE_URL=http://localhost:30000/v1 \
        python -m pytest test/test_sglang_server.py -v

验证点（对齐 mini-swe-agent 的实际调用路径）：

1. ``/v1/models`` 健康检查 —— served name（``qwen3.5-4B``）命中；
2. 纯文本 chat completion —— OpenAI 兼容端点基本可用；
3. **tool-call**：带 mini-swe-agent 的 bash 工具 schema 请求 →
   ``assistant.tool_calls[0].function.name == "bash"`` 且 arguments 含 ``command``
   （这是 Agent 决策步的格式基础，脚本未配 ``--tool-call-parser`` 时此测试必挂）；
4. **工具调用往返**：assistant tool_call + tool 结果 → 模型继续（Agent 循环关键行为）；
5. **litellm 路由**（可选）：``litellm.completion(model="openai/qwen3.5-4B",
   api_base=...)`` —— 与 ``mcts.llm.build_model_config`` / mini-swe-agent 完全同路径。

环境变量：``SGLANG_BASE_URL``（必设，默认跳过）、``SGLANG_MODEL``（默认
``qwen3.5-4B``，须与脚本 ``--served-name`` 一致）。
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# 保证 ``agent`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_URL = os.environ.get("SGLANG_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("SGLANG_MODEL", "qwen3.5-4B")

pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="set SGLANG_BASE_URL (e.g. http://localhost:30000/v1) to test the sglang server; "
           "start it first: bash scripts/run_sglang_qwen4b.sh",
)

# mini-swe-agent 的 bash 工具 schema（与 minisweagent/models/utils/actions_toolcall.py
# 的 BASH_TOOL 完全一致，此处内联避免依赖导入）
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _request(method: str, path: str, payload=None, timeout: int = 180):
    """向 sglang OpenAI 兼容端点发请求，返回解析后的 JSON。"""
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        pytest.fail(f"HTTP {e.code} {e.reason} from {url}: {body[:500]}")


def _chat(messages, *, tools=None, temperature: float = 0.2, max_tokens: int = 256):
    """一次 /v1/chat/completions 调用，返回 choices[0].message。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        payload["tools"] = tools
    resp = _request("POST", "/chat/completions", payload)
    choices = resp.get("choices") or []
    assert choices, f"no choices in response: {str(resp)[:300]}"
    return choices[0]["message"]


class TestServerHealth:
    def test_models_endpoint(self):
        """/v1/models 返回模型列表且包含 served name。"""
        data = _request("GET", "/models", timeout=15)
        ids = [m.get("id", "") for m in data.get("data", [])]
        assert any(MODEL in mid for mid in ids), f"served model {MODEL!r} not in {ids}"


class TestChatCompletion:
    def test_basic_text(self):
        """纯文本对话可用（无 tools）。"""
        msg = _chat([{"role": "user", "content": "Reply with exactly: OK"}])
        content = str(msg.get("content") or "")
        assert "OK" in content, f"unexpected content: {content[:200]}"


class TestToolCalling:
    def test_bash_tool_call(self):
        """带 bash 工具 schema 的请求应返回 assistant.tool_calls（name=bash）。"""
        msg = _chat(
            [{"role": "user", "content": "Use the bash tool to print the current directory (pwd)."}],
            tools=[BASH_TOOL],
        )
        tool_calls = msg.get("tool_calls") or []
        assert tool_calls, (
            f"no tool_calls returned (tool-call parsing not working); "
            f"check --tool-call-parser in run_sglang_qwen4b.sh; msg={str(msg)[:300]}"
        )
        fn = tool_calls[0]["function"]
        assert fn["name"] == "bash", f"unexpected tool name: {fn['name']}"
        args = json.loads(fn["arguments"])
        assert isinstance(args.get("command"), str) and args["command"].strip(), (
            f"arguments missing 'command': {fn['arguments']}"
        )

    def test_tool_call_roundtrip(self):
        """工具调用往返：assistant tool_call + tool 结果 → 模型继续（Agent 循环）。"""
        first = _chat(
            [{"role": "user", "content": "Run `pwd` with the bash tool, then say DONE."}],
            tools=[BASH_TOOL],
        )
        tool_calls = first.get("tool_calls") or []
        assert tool_calls, "first turn did not produce a tool call"
        tc = tool_calls[0]
        tid = tc.get("id") or "call_0"

        second = _chat([
            {"role": "user", "content": "Run `pwd` with the bash tool, then say DONE."},
            {"role": "assistant", "content": first.get("content") or "",
             "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": tid, "content": "/repo"},
        ], tools=[BASH_TOOL])
        content = str(second.get("content") or "")
        assert content.strip(), "model did not continue after tool result"
        # 模型应看到 /repo 后收尾（可能再调一次工具或直接回复，这里只要求有输出）

    def test_tool_call_arguments_json(self):
        """arguments 必须是合法 JSON（mini-swe-agent 用 json.loads 解析）。"""
        msg = _chat(
            [{"role": "user", "content": "Use the bash tool to list files: ls -la"}],
            tools=[BASH_TOOL],
        )
        for tc in msg.get("tool_calls") or []:
            json.loads(tc["function"]["arguments"])  # 解析失败即抛异常


class TestLiteLLMRoute:
    def test_litellm_completion_with_tools(self):
        """与 mini-swe-agent 完全同路径的 litellm 调用（openai/<model> + api_base）。"""
        pytest.importorskip("litellm")
        import litellm  # noqa: E402

        resp = litellm.completion(
            model=f"openai/{MODEL}",
            messages=[{"role": "user",
                       "content": "Use the bash tool to run `echo hello`."}],
            tools=[BASH_TOOL],
            api_base=BASE_URL,
            api_key="local-test",
            temperature=0.2,
        )
        message = resp.choices[0].message
        assert message.tool_calls, (
            f"litellm route returned no tool_calls; resp={str(resp)[:400]}"
        )
        assert message.tool_calls[0].function.name == "bash"
