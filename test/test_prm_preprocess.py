# SPDX-License-Identifier: BSD-3-Clause

"""prm/preprocess.py 单测（docs/prm_training_plan.md §10）。

覆盖：Step 解析往返一致（对齐 mcts/steps.py）、规范化白名单（F5）、
build_messages 四段结构 + 无 gold 泄漏、近似计长。
chat template 渲染行为（F3 空 think 块 / reasoning 渲染语义）依赖 transformers，
单独类 gating（PRM 环境可跑，CodeAgentRL 环境自动 skip）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from prm_fixtures import _HEAD, make_step  # noqa: E402

from mcts.steps import Step, split_steps  # noqa: E402

from prm.preprocess import (  # noqa: E402
    APPROX_CHARS_PER_TOKEN,
    TrajectoryPreprocessor,
    canonical_assistant,
    canonical_tool,
    canonical_user,
)
from prm.prompts import SYSTEM_PRM_V1, USER_CONTEXT_V1, USER_INSTRUCTION_V1  # noqa: E402


# ---------------------------------------------------------------------------
# 规范化白名单（F5）
# ---------------------------------------------------------------------------

class TestCanonical:
    def test_assistant_whitelist_keeps(self):
        raw_msg = make_step(1)["assistant"]
        out = canonical_assistant(raw_msg)
        assert out["role"] == "assistant"
        assert out["content"] == "\n\n"
        assert out["reasoning_content"] == "step 1 reasoning"
        assert out["tool_calls"] == [{
            "id": "call_1", "type": "function",
            "function": {"name": "bash", "arguments": {"command": "ls"}},
        }]

    def test_assistant_whitelist_drops(self):
        out = canonical_assistant(make_step(1)["assistant"])
        # F5 冗余字段全部丢弃
        for dropped in ("function_call", "provider_specific_fields", "extra"):
            assert dropped not in out
        # tool_calls 内的 index 丢弃
        assert "index" not in out["tool_calls"][0]

    def test_assistant_arguments_string_parsed_to_dict(self):
        """arguments JSON 字符串 → dict（chat template 的 |items 过滤器要求 mapping）。"""
        out = canonical_assistant(make_step(2)["assistant"])
        assert isinstance(out["tool_calls"][0]["function"]["arguments"], dict)
        assert out["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}

    def test_assistant_arguments_unparseable_kept(self):
        msg = {"role": "assistant", "content": "",
               "tool_calls": [{"id": "c", "type": "function",
                               "function": {"name": "bash", "arguments": "not-json"}}]}
        assert canonical_assistant(msg)["tool_calls"][0]["function"]["arguments"] == "not-json"

    def test_assistant_no_tool_calls_no_key(self):
        out = canonical_assistant({"role": "assistant", "content": "done"})
        assert "tool_calls" not in out

    def test_assistant_none_content_normalized(self):
        out = canonical_assistant({"role": "assistant", "content": None})
        assert out["content"] == ""

    def test_tool_whitelist(self):
        raw_msg = make_step(1)["tail"][0]
        out = canonical_tool(raw_msg)
        assert out == {"role": "tool", "tool_call_id": "call_1",
                       "content": json.dumps({"returncode": 0, "output": "ok"})}
        assert "extra" not in out  # raw_output 等冗余丢弃

    def test_user_feedback_whitelist(self):
        msg = {"role": "user", "content": "Format error...", "extra": {"interrupt_type": "FormatError"}}
        assert canonical_user(msg) == {"role": "user", "content": "Format error..."}


# ---------------------------------------------------------------------------
# 步解析（直接复用 mcts/steps.py，不重写解析）
# ---------------------------------------------------------------------------

class TestParseSteps:
    def test_roundtrip_with_mcts_steps(self):
        steps = [make_step(i) for i in (1, 2, 3)]
        pre = TrajectoryPreprocessor()
        parsed = pre.parse_steps(json.dumps(steps))
        assert parsed == [Step.from_json(s) for s in steps]
        # 与引擎同口径：解析结果参与 split_steps 的轨迹切片语义一致
        assert [s.commands for s in parsed] == [["ls"], ["ls"], ["ls"]]
        # to_json 往返与原文一致（prefix_node_key 稳定的前提）
        assert [s.to_json() for s in parsed] == steps

    def test_accepts_preparsed_list(self):
        steps = [make_step(1)]
        assert TrajectoryPreprocessor().parse_steps(steps) == [Step.from_json(steps[0])]


# ---------------------------------------------------------------------------
# build_messages（§4）
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_four_section_structure(self):
        pre = TrajectoryPreprocessor()
        steps = [Step.from_json(make_step(i)) for i in (1, 2)]
        msgs = pre.build_messages("ISSUE TEXT", steps)
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "tool", "assistant", "tool", "user"]
        assert msgs[0]["content"] == SYSTEM_PRM_V1
        assert msgs[1]["content"] == USER_CONTEXT_V1.format(issue_and_conventions="ISSUE TEXT")
        assert msgs[-1]["content"] == USER_INSTRUCTION_V1
        # 轨迹段 = 规范化后的 assistant+tail（含被判定步 = 最后一步）
        assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
        assert msgs[3]["role"] == "tool"

    def test_issue_text_verbatim_and_no_gold_leak(self):
        pre = TrajectoryPreprocessor()
        head_user = "Fix the widget bug.\n<issue_description>...</issue_description>"
        msgs = pre.build_messages(head_user, [Step.from_json(make_step(1))])
        assert head_user in msgs[1]["content"]  # 原文无损（§4.2 规则 1）
        blob = json.dumps(msgs, ensure_ascii=False)
        for gold_hint in ("gold_patch", "gold_locations", "expected_patch"):
            assert gold_hint not in blob  # 严禁 gold 信息进入 prompt（§4.2 规则 2）

    def test_reasoning_preserved_in_messages(self):
        """canonical messages 完整保留 reasoning_content（§3.1，升级 rationale 的原料）。"""
        pre = TrajectoryPreprocessor()
        steps = [Step.from_json(make_step(1))]
        msgs = pre.build_messages("ISSUE", steps)
        assert msgs[2]["reasoning_content"] == "step 1 reasoning"


# ---------------------------------------------------------------------------
# rendered_tokens（近似路径；tokenizer 路径见 TestChatTemplateRendering）
# ---------------------------------------------------------------------------

class TestRenderedTokensApprox:
    def test_positive_int_and_grows_with_steps(self):
        pre = TrajectoryPreprocessor()  # 无 tokenizer → 近似
        s1 = [Step.from_json(make_step(1))]
        s3 = [Step.from_json(make_step(i)) for i in (1, 2, 3)]
        t1 = pre.rendered_tokens(pre.build_messages("ISSUE", s1))
        t3 = pre.rendered_tokens(pre.build_messages("ISSUE", s3))
        assert isinstance(t1, int) and t1 > 0
        assert t3 > t1

    def test_tokenizer_backend_none(self):
        assert TrajectoryPreprocessor().tokenizer_backend is None

    def test_approx_chars_per_token_constant(self):
        assert APPROX_CHARS_PER_TOKEN == 3.5


# ---------------------------------------------------------------------------
# chat template 渲染行为（需 transformers + 本地 tokenizer；PRM 环境跑）
# ---------------------------------------------------------------------------

_TOKENIZER_DIR = os.environ.get("PRM_TEST_TOKENIZER", "/media/shared_e/models/Qwen3.5-4B")


class TestChatTemplateRendering:
    @pytest.fixture
    def pre(self):
        transformers = pytest.importorskip("transformers")
        if not Path(_TOKENIZER_DIR).exists():
            pytest.skip(f"tokenizer 不存在: {_TOKENIZER_DIR}")
        tok = transformers.AutoTokenizer.from_pretrained(_TOKENIZER_DIR, trust_remote_code=True)
        return TrajectoryPreprocessor(_TOKENIZER_DIR)

    def test_backend(self, pre):
        assert pre.tokenizer_backend == "transformers"

    def test_nothink_generation_prompt_ends_with_empty_think_block(self, pre):
        """F3 固化：enable_thinking=False 渲染完整空 think 块（开闭标签都在）。"""
        steps = [Step.from_json(make_step(1))]
        text = pre.render_text(pre.build_messages("ISSUE", steps))
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_rendered_tokens_matches_text_encoding(self, pre):
        steps = [Step.from_json(make_step(i)) for i in (1, 2)]
        msgs = pre.build_messages("ISSUE", steps)
        n_ids = pre.rendered_tokens(msgs)
        n_txt = len(pre.tokenizer.encode(pre.render_text(msgs)))
        assert n_ids == n_txt

    def test_history_reasoning_template_behavior(self, pre):
        """推理链渲染语义（模板 last_query_index 行为固化，防模板漂移无感知）。

        Qwen3.5 chat template 仅对"最后一条真实 user 查询**之后**"的 assistant
        渲染 ``<think>``；v1 PRM 布局的末轮判定指令是最后一条 user 查询，因此
        轨迹 assistant 的 reasoning_content 不进入渲染文本（但完整保留在
        canonical messages / parquet 中，见 TestBuildMessages::
        test_reasoning_preserved_in_messages）。此处固化该行为：若未来模板或
        prompt 布局变化导致 reasoning 进入渲染，本测试会失败提示复核。
        """
        steps = [Step.from_json(make_step(1))]
        text = pre.render_text(pre.build_messages("ISSUE", steps))
        assert "step 1 reasoning" not in text

    def test_tool_response_rendered(self, pre):
        steps = [Step.from_json(make_step(1))]
        text = pre.render_text(pre.build_messages("ISSUE", steps))
        assert "<tool_response>" in text          # tool 消息渲染为 tool_response 块
        assert "<tool_call>" in text              # assistant tool_calls 渲染为 tool_call 块
        assert "ISSUE" in text                    # head user 原文在


# ---------------------------------------------------------------------------
# 与 mcts 轨迹的端到端一致性（split_steps → parse_steps 同源）
# ---------------------------------------------------------------------------

class TestTrajectoryConsistency:
    def test_split_steps_then_parse_steps_roundtrip(self):
        """完整轨迹消息 → split_steps → to_json → parse_steps 还原一致。"""
        messages = [dict(_HEAD[0]), dict(_HEAD[1])]
        for i in (1, 2, 3):
            messages.append(make_step(i)["assistant"])
            messages.extend(make_step(i)["tail"])
        steps = split_steps(messages)
        blob = json.dumps([s.to_json() for s in steps])
        reparsed = TrajectoryPreprocessor().parse_steps(blob)
        assert [s.to_json() for s in reparsed] == [s.to_json() for s in steps]
