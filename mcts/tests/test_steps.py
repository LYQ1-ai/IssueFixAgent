# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.steps`` 轨迹 → 步解析测试（纯逻辑）。"""

from mcts.steps import (
    Step,
    extract_exit,
    messages_for_prefix,
    prefix_node_key,
    split_steps,
    steps_to_messages,
)

from mcts.tests.helpers import make_step, make_steps


def _trajectory_messages() -> list[dict]:
    """构造一段合成轨迹：system + user + (assistant, tool) × 2 + exit。"""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1",
         "extra": {"actions": [{"command": "ls", "tool_call_id": "t1"}]}},
        {"role": "tool", "content": "obs1", "tool_call_id": "t1",
         "extra": {"raw_output": "file1", "returncode": 0}},
        {"role": "assistant", "content": "a2",
         "extra": {"actions": [{"command": "cat x", "tool_call_id": "t2"}]}},
        {"role": "tool", "content": "obs2", "tool_call_id": "t2",
         "extra": {"raw_output": "x", "returncode": 1}},
        {"role": "exit", "content": "submission", "extra": {
            "exit_status": "Submitted", "submission": "diff --git a/x.py b/x.py"}},
    ]


class TestSplitSteps:
    def test_basic_split(self):
        steps = split_steps(_trajectory_messages())
        assert len(steps) == 2
        assert steps[0].content == "a1"
        assert steps[0].commands == ["ls"]
        assert len(steps[0].tail) == 1
        assert steps[1].content == "a2"
        assert steps[1].commands == ["cat x"]

    def test_format_error_user_message_attaches_to_current_step(self):
        msgs = _trajectory_messages()
        # 在 step1 的 tool 后插入 FormatError user 反馈（role=user 且带 interrupt_type）
        msgs.insert(4, {"role": "user", "content": "format error feedback",
                        "extra": {"interrupt_type": "FormatError"}})
        steps = split_steps(msgs)
        assert len(steps) == 2
        assert len(steps[0].tail) == 2  # tool + format-error user 都归 step0

    def test_exit_not_a_step(self):
        steps = split_steps(_trajectory_messages())
        contents = [s.content for s in steps]
        assert "submission" not in contents

    def test_empty_messages(self):
        assert split_steps([]) == []


class TestExtractExit:
    def test_submitted(self):
        status, submission = extract_exit(_trajectory_messages())
        assert status == "Submitted"
        assert "diff --git" in submission

    def test_no_exit(self):
        status, submission = extract_exit([{"role": "system", "content": "s"}])
        assert (status, submission) == ("", "")


class TestMessagesForPrefix:
    def test_k0_keeps_system_user(self):
        head = messages_for_prefix(_trajectory_messages(), 0)
        assert [m["role"] for m in head] == ["system", "user"]

    def test_k1_includes_first_step_observations(self):
        prefix = messages_for_prefix(_trajectory_messages(), 1)
        roles = [m["role"] for m in prefix]
        assert roles == ["system", "user", "assistant", "tool"]
        assert prefix[-1]["tool_call_id"] == "t1"

    def test_k_large_returns_all_but_exit(self):
        prefix = messages_for_prefix(_trajectory_messages(), 99)
        assert all(m["role"] != "exit" for m in prefix)
        assert prefix[-1]["role"] == "tool"


class TestPrefixNodeKey:
    def test_root_key(self):
        assert prefix_node_key([]) == "root"

    def test_content_based(self):
        a = make_steps(["s1", "s2"])
        b = make_steps(["s1", "s2"])
        c = make_steps(["s1", "s3"])
        assert prefix_node_key(a) == prefix_node_key(b)
        assert prefix_node_key(a) != prefix_node_key(c)
        assert prefix_node_key(a) != "root"

    def test_roundtrip_via_to_json(self):
        steps = make_steps(["s1", "s2"])
        key1 = prefix_node_key(steps)
        restored = [Step.from_json(s.to_json()) for s in steps]
        assert prefix_node_key(restored) == key1


class TestStepsToMessages:
    def test_roundtrip(self):
        steps = split_steps(_trajectory_messages())
        msgs = steps_to_messages(steps)
        # 还原 = assistant + tool 对（不含 system/user/exit）
        assert [m["role"] for m in msgs] == ["assistant", "tool", "assistant", "tool"]
