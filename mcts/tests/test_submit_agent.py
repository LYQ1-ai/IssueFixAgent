# SPDX-License-Identifier: BSD-3-Clause

"""``SubmitAgent`` 行为测试：submit_locations 提交即结束 + 轮次耗尽/提前结束催收。

用脚本化 FakeModel / FakeEnv 驱动真实 ``SubmitAgent``（继承自官方
``DefaultAgent``，不修改其源码）：覆盖提交协议（exit_status=Submitted、
submission=locations JSON）、格式错误恢复、轮次上限催收（恰好一次 + 放行一轮）
与 RepeatedFormatError 提前结束催收。
"""

import json

import pytest

from agent.submit_agent import SubmitAgent, resolve_agent_class
from agent.submit_tool import SUBMIT_REMINDER, SUBMIT_TOOL_NAME
from minisweagent.exceptions import FormatError, LimitsExceeded, Submitted


def make_assistant(actions, content="analyzing..."):
    return {"role": "assistant", "content": content, "extra": {"actions": actions}}


def bash_action(cmd="ls -la", tid="t1"):
    return {"command": cmd, "tool_call_id": tid}


def submit_action(locations, tid="t2"):
    return {"tool": SUBMIT_TOOL_NAME, "locations": locations, "tool_call_id": tid}


class FakeModel:
    """脚本化模型：按序返回预设 assistant 消息；记录每次调用看到的对话。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen = []

    def query(self, messages):
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeModel responses exhausted")
        return dict(self.responses.pop(0))

    def format_message(self, **kwargs):
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        msgs = []
        actions = message.get("extra", {}).get("actions", [])
        for i, action in enumerate(actions):
            msgs.append({
                "content": f"output[{i}]",
                "role": "tool",
                "tool_call_id": action.get("tool_call_id", f"t{i}"),
                "extra": {"raw_output": "ok", "returncode": 0},
            })
        return msgs

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {}}


class FakeEnv:
    def __init__(self):
        self.executed = []

    def execute(self, action):
        self.executed.append(action)
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {}}


def make_agent(model, env, **cfg):
    cfg.setdefault("system_template", "SYS {{task}}")
    cfg.setdefault("instance_template", "TASK {{task}}")
    return SubmitAgent(model, env, **cfg)


def test_execute_actions_submit_ends_task():
    agent = make_agent(FakeModel([]), FakeEnv(), step_limit=10)
    msg = make_assistant([submit_action([{"file": "a.py", "class_name": "A",
                                          "function_name": "f"}])])
    with pytest.raises(Submitted) as exc:
        agent.execute_actions(msg)
    exit_msg = exc.value.messages[0]
    assert exit_msg["extra"]["exit_status"] == "Submitted"
    assert json.loads(exit_msg["extra"]["submission"]) == [
        {"file": "a.py", "class_name": "A", "function_name": "f"}
    ]
    assert agent.submitted is True


def test_execute_actions_bash_alone_passes_through():
    env = FakeEnv()
    agent = make_agent(FakeModel([]), env, step_limit=10)
    msgs = agent.execute_actions(make_assistant([bash_action("echo hi")]))
    assert len(env.executed) == 1
    assert msgs[0]["role"] == "tool"
    assert agent.submitted is False


def test_execute_actions_mixed_raises_format_error():
    agent = make_agent(FakeModel([]), FakeEnv(), step_limit=10)
    msg = make_assistant([bash_action("echo hi"),
                          submit_action([{"file": "a.py"}])])
    with pytest.raises(FormatError):
        agent.execute_actions(msg)
    assert agent.submitted is False


def test_run_submit_flow():
    """bash 探索 → submit_locations → Submitted（submission=JSON），无魔法串。"""
    env = FakeEnv()
    model = FakeModel([
        make_assistant([bash_action("rg foo -t py")]),
        make_assistant([submit_action([{"file": "src/a.py", "class_name": "A",
                                        "function_name": "f"}])]),
    ])
    agent = make_agent(model, env, step_limit=20)
    result = agent.run(task="fix the bug")
    assert result["exit_status"] == "Submitted"
    assert json.loads(result["submission"]) == [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"}
    ]
    assert len(env.executed) == 1
    # 轨迹末尾是 exit 消息（新的任务结束标志）
    assert agent.messages[-1]["role"] == "exit"
    assert agent.messages[-1]["extra"]["exit_status"] == "Submitted"


def test_reminder_on_step_limit_then_submit():
    """到达轮次上限 → 注入催收 user 消息 → 放行一轮 → 模型提交。"""
    env = FakeEnv()
    model = FakeModel([
        make_assistant([bash_action("ls")]),
        make_assistant([bash_action("ls")]),
        make_assistant([submit_action([{"file": "x.py"}])]),
    ])
    agent = make_agent(model, env, step_limit=2)
    result = agent.run(task="t")
    assert result["exit_status"] == "Submitted"
    assert json.loads(result["submission"]) == [{"file": "x.py", "class_name": None,
                                                 "function_name": None}]
    reminders = [m for m in agent.messages if m.get("role") == "user"
                 and m.get("content") == SUBMIT_REMINDER]
    assert len(reminders) == 1
    assert model.calls == 3  # 2 次正常 + 1 次催收后放行
    # 催收消息确实在最后一次模型调用前可见
    assert any(SUBMIT_REMINDER in str(m.get("content", ""))
               for m in model.seen[-1])


def test_reminder_once_then_limits_exceeded():
    """催收后仍不提交 → 第二次触顶真正结束（LimitsExceeded），催收只发一次。"""
    env = FakeEnv()
    model = FakeModel([
        make_assistant([bash_action("ls")]),
        make_assistant([bash_action("ls")]),
    ])
    agent = make_agent(model, env, step_limit=1)
    result = agent.run(task="t")
    assert result["exit_status"] == "LimitsExceeded"
    reminders = [m for m in agent.messages if m.get("role") == "user"
                 and m.get("content") == SUBMIT_REMINDER]
    assert len(reminders) == 1
    assert model.calls == 2  # 1 次正常 + 1 次催收后放行，之后触顶
    assert agent.messages[-1]["role"] == "exit"
    assert agent.messages[-1]["extra"]["exit_status"] == "LimitsExceeded"


def test_reminder_after_repeated_format_error():
    """提前结束（RepeatedFormatError）且未提交 → 催收 → 放行 → 模型提交。"""
    env = FakeEnv()

    class ErrModel(FakeModel):
        def __init__(self, responses):
            super().__init__(responses)
            self.fail_first = True

        def query(self, messages):
            self.calls += 1
            self.seen.append([dict(m) for m in messages])
            if self.fail_first:
                self.fail_first = False
                raise FormatError({"role": "user", "content": "bad format",
                                   "extra": {"interrupt_type": "FormatError"}})
            return self.responses.pop(0)

    model = ErrModel([make_assistant([submit_action([{"file": "y.py"}])])])
    agent = make_agent(model, env, step_limit=10, max_consecutive_format_errors=1)
    result = agent.run(task="t")
    assert result["exit_status"] == "Submitted"
    reminders = [m for m in agent.messages if m.get("role") == "user"
                 and m.get("content") == SUBMIT_REMINDER]
    assert len(reminders) == 1
    # 格式错误反馈之后、提交之前，催收消息出现过
    reminder_idx = [i for i, m in enumerate(agent.messages)
                    if m.get("content") == SUBMIT_REMINDER][0]
    assert agent.messages[reminder_idx]["role"] == "user"


def test_submitted_exit_breaks_without_reminder():
    """已提交（Submitted）不触发催收。"""
    env = FakeEnv()
    model = FakeModel([
        make_assistant([submit_action([{"file": "z.py"}])]),
    ])
    agent = make_agent(model, env, step_limit=2)
    result = agent.run(task="t")
    assert result["exit_status"] == "Submitted"
    reminders = [m for m in agent.messages if m.get("role") == "user"
                 and m.get("content") == SUBMIT_REMINDER]
    assert reminders == []


def test_resolve_agent_class():
    cls = resolve_agent_class("agent.submit_agent.SubmitAgent")
    assert cls is SubmitAgent
    assert resolve_agent_class(SubmitAgent) is SubmitAgent
    with pytest.raises(ValueError):
        resolve_agent_class("no-such-path")
