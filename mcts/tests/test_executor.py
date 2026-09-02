# SPDX-License-Identifier: BSD-3-Clause

"""``AgentRolloutExecutor`` 结果构造测试（fake env / fake replay，不碰 Docker）。

锁定"结构化判定 + 结果字段"路径：rollout 的 reward 完全由
``exit_status/submission`` 决定（无有效提交 → 0）；``n_calls/cost`` 从轨迹
``info.model_stats`` 提取。
"""

import json

import pytest

from mcts.executor import AgentRolloutExecutor
from mcts.tasks import EnvFactory, RolloutResult, RolloutTask
from mcts.tests.helpers import make_instance


class _FakeEnvFactory:
    """只记录 create/destroy 调用的假 EnvFactory。"""

    def __init__(self):
        self.created = []
        self.destroyed = []

    def create_env(self, instance):
        self.created.append(instance.instance_id)
        return "fake-container"

    def destroy_env(self, container):
        self.destroyed.append(container)

    def exec(self, container, cmd):
        return ""


def _fake_replay_submitted(monkeypatch, submission_json, n_calls=7, cost=0.42):
    """把 ReplayRunner 换成返回固定轨迹（Submitted + JSON submission）的假实现。"""
    class _FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_free(self, instance, container):
            return {
                "info": {"model_stats": {"api_calls": n_calls,
                                         "instance_cost": cost}},
                "messages": [
                    {"role": "assistant", "content": "x",
                     "extra": {"actions": [{"tool": "submit_locations",
                                            "locations": [{"file": "x.py"}],
                                            "tool_call_id": "t1"}]}},
                    {"role": "exit", "content": submission_json,
                     "extra": {"exit_status": "Submitted",
                               "submission": submission_json}},
                ],
                "trajectory_format": "mini-swe-agent-1.1",
            }

        def run_probe(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("root task must use run_free")

    monkeypatch.setattr("mcts.executor.ReplayRunner", _FakeRunner)


def _root_task(instance):
    return RolloutTask(
        instance_id=instance.instance_id, node_key="root", rollout_idx=0,
        kind="root", priority=0, payload={"instance": instance,
                                          "prefix_steps": [], "prefix_messages": []},
    )


def test_executor_submitted_valid_locations(monkeypatch):
    inst = make_instance(instance_id="instA",
                         gold_files=frozenset({"x.py"}))
    submission = json.dumps([{"file": "x.py", "class_name": None,
                              "function_name": None}])
    _fake_replay_submitted(monkeypatch, submission, n_calls=7, cost=0.42)
    ex = AgentRolloutExecutor(_FakeEnvFactory(), model_name="openai/fake",
                              reward_threshold=2.0)
    result = ex.run(_root_task(inst))
    assert result.error is None
    assert result.exit_status == "Submitted"
    # layered：gold=(x.py,None,foo)，预测只到 file 级 → d=1 → Soft-F1 = 0.2
    assert result.reward == pytest.approx(0.2)
    assert result.correct is False
    assert result.n_calls == 7
    assert result.cost == 0.42
    assert len(result.steps) == 1       # submit 步本身算一步（assistant 消息带动作）
    # 连续 reward 与分层细节保留在 rollout 结果中（落库不丢分）
    assert result.reward_details is not None
    assert result.reward_details["mode"] == "layered"
    assert result.reward_details["reward"] == pytest.approx(0.2)
    assert result.reward_details["M"] == 1 and result.reward_details["N"] == 1


def test_executor_layered_full_hit_marks_correct(monkeypatch):
    """func 级全中 → reward=1.0，τ=0.6 判对（默认 layered 路径）。"""
    inst = make_instance(instance_id="instE",
                         gold_files=frozenset({"x.py"}))
    submission = json.dumps([{"file": "x.py", "class_name": None,
                              "function_name": "foo"}])
    _fake_replay_submitted(monkeypatch, submission)
    ex = AgentRolloutExecutor(_FakeEnvFactory(), model_name="openai/fake")
    result = ex.run(_root_task(inst))
    assert result.reward == pytest.approx(1.0)
    assert result.correct is True
    assert result.reward_details["depth_hist"][3] == 1


def test_executor_submitted_invalid_submission_zero(monkeypatch):
    inst = make_instance(instance_id="instB")
    _fake_replay_submitted(monkeypatch, "not-json")
    ex = AgentRolloutExecutor(_FakeEnvFactory(), model_name="openai/fake")
    result = ex.run(_root_task(inst))
    assert result.exit_status == "Submitted"
    assert result.reward == 0.0
    assert result.correct is False


def test_executor_not_submitted_zero(monkeypatch):
    inst = make_instance(instance_id="instC")
    ex = AgentRolloutExecutor(_FakeEnvFactory(), model_name="openai/fake")

    class _RunnerLE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_free(self, instance, container):
            return {
                "info": {"model_stats": {"api_calls": 5, "instance_cost": 0.1}},
                "messages": [
                    {"role": "assistant", "content": "x",
                     "extra": {"actions": [{"command": "ls", "tool_call_id": "t1"}]}},
                    {"role": "tool", "content": "out",
                     "extra": {"raw_output": "out", "returncode": 0}},
                    {"role": "exit", "content": "LimitsExceeded",
                     "extra": {"exit_status": "LimitsExceeded", "submission": ""}},
                ],
                "trajectory_format": "mini-swe-agent-1.1",
            }

        def run_probe(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("root task must use run_free")

    monkeypatch.setattr("mcts.executor.ReplayRunner", _RunnerLE)
    result = ex.run(_root_task(inst))
    assert result.exit_status == "LimitsExceeded"
    assert result.reward == 0.0
    assert result.correct is False
    assert result.n_calls == 5


def test_executor_passes_submit_wiring(monkeypatch):
    """executor 构造时把 model_class / agent_class / magic_submit 传给 ReplayRunner。"""
    captured = {}

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.kwargs = kwargs

        def run_free(self, instance, container):
            return {"info": {"model_stats": {"api_calls": 1, "instance_cost": 0.0}},
                    "messages": [
                        {"role": "assistant", "content": "x",
                         "extra": {"actions": [{"command": "ls", "tool_call_id": "t1"}]}},
                        {"role": "exit", "content": "LimitsExceeded",
                         "extra": {"exit_status": "LimitsExceeded", "submission": ""}},
                    ],
                    "trajectory_format": "mini-swe-agent-1.1"}

        def run_probe(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

    monkeypatch.setattr("mcts.executor.ReplayRunner", _FakeRunner)
    ex = AgentRolloutExecutor(
        _FakeEnvFactory(), model_name="openai/fake",
        model_class="agent.submit_model.SubmitLocationsModel",
        agent_class="agent.submit_agent.SubmitAgent",
        magic_submit=False,
    )
    inst = make_instance(instance_id="instD")
    ex.run(_root_task(inst))
    assert captured["agent_class"] == "agent.submit_agent.SubmitAgent"
    assert captured["env_cfg"]["magic_submit"] is False
    assert captured["model_config"].get("model_class") == \
        "agent.submit_model.SubmitLocationsModel"
    assert captured["step_limit"] == 20  # 默认 20 轮
