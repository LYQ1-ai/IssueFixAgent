# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.tasks`` 纯函数测试：probe 前缀头部提取（修复 2026-08-31）。

覆盖：root 第 0 个 rollout 失败（trajectory=None）时跳过、取第一个有效头部；
全部失败 / 空列表返回 []。
"""

import pytest

from mcts.tasks import (
    RolloutResult,
    build_messages_head,
    messages_head_from_rollouts,
    resolve_messages_head,
)

_OK_TRAJ = {
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1", "extra": {"actions": []}},
        {"role": "tool", "content": "o1", "tool_call_id": "t1"},
    ],
    "trajectory_format": "mini-swe-agent-1.1",
}


def _result(trajectory=None, error=None):
    return RolloutResult(instance_id="i", node_key="root", rollout_idx=0,
                         trajectory=trajectory, error=error)


def test_head_takes_first_valid_rollout_skipping_failed():
    """rollouts[0] 失败（trajectory=None）→ 跳过，取第二个成功轨迹的 system+user。"""
    failed = _result(error="Connection error")
    ok = _result(trajectory=_OK_TRAJ)
    head = messages_head_from_rollouts([failed, ok])
    assert [m["role"] for m in head] == ["system", "user"]
    assert head[1]["content"] == "task"


def test_head_skips_empty_messages():
    ok = _result(trajectory={"messages": [], "trajectory_format": "x"})
    assert messages_head_from_rollouts([ok]) == []


def test_head_all_failed_or_empty_returns_empty():
    assert messages_head_from_rollouts([_result(error="boom")]) == []
    assert messages_head_from_rollouts([]) == []
    assert messages_head_from_rollouts(None) == []


def test_head_returns_copies_not_refs():
    ok = _result(trajectory=_OK_TRAJ)
    head = messages_head_from_rollouts([ok])
    head[0]["content"] = "mutated"
    assert _OK_TRAJ["messages"][0]["content"] == "sys"


# ---------------------------------------------------------------------------
# resolve_messages_head：resume 续跑时 rollouts 从 DB 加载（trajectory=None），
# 回退到落库的 instances.messages_head_json
# ---------------------------------------------------------------------------

_STORED_HEAD = [
    {"role": "system", "content": "sys-db"},
    {"role": "user", "content": "task-db"},
]


def test_resolve_falls_back_to_stored_head_when_rollouts_have_no_trajectory():
    """resume 场景：rollouts 全 trajectory=None（DB 加载）→ 用 stored_head。"""
    db_loaded = [_result(error=None), _result(error=None)]  # trajectory=None
    head = resolve_messages_head(db_loaded, stored_head=_STORED_HEAD)
    assert [m["role"] for m in head] == ["system", "user"]
    assert head[1]["content"] == "task-db"


def test_resolve_prefers_rollout_trajectory_over_stored():
    ok = _result(trajectory=_OK_TRAJ)
    head = resolve_messages_head([ok], stored_head=_STORED_HEAD)
    assert head[1]["content"] == "task"   # 轨迹优先


def test_resolve_empty_when_both_unavailable():
    assert resolve_messages_head([], stored_head=None) == []
    assert resolve_messages_head([_result(error="boom")], stored_head=None) == []
    assert resolve_messages_head([_result(error="boom")], stored_head=[]) == []


# ---------------------------------------------------------------------------
# build_messages_head：从 config 提示词 + 实例原始输入直接还原（不依赖轨迹）
# ---------------------------------------------------------------------------

from mcts.tests.helpers import make_instance  # noqa: E402


def test_build_messages_head_from_instance_and_config():
    inst = make_instance(instance_id="instZ",
                         problem_statement="Fix the bug in x.py: foo() fails")
    head = build_messages_head(inst)
    assert [m["role"] for m in head] == ["system", "user"]
    sys_content = head[0]["content"]
    user_content = head[1]["content"]
    # system = 定位提示词（含 submit_locations 规则）；user = 任务描述
    assert "code localization agent" in sys_content
    assert "submit_locations" in sys_content
    assert "Do NOT modify any code" in sys_content
    assert "Fix the bug in x.py: foo() fails" in user_content


def test_build_messages_head_always_available():
    """任何实例（含历史 head 缺失）都能还原头部——与轨迹/落库无关。"""
    inst = make_instance(instance_id="instHist")
    head = build_messages_head(inst)
    assert len(head) == 2
    assert head[1]["content"]  # user 非空
