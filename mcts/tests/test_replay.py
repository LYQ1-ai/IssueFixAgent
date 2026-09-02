# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.replay.ReplayRunner`` 兜底测试：probe 前缀缺 user 消息时拒绝发请求。

修复 2026-08-31：root 头部缺失导致 prefix_messages 无 user → sglang 400
"No user query found in messages"（litellm 16s/60s 退避重试刷屏）。兜底在
``run_probe`` 入口（构造 env / 调 LLM 之前）直接拒绝。
"""

import pytest

from mcts.replay import ReplayRunner
from mcts.tests.helpers import make_instance


def test_run_probe_rejects_prefix_without_user():
    runner = ReplayRunner(model_name="openai/fake")
    inst = make_instance(instance_id="instP")
    with pytest.raises(ValueError, match="missing a user message"):
        runner.run_probe(
            inst, "fake-container",
            prefix_messages=[{"role": "assistant", "content": "a",
                              "extra": {"actions": []}}],
            prefix_steps=[],
        )


def test_run_probe_empty_prefix_rejected():
    runner = ReplayRunner(model_name="openai/fake")
    inst = make_instance(instance_id="instQ")
    with pytest.raises(ValueError):
        runner.run_probe(inst, "fake-container",
                         prefix_messages=[], prefix_steps=[])
