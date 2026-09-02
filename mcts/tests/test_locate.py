# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.locate`` 二分定位测试（纯逻辑，脚本化 perform_rollouts）。"""

import asyncio
from types import SimpleNamespace

from mcts.locate import annotation_entry, locate_error, split_middle
from mcts.node import MCTSNode
from mcts.steps import prefix_node_key

from mcts.tests.helpers import make_steps


class TestSplitMiddle:
    def test_even(self):
        assert split_middle([1, 2, 3, 4]) == ([1, 2], [3, 4])

    def test_odd(self):
        assert split_middle([1, 2, 3]) == ([1], [2, 3])

    def test_short(self):
        # ReARTeR 同款：mid = len//2 = 0 → 左空右全（locate 循环只在 len>=2 时切分）
        assert split_middle([1]) == ([], [1])


class TestLocateError:
    async def _run(self, rollout_steps, script: dict):
        """script: probe 前缀步数 -> mc。"""
        node = MCTSNode(instance_id="i1", node_key="root")
        registry: dict[str, MCTSNode] = {}

        def get_node(prefix):
            key = prefix_node_key(prefix)
            if key not in registry:
                registry[key] = MCTSNode(instance_id="i1", node_key=key,
                                         prefix_steps=list(prefix))
            return registry[key]

        async def perform_rollouts(n):
            from mcts.tasks import RolloutResult

            mc = script.get(len(n.prefix_steps), 0.0)
            n.set_rollouts([RolloutResult(instance_id="i1", node_key=n.node_key,
                                          rollout_idx=i, correct=(mc > 0.5))
                            for i in range(5)])
            n.mc_score = mc

        rollout = SimpleNamespace(steps=make_steps(rollout_steps))
        return await locate_error(node, rollout, get_node=get_node,
                                  perform_rollouts=perform_rollouts)

    def test_error_in_right_half(self):
        # [s1,s2]→0.8（错误在右半），[s1,s2,s3]→0.2（错误在右半）→ 定位到 s4
        expanded, leaves = asyncio.run(self._run(
            ["s1", "s2", "s3", "s4"],
            {2: 0.8, 3: 0.2},
        ))
        assert [len(e.prefix_steps) for e in expanded] == [2, 3]
        assert leaves == []

    def test_error_in_left_half_yields_leaf(self):
        # [s1,s2]→0.0（错误在左半）→ leaf；随后 [s1]→0.0 → 再记 leaf（ReARTeR 同款收缩）
        expanded, leaves = asyncio.run(self._run(
            ["s1", "s2", "s3", "s4"], {2: 0.0, 1: 0.0}))
        assert expanded == []
        assert len(leaves) == 2
        assert leaves[0].mc_score == 0.0
        assert [len(l.prefix_steps) for l in leaves] == [2, 1]

    def test_prefix_stable_stops(self):
        # [s1,s2]→1.0：此前缀已全对 → 停止，不继续二分
        expanded, leaves = asyncio.run(self._run(["s1", "s2", "s3", "s4"], {2: 1.0}))
        assert expanded == []
        assert leaves == []

    def test_nested_prefix_accumulates(self):
        # 前缀叠加：node.prefix=[p1,p2]，rollout=[s1,s2,s3]
        # probe1=[p1,p2,s1] → 0.6；probe2=[p1,p2,s1,s2] → 0.0 → leaf
        node = MCTSNode(instance_id="i1", node_key="base",
                        prefix_steps=make_steps(["p1", "p2"]))
        rollout = SimpleNamespace(steps=make_steps(["s1", "s2", "s3"]))
        registry: dict[str, MCTSNode] = {}

        def get_node(prefix):
            key = prefix_node_key(prefix)
            if key not in registry:
                registry[key] = MCTSNode("i1", key, list(prefix))
            return registry[key]

        async def perform_rollouts(n):
            from mcts.tasks import RolloutResult

            mc = {3: 0.6, 4: 0.0}.get(len(n.prefix_steps), 0.0)
            n.set_rollouts([RolloutResult("i1", n.node_key, i, correct=(mc > 0.5))
                            for i in range(5)])
            n.mc_score = mc

        expanded, leaves = asyncio.run(locate_error(
            node, rollout, get_node=get_node, perform_rollouts=perform_rollouts))
        assert [len(e.prefix_steps) for e in expanded] == [3]
        assert len(leaves) == 1 and len(leaves[0].prefix_steps) == 4


class TestAnnotationEntry:
    def test_leaf_uses_own_prefix(self):
        node = MCTSNode(instance_id="i1", node_key="leaf-key",
                        prefix_steps=make_steps(["a", "b"]))
        node.mc_score = 0.0
        entry = annotation_entry(node, "leaf")
        assert entry == {"instance_id": "i1", "node_key": "leaf-key",
                         "mc_score": 0.0, "n_steps": 2, "type": "leaf"}
