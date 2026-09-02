# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.node`` MC / QU 閫夋嫨娴嬭瘯 鈥斺€?鏁板€奸攣瀹?ReARTeR docs/01 搂11.1鈥?1.6銆?
QU = 伪^(1鈭扢C)路尾^(len/max_len) + c路鈭?危visits)/(1+visits)锛岄粯璁?伪=0.5, 尾=0.9,
max_len=6, c=0.125銆傜ず渚嬫暟鍊硷紙docs/01 搂11.2 / 搂11.3锛夛細
root锛圡C=0.4, 杞ㄨ抗闀?[5,6,4,4,5]锛夐杞€変腑 t3锛坬u鈮?.6151锛夛紱
鍔犲叆 n1锛圡C=0.8, 闀?[3,4,2,3,4]锛変笌 n2锛圡C=0.2, 闀垮叏 1锛夊悗閫変腑 n1.r3锛坬u鈮?.9656锛夈€?"""

import pytest

from mcts.node import (
    MCTSNode,
    compute_q_value,
    compute_u_value,
    select_best_node,
)
from mcts.tasks import RolloutResult

from mcts.tests.helpers import make_steps

N = 5


def _rollout(idx: int, n_steps: int, correct: bool) -> RolloutResult:
    return RolloutResult(
        instance_id="i1", node_key="k", rollout_idx=idx,
        correct=correct, reward=1.0 if correct else 0.0,
        steps=make_steps([f"s{idx}#{j}" for j in range(n_steps)]),
    )


def _node(instance_id: str, node_key: str, mc: float,
          lens: list[int], flags: list[int], visits: int = 0) -> MCTSNode:
    node = MCTSNode(instance_id=instance_id, node_key=node_key)
    for i, (ln, ok) in enumerate(zip(lens, flags)):
        node.add_rollout(_rollout(i, ln, bool(ok)))
    node.mc_score = mc
    node.visits = visits
    return node


class TestComputeQ:
    def test_rearter_q_value(self):
        # t3锛歭en=4, MC=0.4 鈫?0.5^0.6 脳 0.9^(4/6) 鈮?0.6151
        rollout = _rollout(0, 4, True)
        assert compute_q_value(rollout, 0.4) == pytest.approx(0.6150, abs=2e-3)

    def test_shorter_path_preferred(self):
        r_short, r_long = _rollout(0, 2, True), _rollout(1, 6, True)
        assert compute_q_value(r_short, 0.5) > compute_q_value(r_long, 0.5)


class TestComputeU:
    def test_less_visited_preferred(self):
        a = MCTSNode("i", "a"); a.visits = 0
        b = MCTSNode("i", "b"); b.visits = 3
        assert compute_u_value(a, [a, b]) > compute_u_value(b, [a, b])


class TestSelectBestNode:
    def test_first_round_selects_shortest_rollout(self):
        root = _node("i1", "root", mc=0.4, lens=[5, 6, 4, 4, 5],
                     flags=[1, 0, 1, 0, 0])
        node, idx, qu = select_best_node([root])
        assert node is root
        assert idx == 2                       # t3锛坙en=4 鏈€鐭級
        assert qu == pytest.approx(0.6150, abs=2e-3)
        assert root.visited_flags[2] is True  # 閫変腑鍚庢爣璁板凡璁块棶

    def test_second_round_selects_high_mc_node(self):
        root = _node("i1", "root", mc=0.4, lens=[5, 6, 4, 4, 5],
                     flags=[1, 0, 1, 0, 0], visits=1)
        root.visited_flags[2] = True          # t3 宸插湪绗?1 杞澶勭悊
        n1 = _node("i1", "n1", mc=0.8, lens=[3, 4, 2, 3, 4],
                   flags=[1, 1, 1, 1, 0])
        n2 = _node("i1", "n2", mc=0.2, lens=[1, 1, 1, 1, 1],
                   flags=[1, 0, 0, 0, 0])
        node, idx, qu = select_best_node([root, n1, n2])
        assert node is n1
        assert idx == 2                       # n1.r3锛坙en=2锛?        assert qu == pytest.approx(0.9656, abs=2e-3)
        assert n1.visited_flags[2] is True

    def test_mc_gate_skips_extremes(self):
        # 鍏ㄥ / 鍏ㄩ敊鐨勮妭鐐逛笉鍙€夋嫨
        ok = _node("i1", "ok", mc=1.0, lens=[1] * N, flags=[1] * N)
        bad = _node("i1", "bad", mc=0.0, lens=[1] * N, flags=[0] * N)
        assert select_best_node([ok, bad]) == (None, None, None)

    def test_visited_rollouts_skipped(self):
        n = _node("i1", "n", mc=0.5, lens=[1] * N, flags=[1, 0, 1, 0, 1])
        n.visited_flags = [True] * N
        assert select_best_node([n]) == (None, None, None)

    def test_failed_rollouts_not_counted(self):
        n = _node("i1", "n", mc=0.5, lens=[1] * N, flags=[1, 1, 1, 1, 0])
        n.rollouts[4].error = "boom"          # 澶辫触 rollout 涓嶅弬涓庨€夋嫨
        n.visited_flags[4] = False
        node, idx, qu = select_best_node([n])
        assert node is n and idx in (0, 1, 2, 3)


class TestMCTSNode:
    def test_mc_over_valid_rollouts(self):
        node = _node("i1", "n", mc=None, lens=[1] * 5, flags=[1, 0, 1, 0, 1])
        assert node.compute_mc() == 0.6

    def test_failed_rollout_excluded_from_mc(self):
        node = _node("i1", "n", mc=None, lens=[1] * 5, flags=[1, 0, 1, 0, 1])
        node.rollouts[1].error = "docker fail"
        # 鏈夋晥 4 鏉★細[1,1,0,1] 鈫?0.75
        assert node.n_rollouts == 4
        assert node.compute_mc() == 0.75

    def test_gate(self):
        node = _node("i1", "n", mc=None, lens=[1] * 5, flags=[1, 0, 1, 0, 0])
        assert node.gated is True
        assert node.mc_score == 0.4

