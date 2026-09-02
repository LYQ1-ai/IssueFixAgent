# SPDX-License-Identifier: BSD-3-Clause

"""``submit_locations`` 结构化判定测试（docs/reward_design.md 综合方案）。

覆盖两种模式：
- ``layered``（主路径）：层级路径一对一匹配 + Soft-F1 + Lᵢ 缺失折算 +
  τ=0.6 二值化 + strict_multi_gate；
- ``independent``（对照）：旧三层独立 F1 相加。
"""

import json

import pytest

from mcts.instances import Gold
from mcts.reward import (
    DEFAULT_DEPTH_WEIGHTS,
    locations_from_submission,
    locations_localization_f1,
    locations_localization_f1_layered,
    match_depth,
    parse_structured_locations,
    reward_from_locations,
    reward_from_locations_layered,
    reward_from_trajectory_exit,
    triples_from_gold,
)

GOLD = Gold(
    files=frozenset({"src/a.py", "src/b.py"}),
    modules=frozenset({"src/a.py:A", "src/b.py:g"}),
    entities=frozenset({"src/a.py:A.f", "src/b.py:g"}),
)

# 单位置 gold：类内方法（L=3）
G1 = Gold(files=frozenset({"src/a.py"}),
          modules=frozenset({"src/a.py:A"}),
          entities=frozenset({"src/a.py:A.f"}))


# ---------------------------------------------------------------------------
# 基础构件
# ---------------------------------------------------------------------------

def test_parse_structured_locations_codescout_align():
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "class_name": None, "function_name": "g"},
        {"file": "src/c.py", "class_name": None, "function_name": None},
        {"file": "src/d.py", "class_name": "D", "function_name": None},
    ]
    files, modules, entities = parse_structured_locations(locs)
    assert files == {"src/a.py", "src/b.py", "src/c.py", "src/d.py"}
    assert modules == {"src/a.py:A", "src/b.py:g", "src/d.py:D"}
    assert entities == {"src/a.py:A.f", "src/b.py:g"}


def test_triples_from_gold():
    assert sorted(triples_from_gold(GOLD)) == sorted(
        [("src/a.py", "A", "f"), ("src/b.py", None, "g")])


def test_match_depth_unit_table():
    g = ("a.py", "A", "f")
    assert match_depth(("a.py", "A", "f"), g) == 3
    assert match_depth(("a.py", "A", None), g) == 2
    assert match_depth(("a.py", "A", "wrong"), g) == 2
    assert match_depth(("a.py", None, "f"), g) == 1
    assert match_depth(("a.py", None, None), g) == 1
    assert match_depth(("b.py", "A", "f"), g) == 0
    # gold 无 class（独立函数）
    g2 = ("b.py", None, "g")
    assert match_depth(("b.py", None, "g"), g2) == 3
    assert match_depth(("b.py", "X", "g"), g2) == 3      # gold 无 class，func 对即全中
    assert match_depth(("b.py", None, None), g2) == 1
    # gold 最细为 class（无 func）
    g3 = ("c.py", "C", None)
    assert match_depth(("c.py", "C", "m"), g3) == 2
    assert match_depth(("c.py", "C", None), g3) == 2
    assert match_depth(("c.py", None, None), g3) == 1


# ---------------------------------------------------------------------------
# layered：单位置全深度表（docs/reward_design.md §5 示例 A）
# ---------------------------------------------------------------------------

def test_layered_single_location_depth_table():
    cases = [
        ([{"file": "src/a.py", "class_name": "A", "function_name": "f"}], 1.0, True),
        ([{"file": "src/a.py", "class_name": "A", "function_name": None}], 0.5, False),
        ([{"file": "src/a.py", "class_name": None, "function_name": None}], 0.2, False),
        ([{"file": "src/a.py", "class_name": "A", "function_name": "wrong"}], 0.5, False),
        ([{"file": "src/b.py", "class_name": "A", "function_name": "f"}], 0.0, False),
    ]
    for locs, exp_reward, exp_correct in cases:
        reward, correct, details = reward_from_locations_layered(locs, G1)
        assert reward == pytest.approx(exp_reward), locs
        assert correct is exp_correct, locs
        assert details["M"] == 1 and details["N"] == 1


def test_layered_overprediction_dilutes():
    """1 全中 + 1 错位置 → precision 稀释（M=2, N=1）。"""
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/a.py", "class_name": "A", "function_name": "other"},
    ]
    reward, correct, details = reward_from_locations_layered(locs, G1)
    assert reward == pytest.approx(2 * 1.0 / 3)   # C=1.0, M=2, N=1 → 2/3
    assert correct is True


def test_layered_duplicate_not_double_counted():
    """重复/冗余预测不重复计分（一对一匹配只计最深一次）。"""
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/a.py", "class_name": "A", "function_name": None},  # 冗余（匹配会被深者占用）
    ]
    reward, details = locations_localization_f1_layered(locs, G1)
    assert details["C"] == pytest.approx(1.0)     # 只计 1.0，无重复
    assert details["M"] == 2
    assert reward == pytest.approx(2 * 1.0 / 3)   # 2/3


# ---------------------------------------------------------------------------
# layered：多位置 / 缺失折算 / 防刷（docs/reward_design.md §5 示例 B/C）
# ---------------------------------------------------------------------------

def test_layered_multi_location():
    """两个全中 → 1.0；漏一半 → 2/3；全中+file 级 → 0.6（压线）。"""
    full = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "function_name": "g"},
    ]
    reward, correct, d = reward_from_locations_layered(full, GOLD)
    assert reward == pytest.approx(1.0) and correct is True
    assert d["exact_hits"] == 2

    half = [{"file": "src/a.py", "class_name": "A", "function_name": "f"}]
    reward, correct, d = reward_from_locations_layered(half, GOLD)
    assert reward == pytest.approx(2.0 / 3.0) and correct is True
    # strict_multi_gate：N=2 要求 exact_hits ≥ 1 → 1 ≥ 1 ✓

    mix = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py"},
    ]
    reward, correct, d = reward_from_locations_layered(mix, GOLD)
    assert reward == pytest.approx(2 * 1.2 / 4)   # C=1.0+0.2, M=N=2 → 0.6
    assert correct is True                          # 恰 ≥ τ


def test_layered_strict_multi_gate_blocks_file_padding():
    """N≥2 但 func 命中不足半数 → strict_multi_gate 判错。"""
    # gold 两个位置，只报对 1 个 func（1/2 命中），另一个只报 file
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py"},
    ]
    # N=2, exact_hits=1 ≥ ⌈2/2⌉=1 → gate 通过（上面测试已覆盖）
    # 构造 exact_hits=0 但 reward 可能过线的场景：gold N=2，两处都只报 file
    locs2 = [{"file": "src/a.py"}, {"file": "src/b.py"}]
    reward, correct, d = reward_from_locations_layered(locs2, GOLD)
    # C = 0.2 + 0.2 = 0.4, M=N=2 → reward = 2*0.4/4 = 0.2 < τ → 已判错
    assert correct is False
    # 显式验证 gate 本身：reward 过线但 func 命中不足
    reward2, correct2, d2 = reward_from_locations_layered(
        [{"file": "src/a.py", "class_name": "A", "function_name": "f"},
         {"file": "src/b.py", "function_name": "g"},
         {"file": "src/a.py", "class_name": "A", "function_name": "f"}],  # 冗余提高 M
        GOLD, strict_multi_gate=True)
    # M=3, C=2.0 → reward=2*2/5=0.8 ≥ τ，但 N=2 exact_hits=2 ≥ 1 → 仍对
    assert correct2 is True


def test_layered_strict_multi_gate_fail_case():
    """gold 3 个类内方法位置、只有 1 个 func 命中 → gate 拒绝（1 < ⌈3/2⌉=2）。"""
    gold3 = Gold(
        files=frozenset({"p1.py", "p2.py", "p3.py"}),
        modules=frozenset({"p1.py:C1", "p2.py:C2", "p3.py:C3"}),
        entities=frozenset({"p1.py:C1.F1", "p2.py:C2.F2", "p3.py:C3.F3"}),
    )
    # 两个 func 全中 + 一个 file 级：reward 过线，gate 也过
    locs2 = [
        {"file": "p1.py", "class_name": "C1", "function_name": "F1"},
        {"file": "p2.py", "class_name": "C2", "function_name": "F2"},
        {"file": "p3.py"},
    ]
    reward2, correct2, d2 = reward_from_locations_layered(locs2, gold3)
    # C = 1.0 + 1.0 + 0.2 = 2.2, M=N=3 → reward = 2*2.2/6 = 0.733 ≥ τ
    assert reward2 == pytest.approx(2 * 2.2 / 6)
    assert correct2 is True            # exact_hits=2 ≥ 2 ✓
    # 只有 1 个 func 命中、另两个只到 class 级：reward 过线但 gate 拒绝
    locs3 = [
        {"file": "p1.py", "class_name": "C1", "function_name": "F1"},
        {"file": "p2.py", "class_name": "C2"},
        {"file": "p3.py", "class_name": "C3"},
    ]
    reward3, correct3_on, d3 = reward_from_locations_layered(locs3, gold3, strict_multi_gate=True)
    _, correct3_off, _ = reward_from_locations_layered(locs3, gold3, strict_multi_gate=False)
    # C = 1.0 + 0.5 + 0.5 = 2.0 → reward = 2*2/6 = 0.667 ≥ τ
    assert reward3 == pytest.approx(2 * 2.0 / 6)
    assert correct3_off is True
    assert correct3_on is False        # exact_hits=1 < 2 → gate 拒绝


def test_layered_missing_level_norm():
    """gold 无 func（L=2）：预测到 func 或 class 都能拿满分（折算）。"""
    gold_l2 = Gold(files=frozenset({"a.py"}), modules=frozenset({"a.py:A"}),
                   entities=frozenset())
    for locs in [
        [{"file": "a.py", "class_name": "A", "function_name": "m"}],  # 超过 gold 粒度
        [{"file": "a.py", "class_name": "A"}],
    ]:
        reward, correct, d = reward_from_locations_layered(locs, gold_l2)
        assert reward == pytest.approx(1.0), locs
        assert correct is True
        assert d["N"] == 1
    # 只到 file → 部分分
    reward, _, d = reward_from_locations_layered([{"file": "a.py"}], gold_l2)
    assert reward == pytest.approx(0.4)   # credit = 0.2/0.5


def test_layered_penalties():
    """惩罚项（默认 0；开启时扣在 C 上）。"""
    locs = [{"file": "src/a.py", "class_name": "A", "function_name": "f"}]
    r0, _, _ = reward_from_locations_layered(locs, G1)
    assert r0 == pytest.approx(1.0)
    # overpromise：未匹配预测声称了 class+func → 扣 2δ
    locs2 = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/zz.py", "class_name": "Z", "function_name": "zz"},  # 全错但过度承诺
    ]
    r_pen, _, d = reward_from_locations_layered(locs2, G1, overpromise_penalty=0.05)
    # C = 1.0 - 0.05*2 = 0.9, M=2, N=1 → reward = 2*0.9/3 = 0.6
    assert r_pen == pytest.approx(2 * 0.9 / 3)
    assert d["penalty"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# independent（对照）：旧三层独立 F1 相加
# ---------------------------------------------------------------------------

def test_independent_locations_localization_f1():
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "function_name": "g"},
    ]
    reward, details = locations_localization_f1(locs, GOLD)
    assert reward == pytest.approx(3.0)
    assert details["file_f1"] == pytest.approx(1.0)


def test_independent_reward_from_locations_threshold():
    locs = [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "function_name": "g"},
    ]
    reward, correct, _ = reward_from_locations(locs, GOLD, threshold=2.0)
    assert reward == pytest.approx(3.0) and correct is True


# ---------------------------------------------------------------------------
# 统一入口 reward_from_trajectory_exit
# ---------------------------------------------------------------------------

def test_trajectory_exit_layered_valid():
    submission = json.dumps([
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "function_name": "g"},
    ])
    reward, correct, details = reward_from_trajectory_exit("Submitted", submission, GOLD)
    assert reward == pytest.approx(1.0)
    assert correct is True
    assert details["mode"] == "layered"
    assert details["reward"] == pytest.approx(1.0)   # 连续分数保留在 details
    assert details["exact_hits"] == 2


def test_trajectory_exit_independent_valid():
    submission = json.dumps([
        {"file": "src/a.py", "class_name": "A", "function_name": "f"},
        {"file": "src/b.py", "function_name": "g"},
    ])
    reward, correct, details = reward_from_trajectory_exit(
        "Submitted", submission, GOLD, mode="independent", threshold=2.0)
    assert reward == pytest.approx(3.0)
    assert correct is True
    assert details["mode"] == "independent"


def test_trajectory_exit_not_submitted():
    reward, correct, details = reward_from_trajectory_exit("LimitsExceeded", "", GOLD)
    assert reward == 0.0 and correct is False
    assert details["reason"] == "not_submitted"


def test_trajectory_exit_invalid_submission():
    reward, correct, details = reward_from_trajectory_exit("Submitted", "not-json", GOLD)
    assert reward == 0.0 and correct is False
    assert details["reason"] == "invalid_submission"
    reward2, _, _ = reward_from_trajectory_exit("Submitted", "", GOLD)
    assert reward2 == 0.0


def test_locations_from_submission_roundtrip():
    raw = json.dumps([{"file": "x.py", "class_name": None, "function_name": None}])
    assert locations_from_submission(raw) == [
        {"file": "x.py", "class_name": None, "function_name": None}
    ]


def test_default_depth_weights():
    assert DEFAULT_DEPTH_WEIGHTS == {1: 0.2, 2: 0.5, 3: 1.0}
