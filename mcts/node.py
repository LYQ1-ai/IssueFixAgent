# SPDX-License-Identifier: BSD-3-Clause

"""MCTS 树节点与选择逻辑（PLAN §2.4，移植 ReARTeR ``PRM_Data/module.py``）。

- :class:`MCTSNode`：``(instance_id, prefix_steps)`` 节点；``rollouts`` 为该节点
  已完成的 rollout 结果（``mcts.tasks.RolloutResult``），``visited_flags`` 标记
  每条 rollout 是否已被"选择-定位"过（每条只修正一次，对齐 ReARTeR）；
- :func:`compute_q_value` / :func:`compute_u_value` / :func:`select_best_node`：
  UCB 风格 QU 选择 —— ``QU = α^(1−MC)·β^(len/max_len) + c·√(Σvisits)/(1+visits)``，
  在"全部 0<MC<1 节点 × 未访问 rollout"上取全局最大（树的路径关系由前缀内容
  隐式承载，树是扁平数组，见 docs/01 §11.9-5）。

本模块为**纯逻辑**（不 import agent / minisweagent），MC / QU 数值用 ReARTeR
docs/01 11.1–11.6 的数值示例在单测中锁定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """树节点 = (instance, 轨迹前缀步序列)。``prefix_steps`` 为空 = 根节点。"""

    instance_id: str
    node_key: str
    prefix_steps: list = field(default_factory=list)  # list[Step]
    mc_score: Optional[float] = None
    visits: int = 0
    rollouts: list = field(default_factory=list)      # list[RolloutResult]（含失败占位）
    visited_flags: list = field(default_factory=list)  # 与 rollouts 平行

    # ------------------------------------------------------------------
    # rollout 记账
    # ------------------------------------------------------------------

    def add_rollout(self, rollout: object) -> None:
        self.rollouts.append(rollout)
        self.visited_flags.append(False)

    def set_rollouts(self, rollouts: list) -> None:
        """整体替换（断点续跑加载磁盘结果时用）；按长度补齐 visited 标记。"""
        self.rollouts = list(rollouts)
        self.visited_flags = [False] * len(rollouts)

    def increment_visits(self) -> None:
        self.visits += 1

    # ------------------------------------------------------------------
    # MC 分数（论文 Eq.3：MC = Σcorrect / N）
    # ------------------------------------------------------------------

    @property
    def valid_rollouts(self) -> list:
        """剔除失败（error 非空）的 rollout —— 对齐 ReARTeR 的 bad_gen 跳过语义。"""
        return [r for r in self.rollouts if not getattr(r, "error", None)]

    @property
    def n_rollouts(self) -> int:
        return len(self.valid_rollouts)

    def correct_count(self) -> int:
        return sum(1 for r in self.valid_rollouts if getattr(r, "correct", False))

    def compute_mc(self) -> float:
        """MC = 正确 rollout 数 / 有效 rollout 数；无有效 rollout → 0.0。"""
        n = self.n_rollouts
        return self.correct_count() / n if n else 0.0

    @property
    def gated(self) -> bool:
        """门控：``0 < MC < 1`` 才进入树搜索（全对 / 全错不再定位）。"""
        if self.mc_score is None:
            self.mc_score = self.compute_mc()
        return 0.0 < self.mc_score < 1.0


def compute_q_value(
    rollout: object,
    mc_score: float,
    alpha: float = 0.5,
    beta: float = 0.9,
    max_length: int = 6,
) -> float:
    """利用项（ReARTeR ``compute_q_value``）：低 MC、短路径（错误更靠前）优先。"""
    steps = getattr(rollout, "steps", None)
    length = len(steps) if steps is not None else 0
    return (alpha ** (1.0 - mc_score)) * (beta ** (length / max(max_length, 1)))


def compute_u_value(
    node: MCTSNode,
    all_nodes: list[MCTSNode],
    exploration_param: float = 0.125,
) -> float:
    """探索项：总访问量越大、该节点访问越少，越优先。"""
    total_visits = sum(n.visits for n in all_nodes)
    return exploration_param * (math.sqrt(total_visits) / (1 + node.visits))


def select_best_node(
    nodes: list[MCTSNode],
    alpha: float = 0.5,
    beta: float = 0.9,
    max_length: int = 6,
    exploration_param: float = 0.125,
) -> tuple[Optional[MCTSNode], Optional[int], Optional[float]]:
    """全局 QU 最大选择（对齐 ReARTeR ``select_best_node``）。

    - 只考虑 ``0 < mc_score < 1`` 的节点（全对/全错无修正意义）；
    - 跳过 ``visited_flags[idx] == True`` 的 rollout（每条只修正一次）；
    - 选中后把该 rollout 标记为已访问（``visited_flags[idx] = True``）。

    Returns:
        ``(node, rollout_idx, qu)``；无候选返回 ``(None, None, None)``。
    """
    best_node: Optional[MCTSNode] = None
    best_idx: Optional[int] = None
    best_qu: Optional[float] = None
    for node in nodes:
        mc = node.mc_score if node.mc_score is not None else node.compute_mc()
        node.mc_score = mc
        if not (0.0 < mc < 1.0):
            continue
        u = compute_u_value(node, nodes, exploration_param)
        for idx, rollout in enumerate(node.rollouts):
            if node.visited_flags[idx]:
                continue
            if getattr(rollout, "error", None):
                continue  # 失败 rollout 不参与选择（不计入 N）
            qu = compute_q_value(rollout, mc, alpha, beta, max_length) + u
            if best_qu is None or qu > best_qu:
                best_qu = qu
                best_node = node
                best_idx = idx
    if best_node is not None and best_idx is not None:
        best_node.visited_flags[best_idx] = True
        return best_node, best_idx, best_qu
    return None, None, None


__all__ = [
    "MCTSNode",
    "compute_q_value",
    "compute_u_value",
    "select_best_node",
]
