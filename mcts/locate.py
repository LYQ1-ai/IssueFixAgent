# SPDX-License-Identifier: BSD-3-Clause

"""locate_error 二分定位 + 标注条目构建（PLAN §2.4，移植 ReARTeR ``module.py``）。

- :func:`split_middle`：对半切分步序列（``mid = len // 2``，ReARTeR 同款）；
- :func:`locate_error`：对选中 rollout 的步序列做**二分**，反复试探
  "已确认正确前缀 + 左半段"节点的 MC，确定首个错误步位置：

  | 试探 MC | 判定 |
  | --- | --- |
  | ``== 1`` | 此前缀已全对 → 停止（错误不在左半） |
  | ``0 < MC < 1`` | 错误在右半 → 左半并入正确前缀，继续二分 |
  | ``== 0`` | 加入左半即全错 → 错误在左半，收缩并记 **leaf** |

  树结构与 rollout 执行完全解耦：``get_node(prefix)`` 提供内容寻址节点、
  ``perform_rollouts(node)`` 提供该节点 N 次 rollout（生产 = 任务队列并发提交，
  测试 = 脚本化 Fake）。本模块为纯逻辑，可离线单测。
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from mcts.node import MCTSNode
from mcts.steps import Step

# get_node(prefix_steps) -> MCTSNode
GetNode = Callable[[list[Step]], MCTSNode]
# perform_rollouts(node) -> None（async；生产：任务队列并发 N 次 rollout）
PerformRollouts = Callable[[MCTSNode], Awaitable[None]]


def split_middle(steps: list) -> tuple[list, list]:
    """对半切分（左半在前；``mid = len // 2``，与 ReARTeR 一致）。"""
    mid = len(steps) // 2
    return steps[:mid], steps[mid:]


async def locate_error(
    node: MCTSNode,
    rollout: object,
    *,
    get_node: GetNode,
    perform_rollouts: PerformRollouts,
) -> tuple[list[MCTSNode], list[MCTSNode]]:
    """二分定位首个错误步，返回 ``(expanded, leaves)``。

    ``rollout.steps`` = 该节点 rollout 的**续跑步序列**（相对节点前缀）；
    试探节点前缀 = ``node.prefix_steps + confirmed + left``。
    """
    current_span = list(getattr(rollout, "steps", []) or [])
    confirmed: list[Step] = []
    expanded: list[MCTSNode] = []
    leaves: list[MCTSNode] = []
    while len(current_span) >= 2:
        left, right = split_middle(current_span)
        probe = get_node(node.prefix_steps + confirmed + left)
        await perform_rollouts(probe)
        if probe.mc_score is None:   # 已由 perform_rollouts 设定时不再重算（脚本化测试可精确控制）
            probe.mc_score = probe.compute_mc()
        if probe.mc_score >= 1.0:      # 左半已稳 → 错误不在左半，无需再探
            break
        elif probe.mc_score > 0.0:     # 部分正确 → 错误在右半，左半并入正确前缀
            confirmed += left
            current_span = right
            expanded.append(probe)
        else:                          # 加入左半即全错 → 错误在左半，记 leaf
            current_span = left
            leaves.append(probe)
    return expanded, leaves


def annotation_entry(
    node: MCTSNode, entry_type: str, instance_id: Optional[str] = None
) -> dict:
    """构建 best / leaf / add 标注条目（PLAN §2.4，对齐 ReARTeR 输出格式）。

    与 ReARTeR 的差异（文档记录为修正）：leaf 条目记录**叶子节点自己的**前缀
    （ReARTeR 正常路径误用最后一次选中节点的前缀 —— docs/01 §11.8 已指出）。
    """
    return {
        "instance_id": instance_id or node.instance_id,
        "node_key": node.node_key,
        "mc_score": node.mc_score,
        "n_steps": len(node.prefix_steps),
        "type": entry_type,
    }


__all__ = ["split_middle", "locate_error", "annotation_entry"]
