# SPDX-License-Identifier: BSD-3-Clause

"""标签纯函数（docs/prm_training_plan.md §5.2）。

主训练标签 = **混合 soft**：``label = (1-w)·binary + w·soft``（默认 w=0.2，
即 0.8·binary + 0.2·soft）。对照阅读：ReARTeR 原始做法是纯二值 ``mc > 0.5``
（``ref_papers/ReARTeR/PRM_Data/process_to_prm_data.py`` 的
``label = mc > 0.5``）；本项目以混合 soft 缓冲 29.2% 的 MC 边界标签噪声（F6）。

leaf 链式回填（§5.1）：leaf 节点（非 root 且 mc=0，即"从这里续跑 5 次全失败"，
``mcts/locate.py`` 的首个错误步定位语义）的最后一步标 0，其之前的步骤链式回填
标 1 —— ``[1.0]*(n-1) + [0.0]``。
"""

from __future__ import annotations

from typing import Sequence


def node_label(mc: float) -> tuple[int, float]:
    """节点样本标签：``(binary, soft) = (int(mc > 0.5), mc)``。"""
    return int(mc > 0.5), float(mc)


def mixed_label(binary: int, soft: float, soft_w: float = 0.2) -> float:
    """混合 soft 标签（主训练目标）：``(1 - soft_w)·binary + soft_w·soft``。"""
    return (1.0 - soft_w) * float(binary) + soft_w * float(soft)


def leaf_chain_labels(n_steps: int) -> list[float]:
    """leaf 前缀展开的 L 条样本标签：前 L-1 步回填 1.0，末步（被判定为首个
    错误步）标 0.0；``n_steps=1`` 时仅一条负样本 ``[0.0]``。"""
    if n_steps < 1:
        raise ValueError(f"n_steps 必须 >= 1，得到 {n_steps}")
    return [1.0] * (n_steps - 1) + [0.0]


def class_weights(binary_labels: Sequence[int], cap: float = 4.0) -> tuple[float, float]:
    """正负类权重（§7.3 加权 soft-BCE 用）：``(w_pos, w_neg) = (1.0, min(cap, n_pos/n_neg))``。

    在 **train split 内**按 ``label_binary`` 统计一次（预期 ≈5.56:1 → ``w_neg``
    触顶 4），写入 manifest 与 config，训练/评估共用。
    """
    n = len(binary_labels)
    if n == 0:
        return 1.0, 1.0
    n_pos = sum(1 for b in binary_labels if int(b) == 1)
    n_neg = n - n_pos
    if n_neg == 0:
        return 1.0, 1.0  # 全正：不加权（极端不均衡时应检查数据）
    return 1.0, float(min(cap, n_pos / max(1, n_neg)))


__all__ = ["node_label", "mixed_label", "leaf_chain_labels", "class_weights"]
