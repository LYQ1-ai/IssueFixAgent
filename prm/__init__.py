# SPDX-License-Identifier: BSD-3-Clause

"""prm —— PRM（过程奖励模型）训练包（阶段 2 / M4，docs/prm_training_plan.md）。

数据流::

    outputs/batch500/state.db ─┐
    outputs/mcts/splits.parquet ─┴─→ prm/raw.py（原始加载，纯函数）
                                     ↓
                 prm/build_dataset.py::PRMDatasetBuilder（节点→样本→标签→去重）
                                     ↓
                 outputs/prm/{train,dev,test}.parquet + length_report + manifest
                                     ↓
                 prm/data.py（collator+截断） + prm/modeling.py（VerdictScorer）
                                     ↓
                 prm/train_prm.py（Trainer 微调） → prm/eval_prm.py（评估/best-of-5）

**硬约束**：``outputs/batch500/state.db`` 严格只读（prm/raw.py::open_db_readonly）；
不改动 ``mcts/``、``agent/``。
"""

from prm.labeling import class_weights, leaf_chain_labels, mixed_label, node_label
from prm.prompts import (
    SYSTEM_PRM_V1,
    TEMPLATE_VERSION,
    TRUNCATION_MARKER,
    USER_CONTEXT_V1,
    USER_INSTRUCTION_V1,
    template_hash,
)

__all__ = [
    "class_weights",
    "leaf_chain_labels",
    "mixed_label",
    "node_label",
    "SYSTEM_PRM_V1",
    "TEMPLATE_VERSION",
    "TRUNCATION_MARKER",
    "USER_CONTEXT_V1",
    "USER_INSTRUCTION_V1",
    "template_hash",
]
