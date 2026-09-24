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

**环境变量 / GPU 选择**：仓库根 ``.env``（:mod:`prm.env`）由**各 CLI 入口**
（``python -m prm.{build_dataset,probe,train_prm,eval_prm}`` 的 ``main()``）在开头装载，
**不在包导入时装载**——包导入装载会把整个 `.env`（含 agent 侧配置）灌进宿主进程，
污染库使用者与其他测试。库用法需自行调用 :func:`prm.env.load_project_env`。
「用哪张卡」由 ``.env`` 的 ``CUDA_VISIBLE_DEVICES`` 决定；入口在导入 torch 之前装载，
故进程内 ``cuda:0`` 即被选中的那张卡（用 :func:`prm.env.describe` 打印映射）。
"""

from prm.env import describe as describe_env, load_project_env, visible_devices

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
    "describe_env",
    "leaf_chain_labels",
    "load_project_env",
    "mixed_label",
    "node_label",
    "SYSTEM_PRM_V1",
    "TEMPLATE_VERSION",
    "TRUNCATION_MARKER",
    "USER_CONTEXT_V1",
    "USER_INSTRUCTION_V1",
    "template_hash",
    "visible_devices",
]
