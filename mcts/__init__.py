# SPDX-License-Identifier: BSD-3-Clause

"""阶段 1（PLAN.md）数据生成引擎包：M0 数据预处理已完成的部分。

- ``mcts.instances``：数据读取 / 过滤 / GT 抽取（迁移 codescout 数据处理部分）；
- ``mcts.splits``：三层划分（60% 生成池 + repo 级 80/10/10 + instance 映射）；
- ``mcts.env``：环境准备（无 commit 时不切换，对齐 codescout clone_instance）；
- ``mcts.config``：``config.yaml`` 加载。

注意：``mcts.env`` 不在包导入时加载（其 ``prepare_env`` 惰性导入
``agent.init_env``，避免纯数据流程拉起 mini-swe-agent）。

M1–M5（llm / reward / steps / node / locate / replay / pipeline / run_mcts /
prm 训练）按 PLAN §2–§3 逐步补充。
"""

from mcts.config import load_config, project_root, resolve_path
from mcts.instances import (
    Gold,
    Instance,
    build_dataset,
    extract_gold,
    filter_instances,
    instances_from_parquet,
    is_python_path,
    parse_repo,
    patch_creates_or_deletes_files,
    read_instances,
)
from mcts.splits import SPLIT_COLUMNS, build_splits, make_splits

__all__ = [
    "Gold",
    "Instance",
    "load_config",
    "project_root",
    "resolve_path",
    "build_dataset",
    "extract_gold",
    "filter_instances",
    "instances_from_parquet",
    "is_python_path",
    "parse_repo",
    "patch_creates_or_deletes_files",
    "read_instances",
    "SPLIT_COLUMNS",
    "build_splits",
    "make_splits",
]
