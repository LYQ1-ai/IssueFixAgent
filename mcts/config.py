# SPDX-License-Identifier: BSD-3-Clause

"""``mcts`` 包配置加载：读取 ``mcts/config.yaml``（PLAN §4）。

所有阶段共用一份配置（数据 / 过滤 / gold / split / env / outputs /
后续 M1–M5 的 MCTS 超参占位），加载结果就是普通 dict，后续里程碑直接
``cfg["mcts"]["n_rollouts"]`` 读取。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Optional[str | os.PathLike] = None) -> dict[str, Any]:
    """加载 YAML 配置；缺省读包内 ``config.yaml``。文件不存在或内容为空返回 ``{}``。"""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def project_root() -> Path:
    """项目根目录（``mcts/config.py`` 的上级上级）。"""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: Optional[str | os.PathLike]) -> Optional[Path]:
    """把配置里的路径解析成绝对路径（相对路径以项目根目录为基准）。"""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else project_root() / p


__all__ = ["DEFAULT_CONFIG_PATH", "load_config", "project_root", "resolve_path"]
