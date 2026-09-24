# SPDX-License-Identifier: BSD-3-Clause

"""项目环境变量装载与 **GPU 选择**（docs/prm_training_plan.md §14.5）。

设计约定（2026-09-14 起）：

- **用哪张卡由环境决定**，不在代码/脚本/配置里写死 GPU 号；
- 在仓库根 `.env` 里设 ``CUDA_VISIBLE_DEVICES=N``（多卡写 ``0,1``）；
- **各 CLI 入口**（`python -m prm.{build_dataset,probe,train_prm,eval_prm}` 的 `main()`）
  在开头调用 :func:`load_project_env` 装载 `.env`，此时**早于任何 torch/CUDA 初始化**
  （`CUDA_VISIBLE_DEVICES` 只在首次 CUDA 初始化时被读取，torch 的 CUDA 是懒初始化）；
- **包导入不装载**：库使用者/测试进程不应被整个 `.env` 污染——需要时显式调用
  :func:`load_project_env`；
- 装载后进程内 ``cuda:0`` 即「被选中的第一张卡」，CLI 的 ``--device`` 默认写
  ``cuda`` 即可，不必也不应写 ``cuda:1`` 这类物理卡号（那会和 `.env` 冲突）。

若环境里已存在同名变量（shell ``export``、调度器注入），**不覆盖**——显式环境优先。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("prm.env")

#: `.env` 路径（仓库根）；可用 ``PRM_ENV_FILE`` 覆盖（测试/多环境用）
_ENV_FILE_ENV = "PRM_ENV_FILE"


def default_env_path() -> Path:
    """仓库根 `.env`（``prm/env.py`` → 上两级）。"""
    return Path(__file__).resolve().parent.parent / ".env"


def _parse_dotenv(text: str) -> dict[str, str]:
    """极简 dotenv 解析（python-dotenv 不可用时的兜底）。

    支持：``KEY=value``、``export KEY=value``、``#`` 注释、空行、值两侧的单/双引号。
    不支持多行值与变量插值（这些用 python-dotenv）。
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_project_env(path: Optional[Path] = None, *, override: bool = False) -> dict[str, str]:
    """装载 `.env` 到 ``os.environ``（幂等；默认不覆盖已有变量）。

    Returns:
        实际**新写入**的环境变量字典（已存在的键不在其中）。
    """
    env_path = Path(path) if path is not None else Path(
        os.environ.get(_ENV_FILE_ENV) or default_env_path())
    if not env_path.exists():
        return {}
    try:
        from dotenv import dotenv_values  # python-dotenv（swanlab 依赖链已带）
        values = {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    except Exception:  # pragma: no cover - 兜底解析路径
        values = _parse_dotenv(env_path.read_text(encoding="utf-8"))

    applied: dict[str, str] = {}
    for key, val in values.items():
        if override or key not in os.environ:
            os.environ[key] = str(val)
            applied[key] = str(val)
    return applied


def visible_devices() -> list[str]:
    """解析 ``CUDA_VISIBLE_DEVICES`` → 物理卡号列表（未设置/空 → ``[]`` 表示全部可见）。"""
    raw = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw or raw == "-1":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def describe() -> str:
    """人类可读的 GPU 选择说明（写日志/报告用）。"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return "CUDA_VISIBLE_DEVICES 未设置（全部卡可见；进程默认用 cuda:0 = 物理卡 0）"
    if cvd.strip() == "-1":
        return "CUDA_VISIBLE_DEVICES=-1（禁用 CUDA，全部走 CPU）"
    return (f"CUDA_VISIBLE_DEVICES={cvd.strip()}"
            f"（进程内 cuda:0 ↔ 物理卡 {visible_devices()[0]}，"
            f"共 {len(visible_devices())} 张可见）")


__all__ = ["default_env_path", "load_project_env", "visible_devices", "describe"]
