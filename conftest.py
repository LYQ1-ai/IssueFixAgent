# SPDX-License-Identifier: BSD-3-Clause

"""项目根 conftest：提供沙箱兼容的 ``workdir`` fixture。

pytest 内置 ``tmp_path`` 默认落在系统临时目录（Windows 沙箱下不可写），
这里统一重定向到项目根 ``.pytest-tmp/<test>-<uuid>``（Linux 上同样可用），
测试无需依赖 ``tmp_path``。
"""

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def workdir(request) -> Path:
    """每个测试一个独立工作目录（落盘产物 / 断点续跑验证用）。"""
    base = Path(__file__).resolve().parent / ".pytest-wd"
    d = base / f"{request.node.name}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
