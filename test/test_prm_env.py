# SPDX-License-Identifier: BSD-3-Clause

"""prm/env.py 单测（.env 装载 + GPU 选择解析）。

全离线、无 torch/GPU 依赖（CodeAgentRL 环境也会跑）。覆盖：
- 极简 dotenv 解析（导出前缀/注释/引号/空行）；
- 装载**不覆盖**已存在的环境变量（shell/调度器显式优先）；
- `CUDA_VISIBLE_DEVICES` → 物理卡号列表解析（含未设置、-1、多卡、空项）；
- `describe()` 的三种口径（未设置 / 禁用 / 指定）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prm import env as prm_env  # noqa: E402


class TestDotenvParsing:
    def test_parse_basic(self):
        text = (
            "# comment\n"
            "\n"
            "A=1\n"
            "export B=two\n"
            'C="quoted value"\n'
            "D='single'\n"
            "MALFORMED\n"
            "E=a=b\n"
        )
        got = prm_env._parse_dotenv(text)
        assert got == {"A": "1", "B": "two", "C": "quoted value",
                       "D": "single", "E": "a=b"}

    def test_load_does_not_override(self, tmp_path, monkeypatch):
        f = tmp_path / ".env"
        f.write_text("PRM_TEST_ALPHA=from_file\nPRM_TEST_BETA=from_file\n", encoding="utf-8")
        monkeypatch.setenv("PRM_TEST_ALPHA", "from_shell")
        monkeypatch.delenv("PRM_TEST_BETA", raising=False)
        applied = prm_env.load_project_env(f)
        assert applied == {"PRM_TEST_BETA": "from_file"}          # 已存在的不在 applied
        import os
        assert os.environ["PRM_TEST_ALPHA"] == "from_shell"       # 显式环境优先
        assert os.environ["PRM_TEST_BETA"] == "from_file"

    def test_load_missing_file_is_noop(self, tmp_path):
        assert prm_env.load_project_env(tmp_path / "nope.env") == {}

    def test_env_file_override_var(self, tmp_path, monkeypatch):
        """PRM_ENV_FILE 可指定 .env 路径（多环境/测试用）。"""
        f = tmp_path / "custom.env"
        f.write_text("PRM_TEST_GAMMA=1\n", encoding="utf-8")
        monkeypatch.setenv(prm_env._ENV_FILE_ENV, str(f))
        monkeypatch.delenv("PRM_TEST_GAMMA", raising=False)
        assert prm_env.load_project_env() == {"PRM_TEST_GAMMA": "1"}


class TestVisibleDevices:
    @pytest.mark.parametrize("raw,expected", [
        (None, []),            # 未设置 → 全部可见
        ("", []),
        ("-1", []),            # 禁用 CUDA
        ("1", ["1"]),
        ("0,1", ["0", "1"]),
        (" 2 , 3 ", ["2", "3"]),
    ])
    def test_parse(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
        assert prm_env.visible_devices() == expected

    def test_describe_variants(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert "未设置" in prm_env.describe()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert "禁用 CUDA" in prm_env.describe()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        msg = prm_env.describe()
        assert "CUDA_VISIBLE_DEVICES=3" in msg and "物理卡 3" in msg
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
        assert "共 2 张可见" in prm_env.describe()


def test_package_import_does_not_load_env(monkeypatch, tmp_path):
    """**包导入不装载 `.env`**——回归护栏。

    曾经的实现是在 `prm/__init__.py` 里导入即装载，结果 pytest 收集 PRM 测试时就
    把整个 `.env`（含 `CODEAGENTRL_IMAGE` 等 agent 侧配置）灌进进程，污染了
    `test/test_init_env.py` 的 14 个用例。装载只应发生在 CLI 入口（`main()`）。
    """
    f = tmp_path / ".env"
    f.write_text("PRM_TEST_EPSILON=must_not_leak\n", encoding="utf-8")
    monkeypatch.setenv(prm_env._ENV_FILE_ENV, str(f))
    monkeypatch.delenv("PRM_TEST_EPSILON", raising=False)

    import importlib
    import prm
    importlib.reload(prm)
    import os
    assert "PRM_TEST_EPSILON" not in os.environ


def test_cli_entry_point_loads_env(monkeypatch, tmp_path):
    """CLI 入口（main 开头）装载 `.env`——GPU/CUDA 选择靠它。"""
    f = tmp_path / ".env"
    f.write_text("PRM_TEST_ZETA=loaded_by_cli\n", encoding="utf-8")
    monkeypatch.setenv(prm_env._ENV_FILE_ENV, str(f))
    monkeypatch.delenv("PRM_TEST_ZETA", raising=False)

    # 用 build_dataset 入口：它在 CodeAgentRL（无 torch）环境也能跑，适合全员回归
    from prm.build_dataset import main as bd_main
    with pytest.raises(SystemExit):           # --help 会 SystemExit，但装载已发生
        bd_main(["--help"])
    import os
    assert os.environ["PRM_TEST_ZETA"] == "loaded_by_cli"
