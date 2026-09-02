"""pytest 配置：注册 ``integration`` 标记与 ``--run-integration`` 选项。

默认运行只跑单元测试（mock 掉 docker CLI，无需 Docker/网络）；
带 Docker 的真实环境集成测试需显式加 ``--run-integration``。

集成测试镜像**固定使用项目预构建镜像 ``codeagentrl-agent:ubuntu24``**（自带
git / unzip / ripgrep，``EnvManager._ensure_tools`` 探测全部命中，完全跳过
容器内 apt 安装），避免基础镜像（如 ``python:3.11-slim``）在容器构建时触发
apt 安装导致挂起 / 超时；可用环境变量 ``CODEAGENTRL_TEST_IMAGE`` 覆盖。
"""

import os
import subprocess

import pytest

TEST_IMAGE_ENV = "CODEAGENTRL_TEST_IMAGE"
PREBUILT_IMAGE = "codeagentrl-agent:ubuntu24"  # 项目预构建镜像：自带 git/unzip/rg


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.integration (require Docker + network)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a working Docker daemon and outbound network to "
        "github.com (opt-in via --run-integration)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(
        reason="integration test; run with --run-integration to enable"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def _image_exists(image: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "images", "-q", image],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # pragma: no cover - docker CLI 异常视为镜像不可用
        return False


def resolve_test_image() -> str:
    """解析集成测试用镜像。

    默认**固定**返回项目预构建镜像 ``codeagentrl-agent:ubuntu24``：自带
    git/unzip/ripgrep，可完全跳过容器内 apt 安装（避免 apt 挂起 / 超时）；
    仅当显式设置环境变量 ``CODEAGENTRL_TEST_IMAGE`` 时使用该值覆盖。

    Raises:
        RuntimeError: 预构建镜像在本机不存在且未设置 ``CODEAGENTRL_TEST_IMAGE``
            —— 需先构建镜像，而不是回退到会触发容器内 apt 安装的基础镜像：
            ``docker build -t codeagentrl-agent:ubuntu24 -f agent/Dockerfile .``
    """
    img = os.environ.get(TEST_IMAGE_ENV)
    if img:
        return img
    if not _image_exists(PREBUILT_IMAGE):
        raise RuntimeError(
            f"integration test image {PREBUILT_IMAGE!r} not found locally; "
            "build it first with: docker build -t codeagentrl-agent:ubuntu24 "
            "-f agent/Dockerfile .  (falling back to base images would trigger "
            "container-side apt installs which can hang/time out)"
        )
    return PREBUILT_IMAGE


def github_available() -> bool:
    """宿主机能否访问 github.com（直连或经全局 git 代理，如 127.0.0.1:7890）。"""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "https://github.com/octocat/Hello-World.git", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:  # pragma: no cover
        return False
