# SPDX-License-Identifier: BSD-3-Clause

"""环境准备（PLAN §2.2，对齐 codescout ``src/utils/instance.py::clone_instance``
的 SWE-Smith 分支）—— 把 ``Instance`` 翻译成 ``agent.init_env.get_env`` 的参数。

**无 commit 时不切换**：SWE-Smith 的 ``base_commit`` 恒为 ``None``（单 commit
快照仓库，``repo`` 列本身即仓库地址），因此传给 ``get_env`` 的 ``commit`` 为
``None`` —— ``EnvManager._setup_repo`` 仅在 commit 非空时才执行 ``git checkout``，
无 commit 即跳过切换（保持 HEAD）。``use_patch=True`` 恒成立，克隆后经
``git apply`` 应用 ``patch``（SWE-Smith 的 bug 引入 diff），得到 pre-PR 状态。

本模块不创建 / 不管理容器，全部委托 ``agent.init_env``（容器生命周期归
``EnvManager`` 所有）。``agent`` 包在模块导入时**不加载**（惰性注入
``get_env``），避免纯数据流程（``mcts.instances`` / ``mcts.splits``）拉起
mini-swe-agent。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from mcts.instances import Instance

if TYPE_CHECKING:  # 仅类型标注用（运行时惰性导入）
    from agent.init_env import EnvManager

logger = logging.getLogger("mcts.env")

# 惰性注入：首次 prepare_env 时从 agent.init_env 取真实现（测试可 monkeypatch）
get_env = None  # type: Optional[callable]


def _ensure_get_env():
    global get_env
    if get_env is None:
        from agent.init_env import get_env as _ge

        get_env = _ge
    return get_env


@dataclass(frozen=True)
class EnvParams:
    """一次环境准备所需的参数（对应 ``get_env(repo, commit, patch=...)``）。"""

    repo: str
    commit: Optional[str]
    patch: Optional[str]

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.repo}.git"

    @property
    def checkout(self) -> bool:
        """是否有 commit 需要切换（无 commit 时不切换）。"""
        return self.commit is not None


def instance_env_params(instance: Instance) -> EnvParams:
    """由实例推导环境准备参数。

    - ``commit`` = ``instance.base_commit``：SWE-Smith 恒为 ``None``
      （快照仓库单 commit）-> 无 commit 不切换（跳过 ``git checkout``）；
    - ``patch`` = ``instance.patch``（``use_patch=True`` 时）：克隆后
      ``git apply`` 应用 bug 引入 diff（对齐 ``clone_instance`` 的
      ``patch is not None -> git apply`` 分支）。
    """
    commit = instance.base_commit
    patch = instance.patch if instance.use_patch else None
    return EnvParams(repo=instance.repo, commit=commit, patch=patch)


def prepare_env(
    instance: Instance,
    *,
    manager: Optional["EnvManager"] = None,
    patch: Optional[str] = None,
) -> str:
    """按实例创建一个**全新**执行环境容器，返回容器名（用完由调用方销毁）。

    等价于 codescout ``clone_instance`` 的 SWE-Smith 分支
    （clone 免 checkout + ``git apply`` patch），映射到 ``EnvManager`` 上：
    ``get_env(repo, commit=None, patch=patch)``（v2 语义：每次调用新建，不复用）。
    """
    params = instance_env_params(instance)
    effective_patch = patch if patch is not None else params.patch
    logger.info(
        "prepare_env %s: repo=%s commit=%s checkout=%s apply_patch=%s",
        instance.instance_id, params.repo, params.commit, params.checkout,
        effective_patch is not None,
    )
    return _ensure_get_env()(
        params.repo,
        params.commit,
        patch=effective_patch,
        manager=manager,
    )


__all__ = ["EnvParams", "instance_env_params", "prepare_env"]
