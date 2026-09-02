"""``mcts/env.py``（PLAN §2.2 环境准备）测试。

核心验收点：**无 commit 时不切换** —— SWE-Smith 实例 ``base_commit=None``
推导出的环境参数 ``commit=None``，``prepare_env`` 以 ``get_env(repo, None, patch=...)``
调用（EnvManager 在 commit 非空时才 ``git checkout``，见 ``test/test_init_env.py``
的 ``TestNoCheckoutWhenCommitNone``）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcts.env import instance_env_params, prepare_env  # noqa: E402
from mcts.instances import Gold, Instance  # noqa: E402

SWE_SMITH_PATCH = (
    "diff --git a/src/cli.py b/src/cli.py\n"
    "--- a/src/cli.py\n"
    "+++ b/src/cli.py\n"
    "@@ -1,3 +1,3 @@\n"
    "-old\n"
    "+new\n"
)


def make_instance(**overrides) -> Instance:
    base = dict(
        instance_id="theskumar__python-dotenv.2b8635b7.func_basic__n6cxbsay",
        repo="swesmith/theskumar__python-dotenv.2b8635b7",
        owner="swesmith",
        name="theskumar__python-dotenv",
        commit8="2b8635b7",
        base_commit=None,  # SWE-Smith：恒为 None（单 commit 快照仓库）
        problem_statement="CLI options getting mixed up.",
        patch=SWE_SMITH_PATCH,
        use_patch=True,
        gold=Gold(frozenset(["src/dotenv/cli.py"]), frozenset(), frozenset()),
        source="train",
    )
    base.update(overrides)
    return Instance(**base)


class TestInstanceEnvParams:
    def test_swesmith_no_commit_no_checkout(self):
        """无 commit（base_commit=None）时不切换：commit 推导为 None。"""
        inst = make_instance()
        p = instance_env_params(inst)
        assert p.repo == "swesmith/theskumar__python-dotenv.2b8635b7"
        assert p.commit is None
        assert p.checkout is False  # 无 commit 时不切换（跳过 git checkout）
        assert p.repo_url == "https://github.com/swesmith/theskumar__python-dotenv.2b8635b7.git"

    def test_swesmith_patch_kept_for_apply(self):
        inst = make_instance()
        p = instance_env_params(inst)
        assert p.patch == SWE_SMITH_PATCH  # use_patch=True -> git apply 引入 bug 的 diff

    def test_base_commit_present_checks_out(self):
        inst = make_instance(base_commit="0123456789abcdef")
        p = instance_env_params(inst)
        assert p.commit == "0123456789abcdef"
        assert p.checkout is True

    def test_use_patch_false_drops_patch(self):
        inst = make_instance(use_patch=False)
        p = instance_env_params(inst)
        assert p.patch is None


class TestPrepareEnv:
    def test_delegates_to_get_env_with_commit_none(self, monkeypatch):
        inst = make_instance()
        captured = {}

        def fake_get_env(repo, commit, *, patch=None, manager=None):
            captured.update(repo=repo, commit=commit, patch=patch, manager=manager)
            return "ctr-123"

        monkeypatch.setattr("mcts.env.get_env", fake_get_env)
        container = prepare_env(inst)
        assert container == "ctr-123"
        assert captured["repo"] == "swesmith/theskumar__python-dotenv.2b8635b7"
        assert captured["commit"] is None  # 无 commit 时不切换
        assert captured["patch"] == SWE_SMITH_PATCH

    def test_manual_patch_overrides(self, monkeypatch):
        inst = make_instance()
        captured = {}
        monkeypatch.setattr(
            "mcts.env.get_env",
            lambda repo, commit, *, patch=None, manager=None: captured.update(
                repo=repo, commit=commit, patch=patch
            ) or "ctr",
        )
        prepare_env(inst, patch="manual-diff")
        assert captured["patch"] == "manual-diff"

    def test_manager_passed_through(self, monkeypatch):
        inst = make_instance()
        captured = {}
        manager = object()
        monkeypatch.setattr(
            "mcts.env.get_env",
            lambda repo, commit, *, patch=None, manager=None: captured.update(
                manager=manager
            ) or "ctr",
        )
        prepare_env(inst, manager=manager)
        assert captured["manager"] is manager
