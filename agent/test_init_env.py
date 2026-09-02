"""``agent/init_env.py``（执行环境容器管理器）测试。

用法（在项目根目录 /home/lyq/PycharmProjects/CodeAgentRL 下执行）::

    # 只跑单元测试（mock docker，无需 Docker / 网络）
    python -m pytest test/ -v

    # 单元 + 真实 Docker 集成测试（需要 Docker daemon 与 GitHub 网络）
    python -m pytest test/ -v --run-integration

    # 集成测试镜像：默认固定使用项目预构建镜像 codeagentrl-agent:ubuntu24
    # （自带 git/unzip/rg，免容器内 apt，避免 apt 安装挂起/超时）；
    # 也可用 CODEAGENTRL_TEST_IMAGE 覆盖：
    CODEAGENTRL_TEST_IMAGE=my-image pytest test/ -v --run-integration
"""

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# 保证 ``agent`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import github_available, resolve_test_image  # noqa: E402
from agent.init_env import (  # noqa: E402
    EnvConfig,
    EnvManager,
    _normalize_repo,
    _sanitize,
    get_env,
    release_env,
)


# ----------------------------------------------------------------------
# 工具：可编程的 docker CLI 假实现（拦截 subprocess.run）
# ----------------------------------------------------------------------

class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeDocker:
    """按命令行内容分发响应的 docker 假实现。

    通过 ``monkeypatch.setattr("agent.init_env.subprocess.run", fake)`` 注入，
    使 EnvManager 走完整真实逻辑（创建/校验/复用/释放）而不触碰真实 Docker。
    """

    def __init__(self):
        self.calls: list[tuple[list, dict]] = []
        # 可调行为
        self.run_rc = 0                 # docker run 的返回码（0=成功）
        self.inspect_state = "true"     # inspect 输出：true/false
        self.inspect_missing = False    # inspect 返回非零（容器不存在）
        self.start_rc = 0               # docker start 返回码
        self.rm_rc = 0                  # docker rm -f 返回码
        self.clone_fail_count = 0       # git clone 连续失败的次数（测试 CA 重试）
        self.git_present = True         # command -v git 结果
        self.rg_present = True          # command -v rg 结果
        self.cp_rc = 0                  # docker cp 返回码

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((cmd, kwargs))
        joined = " ".join(cmd)
        op = cmd[1] if len(cmd) > 1 and cmd[0] == "docker" else None

        if op == "run":
            return FakeResult(stdout="fake-container-id", returncode=self.run_rc)
        if op == "cp":
            return FakeResult(returncode=self.cp_rc)
        if op == "inspect":
            if self.inspect_missing:
                return FakeResult(returncode=1, stderr="no such container")
            return FakeResult(stdout=self.inspect_state)
        if op == "start":
            return FakeResult(returncode=self.start_rc)
        if op == "rm":
            return FakeResult(returncode=self.rm_rc)
        if op == "exec":
            if "command -v git" in joined:
                return FakeResult(returncode=0 if self.git_present else 1)
            if "command -v rg" in joined:
                return FakeResult(returncode=0 if self.rg_present else 1)
            if "apt-get" in joined:
                return FakeResult(returncode=0)
            if "git clone" in joined:
                if self.clone_fail_count > 0:
                    self.clone_fail_count -= 1
                    return FakeResult(returncode=1, stderr="SSL certificate problem")
                return FakeResult(returncode=0)
            if "git -C" in joined:
                return FakeResult(returncode=0)
            if "rm -rf" in joined:
                return FakeResult(returncode=0)
            return FakeResult(returncode=0)
        return FakeResult(returncode=0)

    # ---- 断言辅助 ----
    def count(self, *markers: str) -> int:
        """按整条命令行子串匹配（用于 git clone / ca-certificates 等 exec 参数）。"""
        n = 0
        for cmd, _ in self.calls:
            joined = " ".join(cmd)
            if all(m in joined for m in markers):
                n += 1
        return n

    def count_op(self, op: str) -> int:
        """按 docker 子命令位置匹配（docker run / rm / start 等顶层操作）。"""
        return sum(
            1
            for cmd, _ in self.calls
            if len(cmd) > 1 and cmd[0] == "docker" and cmd[1] == op
        )


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr("agent.init_env.subprocess.run", fake)
    return fake


def make_mgr(**cfg_kwargs) -> EnvManager:
    return EnvManager(EnvConfig(**cfg_kwargs))


# ----------------------------------------------------------------------
# 纯逻辑单元测试
# ----------------------------------------------------------------------

class TestNormalize:
    def test_valid_repo(self):
        assert _normalize_repo("django/django") == "django/django"

    def test_strips_git_suffix(self):
        assert _normalize_repo("django/django.git") == "django/django"

    def test_rejects_without_owner(self):
        with pytest.raises(ValueError):
            _normalize_repo("just-a-name")


class TestSanitize:
    def test_docker_name_safe(self):
        out = _sanitize("django/django")
        assert all(c.isalnum() or c in "._-" for c in out)
        assert "/" not in out

    def test_commit_shortening(self):
        mgr = make_mgr()
        name = mgr._new_container_name("django/django", "0123456789abcdef")
        assert "01234567" in name  # commit 取前 8 位
        assert name.startswith("codeagentrl-django__django-")


# ----------------------------------------------------------------------
# 新建 / 销毁（不复用：每次 get_env 都是全新容器，mock docker）
# ----------------------------------------------------------------------

class TestFreshEnv:
    def test_each_get_creates_fresh_container(self, fake_docker):
        mgr = make_mgr()
        c1 = mgr.get_env("django/django", "abc123")
        c2 = mgr.get_env("django/django", "abc123")
        assert c1 != c2                     # 不复用：同一 key 两次调用 = 两个不同容器
        assert fake_docker.count_op("run") == 2

    def test_different_commit_different_container(self, fake_docker):
        mgr = make_mgr()
        c1 = mgr.get_env("django/django", "abc111")
        c2 = mgr.get_env("django/django", "def222")
        assert c1 != c2
        assert fake_docker.count_op("run") == 2

    def test_commit_none_always_fresh(self, fake_docker):
        mgr = make_mgr()
        c1 = mgr.get_env("swesmith/x__y.abc", None)
        c2 = mgr.get_env("swesmith/x__y.abc", None)
        c3 = mgr.get_env("swesmith/x__y.abc", "deadbeef")
        assert len({c1, c2, c3}) == 3       # 全部独立容器

    def test_concurrent_creates_distinct_containers(self, fake_docker):
        mgr = make_mgr()
        n_threads = 8
        names: list = []
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            names.append(mgr.get_env("django/django", "abc123"))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(names)) == n_threads  # 并发创建互不冲突（uuid 容器名）
        assert fake_docker.count_op("run") == n_threads


class TestReleaseEnv:
    def test_release_removes_container(self, fake_docker):
        mgr = make_mgr()
        c = mgr.get_env("django/django", "abc123")
        mgr.release_env(c)
        assert fake_docker.count_op("rm") == 1

    def test_release_idempotent(self, fake_docker):
        mgr = make_mgr()
        c = mgr.get_env("django/django", "abc123")
        mgr.release_env(c)
        mgr.release_env(c)  # 不抛异常
        mgr.release_env("")  # 空值安全
        mgr.release_env(None)

    def test_release_unknown_name_no_error(self, fake_docker):
        make_mgr().release_env("not-in-cache")


class TestCreationInternals:
    def test_creation_failure_cleans_up(self, fake_docker):
        fake_docker.run_rc = 1
        mgr = make_mgr()
        with pytest.raises(RuntimeError):
            mgr.get_env("django/django", "abc123")
        assert fake_docker.count_op("rm") == 1  # 半成品容器被清理

    def test_clone_failure_retries_with_ca_certificates(self, fake_docker):
        fake_docker.clone_fail_count = 1  # 第一次 clone 失败（SSL），第二次成功
        mgr = make_mgr()
        c = mgr.get_env("django/django", "abc123")
        assert c.startswith("codeagentrl-")
        assert fake_docker.count("ca-certificates") >= 1
        assert fake_docker.count("git clone") == 2

    def test_patch_applied_via_stdin(self, fake_docker):
        patch_text = "diff --git a/x.py b/x.py\n"
        mgr = make_mgr()
        c = mgr.get_env("django/django", "abc123", patch=patch_text)
        assert c.startswith("codeagentrl-")
        apply_calls = [
            kw["input"]
            for cmd, kw in fake_docker.calls
            if "git -C /repo apply -" in " ".join(cmd)
        ]
        assert apply_calls == [patch_text]

    def test_installs_git_when_missing(self, fake_docker):
        fake_docker.git_present = False
        fake_docker.rg_present = False
        mgr = make_mgr()
        mgr.get_env("django/django", "abc123")
        assert fake_docker.count("apt-get") >= 1


class TestRepoZipCache:
    """创建容器前查本地 zip 缓存：命中走 docker cp + 解压；
    未命中但配置了缓存目录 -> 先在宿主机 clone 打成 zip 入库，之后与命中路径一致；
    未配置缓存目录 -> 容器内 git clone 回退。"""

    def test_zip_hit_uses_docker_cp_not_clone(self, fake_docker, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "django__django.zip").write_bytes(b"fake-zip")  # fake 不解析内容
        mgr = make_mgr(repo_cache_dir=str(cache))
        c = mgr.get_env("django/django", "abc123")
        assert c.startswith("codeagentrl-")
        assert fake_docker.count_op("cp") == 1          # docker cp 复制 zip
        assert fake_docker.count("unzip -q") == 1       # 容器内解压
        assert fake_docker.count("git clone") == 0      # 不再 git clone
        assert fake_docker.count("git -C /repo checkout") == 1  # 仍切 commit

    def test_zip_hit_still_checks_out_commit(self, fake_docker, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "django__django.zip").write_bytes(b"fake-zip")
        mgr = make_mgr(repo_cache_dir=str(cache))
        mgr.get_env("django/django", "deadbeef1234")
        checkout_calls = [
            " ".join(cmd)
            for cmd, _ in fake_docker.calls
            if "git -C /repo checkout" in " ".join(cmd)
        ]
        assert checkout_calls and "deadbeef1234" in checkout_calls[0]

    def test_zip_miss_auto_populates_then_same_path(
        self, fake_docker, tmp_path, monkeypatch
    ):
        """缓存未命中：先在宿主机拉取 repo 打成 zip 存入缓存，之后与缓存命中
        路径完全一致（docker cp + 解压 + checkout），容器内不再 git clone。"""
        cache = tmp_path / "cache"
        cache.mkdir()
        mgr = make_mgr(repo_cache_dir=str(cache))

        def fake_populate(repo, commit=None):  # 模拟宿主机 clone + zip 入库
            p = Path(cache) / f"{_sanitize(repo)}.zip"
            p.write_bytes(b"fake-zip")
            return str(p)

        monkeypatch.setattr(mgr, "populate_repo_cache", fake_populate)
        c = mgr.get_env("django/django", "abc123")
        assert c.startswith("codeagentrl-")
        assert (cache / "django__django.zip").exists()  # zip 已存入缓存
        assert fake_docker.count_op("cp") == 1          # docker cp 复制 zip
        assert fake_docker.count("unzip -q") == 1       # 容器内解压
        assert fake_docker.count("git clone") == 0      # 容器内不再 clone
        assert fake_docker.count("git -C /repo checkout") == 1  # 仍切 commit

    def test_zip_miss_populate_failure_raises(self, fake_docker, tmp_path, monkeypatch):
        """缓存补齐失败（如宿主机 clone 失败）时 get_env 报错，且不产生容器。"""
        cache = tmp_path / "cache"
        cache.mkdir()
        mgr = make_mgr(repo_cache_dir=str(cache))
        monkeypatch.setattr(
            mgr, "populate_repo_cache",
            lambda repo, commit=None: (_ for _ in ()).throw(RuntimeError("host clone failed")),
        )
        with pytest.raises(RuntimeError, match="host clone failed"):
            mgr.get_env("django/django", "abc123")
        assert fake_docker.count_op("run") == 0  # 补齐在 docker run 之前，未建容器

    def test_cache_dir_unset_clones(self, fake_docker):
        mgr = make_mgr(repo_cache_dir=None)  # 未配置缓存目录
        mgr.get_env("django/django", "abc123")
        assert fake_docker.count_op("cp") == 0
        assert fake_docker.count("git clone") == 1

    def test_find_repo_zip(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "django__django.zip").write_bytes(b"x")
        mgr = make_mgr(repo_cache_dir=str(cache))
        assert mgr._find_repo_zip("django/django") == str(cache / "django__django.zip")
        assert mgr._find_repo_zip("other/repo") is None
        assert make_mgr(repo_cache_dir=None)._find_repo_zip("django/django") is None

    def test_populate_repo_cache_without_cache_dir_raises(self, fake_docker):
        mgr = make_mgr(repo_cache_dir=None)
        with pytest.raises(RuntimeError, match="repo_cache_dir"):
            mgr.populate_repo_cache("django/django")


class TestNoCheckoutWhenCommitNone:
    """无 commit 时不切换（git checkout 跳过）—— SWE-Smith 快照仓库
    （base_commit=None）环境准备的关键行为，两条路径（zip 缓存 / 容器内 clone）
    都必须满足。"""

    def test_clone_path_no_checkout(self, fake_docker):
        mgr = make_mgr(repo_cache_dir=None)  # 未配置 zip 缓存 -> 容器内 git clone
        mgr.get_env("swesmith/x__y.abc", None)
        assert fake_docker.count("git clone") == 1
        assert fake_docker.count("git -C /repo checkout") == 0  # 无 commit 不切换

    def test_zip_path_no_checkout(self, fake_docker, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "swesmith__x__y.abc.zip").write_bytes(b"fake-zip")
        mgr = make_mgr(repo_cache_dir=str(cache))
        mgr.get_env("swesmith/x__y.abc", None)
        assert fake_docker.count_op("cp") == 1          # 走 docker cp + 解压
        assert fake_docker.count("git clone") == 0
        assert fake_docker.count("git -C /repo checkout") == 0  # 无 commit 不切换

    def test_with_commit_still_checks_out(self, fake_docker):
        mgr = make_mgr(repo_cache_dir=None)
        mgr.get_env("swesmith/x__y.abc", "deadbeef1234")
        checkout_calls = [
            " ".join(cmd)
            for cmd, _ in fake_docker.calls
            if "git -C /repo checkout" in " ".join(cmd)
        ]
        assert checkout_calls and "deadbeef1234" in checkout_calls[0]

    def test_none_and_commit_are_separate_envs(self, fake_docker):
        mgr = make_mgr()
        c1 = mgr.get_env("swesmith/x__y.abc", None)
        c2 = mgr.get_env("swesmith/x__y.abc", "deadbeef")
        assert c1 != c2  # (repo, None) 与 (repo, commit) 是不同环境键


class TestModuleLevelAPI:
    def test_get_env_and_release_env(self, fake_docker, monkeypatch):
        # 模块级函数使用默认管理器
        monkeypatch.setattr("agent.init_env._default_manager", make_mgr())
        c = get_env("django/django", "abc123")
        assert c.startswith("codeagentrl-")
        release_env(c)
        assert fake_docker.count_op("rm") == 1

    def test_module_get_env_honors_empty_custom_manager(self, fake_docker):
        """回归：模块级 get_env 必须显式判 None（自定义 manager 不被静默丢弃）。"""
        mgr = make_mgr(name_prefix="custom-mgr")
        c = get_env("django/django", "abc123", manager=mgr)
        assert c.startswith("custom-mgr-")   # 用的是传入的 manager，不是默认配置

    def test_module_release_env_honors_custom_manager(self, fake_docker):
        mgr = make_mgr(name_prefix="custom-mgr")
        c = get_env("django/django", "abc123", manager=mgr)
        release_env(c, manager=mgr)
        assert fake_docker.count_op("rm") == 1


# ----------------------------------------------------------------------
# 真实 Docker 集成测试（--run-integration）
# ----------------------------------------------------------------------

def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


_NEED_DOCKER = pytest.mark.skipif(
    not _docker_available(), reason="docker daemon not available"
)

_NEED_GITHUB = pytest.mark.skipif(
    not github_available(),
    reason="host cannot reach github.com (直连或 127.0.0.1:7890 代理不可用)",
)


# 集成测试用仓库/commit（test_zip_cache_path 使用）
HELLO_REPO = "octocat/Hello-World"
HELLO_COMMIT = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"


@pytest.fixture
def live_mgr():
    # 不复用语义：无缓存，无需 close_all；测试内自行 release 清理
    return EnvManager(EnvConfig(image=resolve_test_image(), name_prefix="codeagentrl-test"))


@_NEED_DOCKER
@pytest.mark.integration
class TestIntegration:
    @_NEED_GITHUB
    def test_create_checkout_release(self, live_mgr):
        repo = "octocat/Hello-World"
        commit = "7fd1a60b01f91b314f59955a4e4d4e80d8edf11d"
        c1 = live_mgr.get_env(repo, commit)
        assert c1.startswith("codeagentrl-test-")
        # 容器内 HEAD 必须是指定 commit
        r = subprocess.run(
            ["docker", "exec", "--workdir", "/repo", c1,
             "bash", "-lc", "git rev-parse HEAD"],
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0
        assert r.stdout.strip() == commit
        # 归还 -> 容器被删除
        live_mgr.release_env(c1)
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        assert c1 not in out.splitlines()

    @_NEED_GITHUB
    def test_swesmith_snapshot_and_ripgrep(self, live_mgr):
        """SWE-Smith 风格：commit=None 保持 HEAD，且容器内装好 ripgrep。"""
        repo = "swesmith/davidhalter__parso.338a5760"
        c = live_mgr.get_env(repo, None)
        try:
            r = subprocess.run(
                ["docker", "exec", "--workdir", "/repo", c,
                 "bash", "-lc",
                 "git rev-parse HEAD && ls /repo | head -3 && command -v rg"],
                capture_output=True, text=True, timeout=600,
            )
            assert r.returncode == 0, r.stderr
            lines = r.stdout.strip().splitlines()
            assert len(lines[0]) == 40  # HEAD 是完整 SHA
            assert lines[-1] == "/usr/bin/rg"  # ripgrep 已安装
        finally:
            live_mgr.release_env(c)

    @_NEED_GITHUB
    def test_zip_miss_auto_populates_real(self, tmp_path):
        """缓存未命中：宿主机真实 clone + zip 入库（自动补齐），
        再走 docker cp + 解压 + checkout，验证 HEAD == 指定 commit。"""
        cache = str(tmp_path / "repo_cache")
        mgr = EnvManager(
            EnvConfig(
                image=resolve_test_image(),
                name_prefix="codeagentrl-test",
                repo_cache_dir=cache,
            )
        )
        c = mgr.get_env(HELLO_REPO, HELLO_COMMIT)
        try:
            # 缓存 zip 已自动生成
            zip_path = os.path.join(cache, f"{HELLO_REPO.replace('/', '__')}.zip")
            assert os.path.isfile(zip_path)
            r = subprocess.run(
                ["docker", "exec", "--workdir", "/repo", c,
                 "bash", "-lc", "git rev-parse HEAD"],
                capture_output=True, text=True, timeout=300,
            )
            assert r.returncode == 0, r.stderr
            assert r.stdout.strip() == HELLO_COMMIT
        finally:
            mgr.release_env(c)

    @_NEED_GITHUB
    def test_zip_cache_path(self, tmp_path):
        """populate_repo_cache 生成 zip 后，get_env 走 docker cp 快速路径并切 commit。"""
        cache = str(tmp_path / "repo_cache")
        mgr = EnvManager(
            EnvConfig(
                image=resolve_test_image(),
                name_prefix="codeagentrl-test",
                repo_cache_dir=cache,
            )
        )
        zip_path = mgr.populate_repo_cache(HELLO_REPO, HELLO_COMMIT)
        assert os.path.isfile(zip_path)
        assert zip_path.endswith(f"{HELLO_REPO.replace('/', '__')}.zip")
        c = mgr.get_env(HELLO_REPO, HELLO_COMMIT)
        try:
            r = subprocess.run(
                ["docker", "exec", "--workdir", "/repo", c,
                 "bash", "-lc", "git rev-parse HEAD"],
                capture_output=True, text=True, timeout=300,
            )
            assert r.returncode == 0, r.stderr
            assert r.stdout.strip() == HELLO_COMMIT
        finally:
            mgr.release_env(c)

    def test_zip_hit_local_repo_offline(self, tmp_path):
        """离线端到端（不依赖 github.com）：本地 git 仓库打成 zip 缓存 ->
        get_env 走 docker cp + 解压 + checkout，验证 HEAD == commit。"""
        import zipfile

        # 1) 宿主机造一个本地 git 仓库并打成 zip（zip 内包含 .git 完整历史）
        src = tmp_path / "src"
        src.mkdir()
        subprocess.run(["git", "init", "-q", str(src)], check=True, capture_output=True)
        (src / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(src), "add", "README.md"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t",
             "commit", "-q", "-m", "init"],
            check=True, capture_output=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        cache = tmp_path / "cache"
        cache.mkdir()
        with zipfile.ZipFile(
            cache / "octocat__Hello-World.zip", "w", zipfile.ZIP_DEFLATED
        ) as zf:
            for root, _dirs, files in os.walk(src):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.relpath(full, src))

        mgr = EnvManager(
            EnvConfig(
                image=resolve_test_image(),
                name_prefix="codeagentrl-test",
                install_ripgrep=False,
                repo_cache_dir=str(cache),
            )
        )
        c = mgr.get_env("octocat/Hello-World", commit)
        try:
            r = subprocess.run(
                ["docker", "exec", "--workdir", "/repo", c,
                 "bash", "-lc", "git rev-parse HEAD && ls /repo"],
                capture_output=True, text=True, timeout=300,
            )
            assert r.returncode == 0, r.stderr
            assert r.stdout.strip().splitlines()[0] == commit
        finally:
            mgr.release_env(c)
