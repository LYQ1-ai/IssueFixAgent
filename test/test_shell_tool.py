"""``agent/shell_tool.py``（只读 shell 工具）测试。

用法（在项目根目录 /home/lyq/PycharmProjects/CodeAgentRL 下执行）::

    # 只跑单元测试（mock docker CLI，无需 Docker / 网络）
    python -m pytest test/test_shell_tool.py -v

    # 全部测试（含真实 Docker 集成测试，需 Docker daemon 与 GitHub 网络）
    python -m pytest test/ -v --run-integration

说明：shell_tool 不再管理执行环境 —— 容器由 ``agent.init_env`` 创建并传入
``ShellTool(container=...)``。因此单元测试拦截 ``subprocess.run`` 模拟 docker
exec，集成测试则先经 ``EnvManager.get_env`` 获取真实容器再执行。
"""

import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

# 保证 ``agent`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import github_available, resolve_test_image  # noqa: E402
from agent.shell_tool import (  # noqa: E402
    ReadOnlyViolation,
    ShellTool,
    ShellToolConfig,
    validate_readonly,
)


# ----------------------------------------------------------------------
# 工具：可编程的 docker CLI 假实现（拦截 subprocess.run）
# ----------------------------------------------------------------------

class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeDockerShell:
    """按命令行内容分发响应的 docker 假实现（针对 shell_tool 的 docker exec）。

    通过 ``monkeypatch.setattr("agent.shell_tool.subprocess.run", fake)`` 注入，
    使 ShellTool 走完整真实逻辑（只读校验 -> docker exec -> 返回 dict）而不
    触碰真实 Docker。
    """

    def __init__(self):
        self.calls: list[tuple[list, dict]] = []
        self.exec_rc = 0
        self.exec_stdout = "mock-exec-output"
        self.raise_timeout = False  # 模拟 subprocess.TimeoutExpired
        self.raise_other = False    # 模拟其他异常

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((cmd, kwargs))
        joined = " ".join(cmd)
        if "docker" in joined and " exec " in joined:
            if self.raise_timeout:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 30))
            if self.raise_other:
                raise RuntimeError("boom")
            return FakeResult(returncode=self.exec_rc, stdout=self.exec_stdout)
        return FakeResult(returncode=0)

    # ---- 断言辅助 ----
    def exec_calls(self) -> list[str]:
        """返回所有 docker exec 调用的命令行（join 形式）。"""
        return [
            " ".join(cmd)
            for cmd, _ in self.calls
            if len(cmd) > 1 and cmd[0] == "docker" and cmd[1] == "exec"
        ]


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDockerShell()
    monkeypatch.setattr("agent.shell_tool.subprocess.run", fake)
    return fake


def make_tool(**cfg_kwargs) -> ShellTool:
    return ShellTool(
        container="fake-container",
        config=ShellToolConfig(**cfg_kwargs),
    )


# ----------------------------------------------------------------------
# validate_readonly：纯逻辑单元测试
# ----------------------------------------------------------------------

class TestValidateReadonly:
    def test_allows_readonly_commands(self):
        ok_cases = [
            "ls -la",
            "cat src/main.py",
            "grep -rn 'TODO' .",
            "find . -name '*.py'",
            "git log --oneline -5",
            "git diff HEAD~1",
            "git status --short",
            "python3 -c 'print(open(\"main.py\").read())'",
            "head -50 README.md && tail -10 main.py",
            "cd src && grep -r foo . | head",
            "sed -n '10,20p' file.txt",
            "echo 'hello'",
            "wc -l *.py",
            "pwd && ls -R .",
        ]
        for c in ok_cases:
            validate_readonly(c)  # 不应抛异常

    def test_blocks_write_commands(self):
        bad_cases = [
            "rm -rf /",
            "rm -rf /repo",
            "rm -rf /usr",
            "mv a b",
            "cp a b",
            "touch newfile",
            "mkdir dir",
            "echo 'x' > file",
            "echo 'x' >> file",
            "echo 'x' > /repo/out.txt",
            "echo 'x' > /etc/passwd",
            "sed -i 's/a/b/' f",
            "git add .",
            "git commit -m x",
            "git push origin main",
            "pip install numpy",
            "apt-get update",
            "chmod +x f",
            "chown root f",
            "python3 -c 'open(\"f\",\"w\").write(\"x\")'",
            "cat f | tee out",
            "cat f | tee /repo/out",
            "kill -9 1",
            "make build",
            "touch /repo/x",
        ]
        for c in bad_cases:
            with pytest.raises(ReadOnlyViolation):
                validate_readonly(c)

    def test_tmp_writes_allowed_by_default(self):
        ok = [
            "echo 'x' > /tmp/script.py",
            "echo 'y' >> /tmp/out.log",
            "touch /tmp/scratch",
            "mkdir -p /tmp/work && cat /tmp/work/a.txt",
            "rm -f /tmp/old.log",
            "grep foo file.txt > /dev/null",
        ]
        for c in ok:
            validate_readonly(c)  # allow_tmp_writes=True（默认）

    def test_tmp_writes_blocked_when_disabled(self):
        with pytest.raises(ReadOnlyViolation):
            validate_readonly("touch /tmp/x", allow_tmp_writes=False)

    def test_path_traversal_blocked(self):
        # /tmp/../repo 规范化后落在仓库，必须拒绝
        with pytest.raises(ReadOnlyViolation):
            validate_readonly("rm -rf /tmp/../repo")

    def test_empty_command_blocked(self):
        with pytest.raises(ReadOnlyViolation):
            validate_readonly("   ")

    def test_allowlist_strict_mode(self):
        validate_readonly("ls -la", allowlist_only=True)
        validate_readonly("git log --oneline -3", allowlist_only=True)
        # python3 在白名单中（允许只读分析脚本）
        validate_readonly("python3 /tmp/x.py", allowlist_only=True)
        with pytest.raises(ReadOnlyViolation):
            validate_readonly("perl /tmp/x.pl", allowlist_only=True)  # 不在白名单
        with pytest.raises(ReadOnlyViolation):
            validate_readonly("rm /tmp/x.py", allowlist_only=True)

    def test_error_has_reason_and_command(self):
        try:
            validate_readonly("rm -rf /")
        except ReadOnlyViolation as e:
            assert "rm" in e.reason
            assert e.command == "rm -rf /"
        else:  # pragma: no cover
            pytest.fail("should have raised")


# ----------------------------------------------------------------------
# ShellTool schema
# ----------------------------------------------------------------------

class TestShellToolSchema:
    def test_schema_structure(self):
        schema = make_tool().get_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "bash"
        assert fn["parameters"]["type"] == "object"
        assert "command" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["command"]

    def test_schema_mentions_readonly(self):
        desc = make_tool().get_schema()["function"]["description"]
        assert "read-only" in desc


# ----------------------------------------------------------------------
# ShellTool.execute（mock docker）
# ----------------------------------------------------------------------

class TestShellToolExecute:
    def test_executes_readonly_command(self, fake_docker):
        tool = make_tool()
        result = tool.execute("ls -la")
        assert result["returncode"] == 0
        assert result["output"] == "mock-exec-output"
        assert result["exception_info"] == ""
        calls = fake_docker.exec_calls()
        assert len(calls) == 1
        # 校验 docker exec 参数：workdir、容器名、timeout 包裹
        assert "--workdir /repo" in calls[0]
        assert "fake-container" in calls[0]
        assert "timeout --signal=KILL 30 bash -lc" in calls[0]

    def test_write_command_blocked_before_docker(self, fake_docker):
        tool = make_tool()
        with pytest.raises(ReadOnlyViolation):
            tool.execute("rm -rf /repo")
        assert fake_docker.exec_calls() == []  # 根本没调 docker

    def test_call_magic_is_execute(self, fake_docker):
        tool = make_tool()
        result = tool("pwd")
        assert result["returncode"] == 0

    def test_timeout_returns_error_dict(self, fake_docker):
        fake_docker.raise_timeout = True
        result = make_tool().execute("sleep 100")
        assert result["returncode"] == -1
        assert "timed out" in result["exception_info"].lower()

    def test_other_exception_returns_error_dict(self, fake_docker):
        fake_docker.raise_other = True
        result = make_tool().execute("pwd")
        assert result["returncode"] == -1
        assert "error" in result["exception_info"].lower()

    def test_output_truncation(self, fake_docker):
        fake_docker.exec_stdout = "x" * 5000
        result = make_tool(max_output_chars=100).execute("pwd")
        assert len(result["output"]) < 5000
        assert "elided_chars" in result["output"]

    def test_allowlist_config_enforced(self, fake_docker):
        tool = make_tool(readonly_allowlist_only=True)
        with pytest.raises(ReadOnlyViolation):
            tool.execute("perl /tmp/x.pl")

    def test_no_semaphore_when_unlimited(self, fake_docker):
        tool = make_tool(max_concurrent=0)
        assert tool._semaphore is None
        assert tool.execute("pwd")["returncode"] == 0


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


@pytest.fixture
def live_tool():
    """经 init_env 获取真实环境容器，注入 ShellTool；测试后归还环境。"""
    from agent.init_env import EnvConfig, EnvManager

    mgr = EnvManager(
        EnvConfig(image=resolve_test_image(), name_prefix="codeagentrl-test")
    )
    container = mgr.get_env("octocat/Hello-World")
    tool = ShellTool(
        container,
        config=ShellToolConfig(timeout=60),
    )
    yield tool, mgr, container
    mgr.release_env(container)  # 无论成功失败都归还


@_NEED_DOCKER
@pytest.mark.skipif(
    not github_available(),
    reason="host cannot reach github.com (直连或 127.0.0.1:7890 代理不可用)",
)
@pytest.mark.integration
class TestIntegration:
    def test_readonly_commands_in_real_env(self, live_tool):
        tool, mgr, container = live_tool
        assert container.startswith("codeagentrl-test-")

        r = tool.execute("pwd && ls -la /repo")
        assert r["returncode"] == 0, r["exception_info"]
        assert "/repo" in r["output"]

        r = tool.execute("git log --oneline -3")
        assert r["returncode"] == 0
        assert r["output"].strip()  # 有提交历史

        r = tool.execute("cat README* 2>/dev/null || cat /repo/README*")
        assert r["returncode"] == 0

    def test_write_blocked_in_real_env(self, live_tool):
        tool, _, _ = live_tool
        with pytest.raises(ReadOnlyViolation):
            tool.execute("rm -rf /repo")

    def test_concurrent_requests_same_env(self, live_tool):
        tool, _, _ = live_tool
        n_threads = 6
        results: list = []
        barrier = threading.Barrier(n_threads)

        def worker(i):
            barrier.wait()
            r = tool.execute(f"echo run-{i} && git rev-parse HEAD")
            results.append((i, r["returncode"], r["output"].strip()))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(rc == 0 for _, rc, _ in results)
        # 输出包含每线程的 run-i 前缀，只取最后一行的 git HEAD
        heads = {out.splitlines()[-1] for _, _, out in results}
        assert len(heads) == 1  # 同一容器内 HEAD 一致
