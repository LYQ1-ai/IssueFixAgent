"""``agent/base_agent.py``（在指定容器中运行 mini-swe-agent）测试。

用法（在项目根目录 /home/lyq/PycharmProjects/CodeAgentRL 下执行）::

    # 只跑单元测试（mock docker CLI，无需 Docker / 网络 / LLM）
    python -m pytest test/test_base_agent.py -v

    # 真实 Docker 集成测试（需要 Docker daemon）
    python -m pytest test/test_base_agent.py -v --run-integration
"""

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

# 静默 mini-swe-agent 启动横幅（必须在 import minisweagent 之前设置）
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

# 保证 ``agent`` 包可导入（无论从项目根还是 test/ 目录启动）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import resolve_test_image  # noqa: E402
from agent.base_agent import (  # noqa: E402
    AttachContainerConfig,
    AttachContainerEnvironment,
    RepoAgent,
)
from minisweagent.exceptions import Submitted  # noqa: E402
from minisweagent.models.test_models import DeterministicModel, make_output  # noqa: E402

MAGIC = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


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

    通过 ``monkeypatch.setattr("minisweagent.environments.docker.subprocess.run", fake)``
    注入（subprocess 模块是全局共享的，agent.base_agent 里的 docker inspect
    校验也会被同一 fake 拦截）。
    """

    def __init__(self):
        self.calls: list[tuple[list, dict]] = []
        self.inspect_state = "true"     # docker inspect 输出：true/false
        self.exec_output = "ok"         # docker exec 的 stdout
        self.exec_rc = 0                # docker exec 的返回码

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((cmd, kwargs))
        joined = " ".join(cmd)
        if "inspect" in joined:
            return FakeResult(stdout=self.inspect_state)
        if "exec" in joined:
            if MAGIC in joined:
                return FakeResult(stdout=f"{MAGIC}\n", returncode=0)
            return FakeResult(stdout=self.exec_output, returncode=self.exec_rc)
        return FakeResult(returncode=0)

    # ---- 断言辅助 ----
    def count(self, *markers: str) -> int:
        return sum(1 for cmd, _ in self.calls if all(m in " ".join(cmd) for m in markers))

    def count_op(self, op: str) -> int:
        return sum(
            1 for cmd, _ in self.calls
            if len(cmd) > 1 and cmd[0] == "docker" and cmd[1] == op
        )


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr("minisweagent.environments.docker.subprocess.run", fake)
    return fake


# ----------------------------------------------------------------------
# AttachContainerEnvironment：挂到指定容器，不新建 / 不删除
# ----------------------------------------------------------------------


class TestAttachContainerEnvironment:
    def test_executes_in_given_container_without_creating(self, fake_docker):
        env = AttachContainerEnvironment(container_name="cont-abc", cwd="/repo")
        out = env.execute({"command": "ls -la"})
        assert env.container_id == "cont-abc"
        assert fake_docker.count_op("run") == 0          # 绝不 docker run
        assert fake_docker.count("cont-abc", "ls -la") >= 1
        assert fake_docker.count("-w", "/repo") >= 1
        # 与 mini-swe-agent 环境契约一致
        assert set(out) >= {"output", "returncode", "exception_info"}

    def test_requires_running_container(self, fake_docker):
        fake_docker.inspect_state = "false"
        with pytest.raises(RuntimeError, match="not running"):
            AttachContainerEnvironment(container_name="cont-stop")

    def test_requires_container_name(self, fake_docker):
        with pytest.raises(ValueError, match="container_name"):
            AttachContainerEnvironment()

    def test_cleanup_is_noop(self, fake_docker):
        env = AttachContainerEnvironment(container_name="cont-abc")
        env.cleanup()
        del env  # __del__ 也会调 cleanup
        assert fake_docker.count_op("rm") == 0
        assert fake_docker.count("stop") == 0

    def test_magic_string_submission(self, fake_docker):
        env = AttachContainerEnvironment(container_name="cont-abc")
        with pytest.raises(Submitted) as ei:
            env.execute({"command": f"echo {MAGIC}"})
        exit_msg = ei.value.messages[0]
        assert exit_msg["role"] == "exit"
        assert exit_msg["extra"]["exit_status"] == "Submitted"

    def test_config_defaults(self):
        cfg = AttachContainerConfig(container_name="c")
        assert cfg.cwd == "/repo"        # 与 init_env 工作目录约定一致
        assert cfg.interpreter == ["bash", "-lc"]


# ----------------------------------------------------------------------
# RepoAgent：先获取容器 -> 运行 -> 释放（mock docker + 确定性模型）
# ----------------------------------------------------------------------


class TestRepoAgentLifecycle:
    def test_acquire_run_release_order(self, fake_docker, monkeypatch):
        events = []
        monkeypatch.setattr(
            "agent.base_agent.get_env",
            lambda repo, commit=None, patch=None, manager=None: events.append(
                ("acquire", repo, commit)
            ) or "cont-1",
        )
        monkeypatch.setattr(
            "agent.base_agent.release_env",
            lambda container, manager=None: events.append(("release", container)),
        )
        model = DeterministicModel(
            outputs=[
                make_output("let me look", [{"command": "ls"}]),
                make_output("done", [{"command": f"echo {MAGIC}"}]),
            ]
        )
        agent = RepoAgent(
            "django/django", "abc123", task="fix bug",
            model=model, config_file="mini.yaml",
        )
        result = agent.run()

        assert events == [("acquire", "django/django", "abc123"), ("release", "cont-1")]
        assert result["exit_status"] == "Submitted"
        assert result["container"] == "cont-1"
        assert result["n_calls"] == 2
        roles = [m["role"] for m in result["trajectory"]["messages"]]
        assert roles[0] == "system" and roles[-1] == "exit"
        assert fake_docker.count_op("run") == 0  # 容器由 init_env 提供，不是本 agent 创建

    def test_releases_even_on_error(self, fake_docker, monkeypatch):
        events = []
        monkeypatch.setattr(
            "agent.base_agent.get_env",
            lambda repo, commit=None, patch=None, manager=None: events.append(
                ("acquire", repo)
            ) or "cont-1",
        )
        monkeypatch.setattr(
            "agent.base_agent.release_env",
            lambda container, manager=None: events.append(("release", container)),
        )
        model = DeterministicModel(outputs=[make_output("only one turn", [{"command": "ls"}])])
        agent = RepoAgent("django/django", task="x", model=model, config_file="mini.yaml")
        with pytest.raises(IndexError):  # 模型输出耗尽 -> agent.run 抛异常
            agent.run()
        assert events == [("acquire", "django/django"), ("release", "cont-1")]

    def test_preprovided_container_not_released(self, fake_docker, monkeypatch):
        monkeypatch.setattr("agent.base_agent.get_env", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not acquire")))
        monkeypatch.setattr("agent.base_agent.release_env", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not release")))
        model = DeterministicModel(outputs=[make_output("done", [{"command": f"echo {MAGIC}"}])])
        agent = RepoAgent(
            "django/django", container="pre-1", task="x",
            model=model, config_file="mini.yaml",
        )
        result = agent.run()
        assert result["container"] == "pre-1"
        assert result["exit_status"] == "Submitted"

    def test_keep_container_skips_release(self, fake_docker, monkeypatch):
        events = []
        monkeypatch.setattr(
            "agent.base_agent.get_env",
            lambda repo, commit=None, patch=None, manager=None: events.append("acquire") or "cont-1",
        )
        monkeypatch.setattr(
            "agent.base_agent.release_env",
            lambda container, manager=None: events.append("release"),
        )
        model = DeterministicModel(outputs=[make_output("done", [{"command": f"echo {MAGIC}"}])])
        agent = RepoAgent(
            "django/django", task="x", model=model,
            config_file="mini.yaml", keep_container=True,
        )
        result = agent.run()
        assert events == ["acquire"]          # 不释放
        assert result["exit_status"] == "Submitted"

    def test_default_prompt_from_mini_yaml(self, fake_docker, monkeypatch):
        """默认配置复用 mini-swe-agent 自带 mini.yaml 的提示词。"""
        monkeypatch.setattr(
            "agent.base_agent.get_env",
            lambda repo, commit=None, patch=None, manager=None: "cont-1",
        )
        monkeypatch.setattr("agent.base_agent.release_env", lambda *a, **k: None)
        model = DeterministicModel(outputs=[make_output("done", [{"command": f"echo {MAGIC}"}])])
        agent = RepoAgent("django/django", task="x", model=model)  # 默认 config_file=mini.yaml
        result = agent.run()
        traj = result["trajectory"]["messages"]
        system_msg = next(m for m in traj if m["role"] == "system")
        assert "helpful assistant" in system_msg["content"]
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in traj[1]["content"]


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
def live_container():
    """启动一个真实容器（项目预构建镜像，工作目录 /repo），结束后删除。"""
    image = resolve_test_image()
    name = f"base-agent-test-{uuid.uuid4().hex[:6]}"
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "--workdir", "/repo",
         image, "sleep", "infinity"],
        check=True, capture_output=True, timeout=300,
    )
    yield name
    subprocess.run(
        ["docker", "rm", "-f", name], capture_output=True, timeout=60, check=False,
    )


@_NEED_DOCKER
@pytest.mark.integration
class TestIntegration:
    def test_attach_env_in_real_container(self, live_container):
        env = AttachContainerEnvironment(container_name=live_container, cwd="/")
        out = env.execute({"command": "echo hello-from-agent"})
        assert out["returncode"] == 0
        assert "hello-from-agent" in out["output"]

    def test_repo_agent_end_to_end_in_real_container(self, live_container):
        model = DeterministicModel(
            outputs=[
                make_output("inspect", [{"command": "ls /"}]),
                make_output("submit", [{"command": f"echo {MAGIC}"}]),
            ]
        )
        agent = RepoAgent(
            "octocat/Hello-World",
            container=live_container,      # 外部提供容器 -> 不释放
            task="hello world",
            model=model,
            config_file="mini.yaml",
        )
        result = agent.run()
        assert result["exit_status"] == "Submitted"
        assert result["container"] == live_container
        # attach 模式：容器仍存在（生命周期归调用方/EnvManager）
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        assert live_container in out.splitlines()

    def test_repo_agent_full_lifecycle_with_init_env(self, tmp_path):
        """完整闭环（离线，不依赖 GitHub 网络）：
        zip 缓存准备本地仓库 -> init_env.get_env 获取容器 -> attach 运行
        -> release_env 删除容器。
        """
        import zipfile

        from agent.base_agent import RepoAgent
        from agent.init_env import EnvConfig, EnvManager

        # 1) 在宿主机造一个本地 git 仓库，打成 zip 缓存（走 docker cp + 解压路径，
        #    完全避开对 github.com 的克隆，保证测试确定性）。
        src = tmp_path / "src"
        src.mkdir()
        subprocess.run(["git", "init", "-q", str(src)], check=True, capture_output=True)
        (src / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(src), "add", "README.md"], check=True, capture_output=True)
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
        zip_path = cache / "octocat__Hello-World.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
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
        model = DeterministicModel(
            outputs=[
                make_output("look", [{"command": "ls /repo"}]),
                make_output("submit", [{"command": f"echo {MAGIC}"}]),
            ]
        )
        try:
            agent = RepoAgent(
                "octocat/Hello-World", commit, task="hello", model=model,
                manager=mgr, config_file="mini.yaml", keep_container=True,
            )
            result = agent.run()
            assert result["exit_status"] == "Submitted"
            assert result["container"].startswith("codeagentrl-test-")
            # keep_container=True：run() 不释放，容器还在 —— 验证 attach 到的
            # 确实是 init_env 准备的容器且已 checkout 到指定 commit
            r = subprocess.run(
                ["docker", "exec", "--workdir", "/repo", result["container"],
                 "bash", "-lc", "git rev-parse HEAD"],
                capture_output=True, text=True, timeout=60,
            )
            assert r.returncode == 0, r.stderr
            assert r.stdout.strip() == commit
            # 显式释放（等价于默认 keep_container=False 时 run() 的 finally 行为）
            mgr.release_env(result["container"])
        finally:
            # v2 不复用语义：run() 用 get_env 新建的容器由 run() 释放；这里兜底删除
            # run 异常路径可能残留的容器（release_env 幂等，已删除时仅告警）
            last = getattr(agent, "last_container", None)
            if last:
                mgr.release_env(last)
        # 释放后容器已被删除
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        assert result["container"] not in out.splitlines()
