# SPDX-License-Identifier: BSD-3-Clause

"""在指定 Docker 容器中运行 mini-swe-agent 的基础 Agent（不改动 mini-swe-agent 源码）。

执行流程（与 :mod:`agent.init_env` 的生命周期约定对齐）::

    container = get_env(repo, commit)              # 1. 执行前：获取已启动的容器
    env  = AttachContainerEnvironment(container)   # 2. 把环境"挂"到该容器（不新建容器）
    agent = DefaultAgent(model, env, **cfg)        # 3. 原样运行 mini-swe-agent
    release_env(container)                         # 4. 完成后：释放（删除）容器

实现要点:

- :class:`AttachContainerEnvironment` 继承 mini-swe-agent 的 ``DockerEnvironment``，
  只覆写两处：``_start_container`` 不再 ``docker run`` 新建容器，而是校验并复用
  指定容器；``cleanup`` 变为 no-op（容器生命周期归 ``EnvManager`` 所有）。
  ``execute`` / ``_check_finished`` / 魔法字符串提交 / 轨迹序列化全部继承原实现，
  因此 Agent 侧（DefaultAgent、轨迹格式、提交协议）与官方实现完全一致。
- :class:`RepoAgent` 负责编排生命周期：``run()`` 先 ``get_env``（或复用传入的
  容器），任务结束后在 ``finally`` 中 ``release_env``（成功与异常路径都释放）；
  支持共享 ``EnvManager`` 实例与预取容器两种复用方式。
- Phoenix 追踪（可选）：``RepoAgent(phoenix_tracing=...)`` 开启后，每次
  ``run()`` 期间的所有 litellm 模型调用都会上报到 Phoenix（OTLP），并在结果
  中返回本次执行的 ``trace_id``；调用方可按 ``trace_id`` 用
  ``agent.tracing.export_trace`` 从服务端导出该次执行轨迹（可作为后续
  MCTS 采样的轨迹数据保存 / 读取方式）。
- 提示词与模型配置默认复用 mini-swe-agent 自带 ``config/mini.yaml``
  （system_template / instance_template / observation_template /
  format_error_template），可用 ``config_file`` 换成其他配置文件
  （如 ``swebench.yaml``）。

命令行用法::

    python -m agent.base_agent run owner/repo --commit <sha> --task "fix ..." --model gpt-5
    python -m agent.base_agent run owner/repo --task "fix ..." --phoenix-tracing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("agent.base_agent")

# 哨兵：run() 未显式传 phoenix_tracing 时，回落到构造函数配置
_UNSET = object()

# ---------------------------------------------------------------------------
# minisweagent 导入引导：优先已安装的包；否则把仓库 src 目录加入 sys.path。
# 可通过环境变量 MSWEA_PACKAGE_SRC 指向移动后的 mini-swe-agent 目录。
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MSWEA_SRC = Path(
    os.getenv("MSWEA_PACKAGE_SRC", _PROJECT_ROOT / "ref_papers" / "mini-swe-agent" / "src")
)


def _ensure_minisweagent_importable() -> None:
    """确保 minisweagent 可导入（不修改其源码）。"""
    try:
        import minisweagent  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in (_MSWEA_SRC, _PROJECT_ROOT / "ref_papers" / "mini-swe-agent" / "src"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            logger.info("Added %s to sys.path for minisweagent", candidate)
    try:
        import minisweagent  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Could not import minisweagent. Install it (pip install -e "
            "ref_papers/mini-swe-agent) or point MSWEA_PACKAGE_SRC at its src/ dir."
        ) from e


_ensure_minisweagent_importable()

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.utils.serialize import recursive_merge  # noqa: E402

from agent.init_env import EnvManager, get_env, release_env  # noqa: E402

# ---------------------------------------------------------------------------
# 环境：在指定容器中执行（mini-swe-agent DockerEnvironment 的子类）
# ---------------------------------------------------------------------------


class AttachContainerConfig(DockerEnvironmentConfig):
    """``AttachContainerEnvironment`` 的配置：在已有容器中执行（不新建容器）。

    - ``container_name``: 目标容器名/ID（由 ``agent.init_env.get_env`` 返回）。
    - ``image``: 仅保留以兼容父类字段，attach 模式不会用它启动容器。
    - ``cwd``: 容器内工作目录（仓库根目录，默认与 ``init_env`` 的
      ``CODEAGENTRL_WORKDIR`` 一致，可用 ``MSWEA_WORKDIR`` 覆盖）。
    - ``magic_submit``: 是否保留 bash 魔法串提交（``echo
      COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` 首行即触发 ``Submitted``）。
      新流程（``submit_locations`` 工具提交）默认禁用魔法串，仅工具调用触发
      提交（PLAN §2.2 提交协议改造）；True 恢复旧行为。
    """

    container_name: str = ""
    image: str = ""
    cwd: str = os.getenv("MSWEA_WORKDIR", os.getenv("CODEAGENTRL_WORKDIR", "/repo"))
    magic_submit: bool = True


class AttachContainerEnvironment(DockerEnvironment):
    """在已由 ``agent.init_env`` 创建/复用的 Docker 容器中执行 bash。

    与 ``DockerEnvironment`` 的区别仅在于容器来源：

    - ``_start_container``：不执行 ``docker run``，而是校验 ``config.container_name``
      指定的容器存在且处于运行状态，然后直接复用（失败抛出清晰的错误）。
    - ``cleanup``：no-op —— 容器由 ``EnvManager`` 管理，Agent 结束后由
      ``release_env`` 统一删除，这里绝不能 stop/rm 掉外部所有的容器。

    其余行为（``docker exec`` 执行、魔法字符串提交检测、模板渲染、序列化）
    完全继承自 mini-swe-agent 的 ``DockerEnvironment``。
    """

    def __init__(self, *, config_class: type = AttachContainerConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _start_container(self):
        container = self.config.container_name
        if not container:
            raise ValueError(
                "AttachContainerEnvironment requires a 'container_name'; "
                "acquire one first via agent.init_env.get_env(repo, commit)"
            )
        if not self._is_running(container):
            raise RuntimeError(
                f"Container {container!r} is not running; acquire a started container "
                "first via agent.init_env.get_env(repo, commit)"
            )
        self.container_id = container
        self.logger.info("Attached to existing container %s", container)

    def _is_running(self, container: str) -> bool:
        """校验容器存在且处于运行状态（存在但停止也算不可用）。"""
        try:
            result = subprocess.run(
                [self.config.executable, "inspect", "-f", "{{.State.Running}}", container],
                capture_output=True, text=True,
                timeout=30, check=False,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:  # pragma: no cover - docker CLI 异常统一视为不可用
            return False

    def cleanup(self):
        """No-op：容器生命周期由 agent.init_env 的 EnvManager 管理。"""
        return None

    def _check_finished(self, output: dict):
        """提交检测：``magic_submit=False`` 时禁用 bash 魔法串提交（新流程：
        任务结束只由 ``submit_locations`` 工具触发，见 agent.submit_agent）。"""
        if not self.config.magic_submit:
            return None
        return super()._check_finished(output)


# ---------------------------------------------------------------------------
# 编排器：获取容器 -> 运行 mini-swe-agent -> 释放容器
# ---------------------------------------------------------------------------


class RepoAgent:
    """在指定 ``(repo, commit)`` 容器中运行 mini-swe-agent 的编排器。

    生命周期：``run()`` 开始前通过 ``get_env`` 获取（或复用）已启动容器，
    结束后（含异常路径）通过 ``release_env`` 释放。默认不修改 mini-swe-agent
    的提示词与模型配置，直接复用其自带 ``config/mini.yaml``。
    """

    def __init__(
        self,
        repo_name: str,
        commit: Optional[str] = None,
        *,
        task: Optional[str] = None,
        patch: Optional[str] = None,
        container: Optional[str] = None,
        manager: Optional[EnvManager] = None,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        model_config: Optional[dict] = None,
        env_config: Optional[dict] = None,
        agent_config: Optional[dict] = None,
        agent_class: Optional[type] = None,
        config_file: str | Path = "mini.yaml",
        keep_container: bool = False,
        output_path: Optional[str | Path] = None,
        phoenix_tracing: Optional[Union[bool, dict]] = None,
    ):
        self.repo_name = repo_name
        self.commit = commit
        self.task = task
        self.patch = patch
        # 外部预取的容器；为 None 时 run() 内部 get_env 获取（并在结束后释放）
        self.container = container
        self.manager = manager
        self.model = model  # 直接注入 model 实例（测试/自定义模型用）；None 则按配置构造
        self.model_name = model_name
        # agent 类（None = 官方 DefaultAgent；结果提交流程用
        # ``agent.submit_agent.SubmitAgent`` —— 提交方式为 submit_locations 工具）。
        # 运行时再解析模块级 DefaultAgent，保证测试 monkeypatch 生效。
        self.agent_class = agent_class
        self.keep_container = keep_container
        self.output_path = Path(output_path) if output_path is not None else None
        self.last_container: Optional[str] = None
        # Phoenix 追踪（可选）：True 用默认配置（endpoint/project 读环境变量）；
        # dict 可传 {"endpoint", "project_name", "trace_id", "rest_base_url"}。
        # 每次 run() 返回结果中的 trace_id 即该次执行的 trace，可经
        # agent.tracing.export_trace 导出执行轨迹（MCTS 采样数据来源）。
        self.phoenix_tracing = phoenix_tracing
        # 本次执行生效的 trace 导出信息（由 run() 在启用追踪时填充）
        self._trace_project: Optional[str] = None
        self._trace_base_url: Optional[str] = None

        # 配置解析：默认读 mini-swe-agent 自带 mini.yaml 的 agent/model/environment 三段，
        # 显式传入的 dict 递归覆盖。
        base = get_config_from_spec(config_file) or {}
        self.agent_cfg = recursive_merge(base.get("agent", {}), agent_config or {})
        self.env_cfg = recursive_merge(base.get("environment", {}), env_config or {})
        self.model_cfg = recursive_merge(base.get("model", {}), model_config or {})
        # agent 段中属于交互式 agent（InteractiveAgentConfig）的键，DefaultAgent 不识别
        for key in ("mode", "confirm_exit", "whitelist_actions"):
            self.agent_cfg.pop(key, None)
        # 环境段中属于环境选择器（get_environment）的键
        self.env_cfg.pop("environment_class", None)
        if self.output_path is not None:
            self.agent_cfg["output_path"] = str(self.output_path)

    def run(
        self,
        task: Optional[str] = None,
        *,
        phoenix_tracing: Optional[Union[bool, dict]] = _UNSET,
        **template_vars,
    ) -> dict:
        """执行一次任务：获取容器 -> 运行 mini-swe-agent -> 释放容器。

        Args:
            task: 任务描述；缺省用构造时的 ``task``。
            phoenix_tracing: 本次执行的追踪开关（缺省用构造时的配置）：
                ``True`` / ``dict`` 开启 Phoenix 追踪；``False`` / ``None`` 关闭。
                dict 键：``endpoint``（OTLP collector）、``project_name``、
                ``trace_id``（可选，缺省自动生成）、``rest_base_url``（导出用）。
            **template_vars: 额外透传给 mini-swe-agent 模板的变量。

        Returns:
            结果字典：``exit_status`` / ``submission`` / ``cost`` / ``n_calls`` /
            ``container`` / ``trajectory``（完整轨迹，见
            ``minisweagent.agents.default.DefaultAgent.serialize``）/
            ``trace_id``（启用 Phoenix 追踪时返回本次执行的 trace id，否则为
            ``None``；可用 ``agent.tracing.export_trace(trace_id, ...)`` 从
            phoenix 服务端导出该次执行轨迹）/ ``tracing``（trace 导出信息：
            project_name / base_url，未启用时为 ``None``）。
        """
        # Phoenix 追踪（可选）：为本次执行注入确定的 trace 上下文，使期间所有
        # litellm 调用（mini-swe-agent 的每个模型请求）归属同一 trace。
        trace_token: Any = None
        trace_id: Optional[str] = None
        cfg = phoenix_tracing if phoenix_tracing is not _UNSET else self.phoenix_tracing
        if cfg:
            try:
                from agent.tracing import (
                    end_trace_context,
                    setup_phoenix_tracing,
                    start_trace_context,
                )

                tcfg = cfg if isinstance(cfg, dict) else {}
                endpoint = tcfg.get("endpoint")
                project = tcfg.get("project_name")
                setup_phoenix_tracing(endpoint=endpoint, project_name=project)
                trace_token, trace_id = start_trace_context(trace_id=tcfg.get("trace_id"))
                self._trace_project = project or os.getenv("PHOENIX_PROJECT", "codeagentrl")
                self._trace_base_url = (
                    tcfg.get("rest_base_url")
                    or os.getenv("PHOENIX_REST_URL", "http://localhost:6006")
                ).rstrip("/")
            except Exception as e:  # 追踪失败不阻断任务执行
                logger.warning("Phoenix tracing disabled for this run (%s); continuing", e)
                trace_token, trace_id = None, None

        task = task or self.task or ""
        container = self.container
        acquired = False
        if container is None:
            container = get_env(self.repo_name, self.commit, patch=self.patch, manager=self.manager)
            acquired = True
        self.last_container = container
        try:
            env = AttachContainerEnvironment(container_name=container, **self.env_cfg)
            if self.model is None:
                model = get_model(self.model_name, config=dict(self.model_cfg))
            else:
                model = self.model
            agent = (self.agent_class or DefaultAgent)(model, env, **self.agent_cfg)
            result = agent.run(task, **template_vars)
            return {
                "exit_status": result.get("exit_status", ""),
                "submission": result.get("submission", ""),
                "cost": agent.cost,
                "n_calls": agent.n_calls,
                "container": container,
                "trace_id": trace_id,
                "tracing": (
                    {"project_name": self._trace_project, "base_url": self._trace_base_url}
                    if trace_id
                    else None
                ),
                "trajectory": agent.serialize(),
            }
        finally:
            if trace_token is not None:
                try:
                    from agent.tracing import end_trace_context

                    end_trace_context(trace_token)
                except Exception:  # pragma: no cover - 撤销失败不影响主流程
                    pass
            # 只有本方法通过 get_env 获取的容器才负责释放（成功与异常路径都释放）
            if acquired and not self.keep_container:
                release_env(container, manager=self.manager)

    # -- 上下文管理器：仅保证 run() 的生命周期语义（释放由 run 的 finally 完成） --

    def __enter__(self) -> "RepoAgent":
        return self

    def __exit__(self, *exc) -> None:
        pass


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent in a (repo, commit) docker container "
        "(acquire -> run -> release).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="acquire container, run agent, release container")
    p_run.add_argument("repo", help="owner/repo")
    p_run.add_argument("--commit", default=None, help="commit to checkout (default: HEAD)")
    p_run.add_argument("--task", default=None, help="task/problem statement")
    p_run.add_argument("--patch-file", default=None, help="optional git diff file to apply")
    p_run.add_argument("--model", default=None, help="model name (default: $MSWEA_MODEL_NAME)")
    p_run.add_argument(
        "--config", default="mini.yaml",
        help="mini-swe-agent config file name (default: mini.yaml)",
    )
    p_run.add_argument("--output", default=None, help="path to save trajectory json")
    p_run.add_argument(
        "--keep-container", action="store_true",
        help="do not release (delete) the container after the run",
    )
    p_run.add_argument(
        "--phoenix-tracing", action="store_true",
        help="enable phoenix tracing for this run (endpoint via $PHOENIX_COLLECTOR_ENDPOINT, "
        "default http://localhost:4317)",
    )
    p_run.add_argument(
        "--phoenix-project", default=None,
        help="phoenix project name (default $PHOENIX_PROJECT / codeagentrl)",
    )
    p_run.add_argument(
        "--phoenix-endpoint", default=None,
        help="phoenix OTLP collector endpoint (default http://localhost:4317)",
    )
    p_run.add_argument(
        "--phoenix-rest-url", default=None,
        help="phoenix REST base url for trace export (default $PHOENIX_REST_URL / http://localhost:6006)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    patch = None
    if args.patch_file:
        patch = Path(args.patch_file).read_text(encoding="utf-8")

    phoenix_tracing: Optional[dict] = None
    if args.phoenix_tracing:
        phoenix_tracing = {
            "project_name": args.phoenix_project,
            "endpoint": args.phoenix_endpoint,
            "rest_base_url": args.phoenix_rest_url,
        }

    agent = RepoAgent(
        args.repo,
        args.commit,
        task=args.task,
        patch=patch,
        model_name=args.model,
        config_file=args.config,
        keep_container=args.keep_container,
        output_path=args.output,
        phoenix_tracing=phoenix_tracing,
    )
    result = agent.run()
    summary = {k: v for k, v in result.items() if k != "trajectory"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.get("trace_id"):
        print(f"[trace] export with: python -c 'from agent.tracing import export_trace; "
              f"import json; t=export_trace(\"{result['trace_id']}\"); "
              f"print(json.dumps(t, ensure_ascii=False)[:500])'")
    if args.output:
        Path(args.output).write_text(
            json.dumps(result["trajectory"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AttachContainerConfig",
    "AttachContainerEnvironment",
    "RepoAgent",
    "main",
]
