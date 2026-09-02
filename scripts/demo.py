#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""最简单的 base_agent 启动示例：让 agent 在指定仓库容器里完成"项目概述"。

流程（复用 ``agent/base_agent.py`` 的 :class:`RepoAgent`，不改动 mini-swe-agent）:

    1. ``agent.init_env.get_env`` 获取（或复用）一个已启动的 ``(repo, commit)`` 容器；
    2. mini-swe-agent 的 ``DefaultAgent`` 在**指定容器**里分析代码、生成概述；
    3. ``run()`` 结束后 ``release_env`` 释放容器（``--keep-container`` 可保留调试）；
    4. 打印 ``exit_status`` / ``submission``（项目概述全文），轨迹存到 ``--output``。

用法（在项目根目录执行，推荐用 CodeAgentRL conda 环境）::

    # 用 A800 本地的 vllm 模型（8010 端口已起 gemma-4 服务，无需 API key）
    python scripts/demo.py octocat/Hello-World \
        --model openai/gemma-4 --base-url http://localhost:8010/v1

    # 用云端模型（需设置 MSWEA_MODEL_NAME 或 --model，并配好 API key）
    python scripts/demo.py django/django --commit 6da8c1f0c46a8a0f1b8a --model gpt-5

    # 指定仓库 zip 缓存（跳过容器内 git clone，A800 上 github 网络不稳时推荐）
    CODEAGENTRL_REPO_CACHE_DIR=/data/repo_cache \
        python -m agent.init_env cache octocat/Hello-World      # 预生成缓存
    CODEAGENTRL_REPO_CACHE_DIR=/data/repo_cache \
        python scripts/demo.py octocat/Hello-World --model openai/gemma-4 \
        --base-url http://localhost:8010/v1

    # 启用 Phoenix 追踪（可选，见 README §7）：由外部脚本参数控制。本脚本**不检查、
    # 不启动** phoenix serve —— 配置了追踪即假定 `phoenix serve` 已运行；后续任何
    # 一步失败（如导出 trace 时服务端不可达）都会直接抛出异常，不静默吞掉。
    python scripts/demo.py octocat/Hello-World --model openai/gemma-4 \
        --base-url http://localhost:8010/v1 --phoenix-tracing
    #   可选：--phoenix-project P --phoenix-endpoint http://localhost:4317 \
    #         --phoenix-rest-url http://localhost:6006

环境变量：支持项目根目录 ``.env``（python-dotenv 自动加载，须在 import agent
之前生效），也可在启动命令前内联设置，例如 ``CODEAGENTRL_IMAGE``（基础镜像，
默认 python:3.11-slim，推荐用 agent/Dockerfile 构建的 codeagentrl-agent:ubuntu24）。
Phoenix 追踪相关：``PHOENIX_COLLECTOR_ENDPOINT`` / ``PHOENIX_PROJECT`` /
``PHOENIX_REST_URL`` / ``PHOENIX_API_KEY``（详见 ``.env_template``）。
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 保证 ``agent`` 包可导入（无论从项目根还是其它目录启动）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 必须先于 ``from agent import ...`` 加载 .env：agent.init_env 的模块级默认
# EnvManager 在 import 时就会读取 CODEAGENTRL_IMAGE 等环境变量；若先 import
# 再 load_dotenv，默认管理器会拿到默认值（python:3.11-slim 起容器并触发
# apt 安装 git/unzip/rg）。
load_dotenv()

from agent import RepoAgent  # noqa: E402

DEFAULT_TASK = """请对当前仓库做一份项目概述（用 markdown 编写，保存为 /repo/PROJECT_OVERVIEW.md），内容包括：

1. 项目定位：这个仓库是做什么的、解决什么问题
2. 目录结构：顶层目录/模块划分及各自职责
3. 核心模块：关键源码文件、类/函数入口与调用链
4. 依赖清单：主要第三方依赖及用途
5. 构建与运行方式：如何安装依赖、运行测试、启动服务

完成后用下面**这一条**命令提交（先写文件，再在同一条命令里输出，提交后不能再继续）：

    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /repo/PROJECT_OVERVIEW.md
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="启动 base_agent 在指定仓库容器中生成项目概述。",
    )
    parser.add_argument("repo", help="owner/repo，如 octocat/Hello-World")
    parser.add_argument("--commit", default=None, help="要 checkout 的 commit（默认 HEAD）")
    parser.add_argument("--task", default=DEFAULT_TASK, help="任务描述（默认：项目概述）")
    parser.add_argument("--model", default=None, help="模型名（默认读 $MSWEA_MODEL_NAME）")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI 兼容服务的 base_url（如本地 vllm http://localhost:8010/v1）",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key（默认读 $OPENAI_API_KEY；配 --base-url 访问本地服务时默认用占位符，"
             "因为 vllm 等本地服务不校验 key）",
    )
    parser.add_argument(
        "--output", default=None,
        help="轨迹保存路径（默认 outputs/demo_<repo>_<时间戳>.traj.json）",
    )
    parser.add_argument(
        "--keep-container", action="store_true",
        help="运行结束后保留容器（便于调试；默认释放删除）",
    )
    parser.add_argument(
        "--phoenix-tracing", action="store_true",
        help="enable phoenix tracing for this run (assumes `phoenix serve` is already running; "
             "OTLP endpoint via $PHOENIX_COLLECTOR_ENDPOINT, default http://localhost:4317)",
    )
    parser.add_argument(
        "--phoenix-project", default=None,
        help="phoenix project name (default $PHOENIX_PROJECT / codeagentrl)",
    )
    parser.add_argument(
        "--phoenix-endpoint", default=None,
        help="phoenix OTLP collector endpoint (default http://localhost:4317)",
    )
    parser.add_argument(
        "--phoenix-rest-url", default=None,
        help="phoenix REST base url for trace export (default $PHOENIX_REST_URL / http://localhost:6006)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Phoenix 追踪（可选）：由外部脚本参数控制。启用时**假定** `phoenix serve` 已
    # 启动（本脚本不检查、不启动）；这里提前完成 register + 插桩 litellm，依赖
    # 缺失会在此直接抛异常；后续导出 trace 等服务端操作失败同样直接抛出。
    phoenix_tracing = None
    if args.phoenix_tracing:
        from agent.tracing import export_trace, save_trace_json, setup_phoenix_tracing

        setup_phoenix_tracing(
            endpoint=args.phoenix_endpoint,
            project_name=args.phoenix_project,
        )
        phoenix_tracing = {
            "project_name": args.phoenix_project,
            "endpoint": args.phoenix_endpoint,
            "rest_base_url": args.phoenix_rest_url,
        }

    # 本地/自定义模型没有成本表，关闭成本核算报错
    model_config: dict = {"cost_tracking": "ignore_errors"}
    if args.base_url:
        model_config["model_kwargs"] = {
            "api_base": args.base_url,
            "api_key": args.api_key or "local-demo",  # vllm 等本地服务不校验 key
        }
    elif args.api_key:
        model_config["model_kwargs"] = {"api_key": args.api_key}

    repo_token = args.repo.replace("/", "__")
    commit_token = (args.commit or "head")[:8]
    output = Path(args.output or PROJECT_ROOT / "outputs" / f"demo_{repo_token}_{commit_token}_{datetime.now():%Y%m%d_%H%M%S}.traj.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[demo] 目标: {args.repo}@{args.commit or 'HEAD'}  模型: {args.model or '<$MSWEA_MODEL_NAME>'}  轨迹: {output}")
    agent = RepoAgent(
        args.repo,
        args.commit,
        task=args.task,
        model_name=args.model,
        model_config=model_config,
        keep_container=args.keep_container,
        output_path=output,
        phoenix_tracing=phoenix_tracing,
    )
    result = agent.run()

    print(f"[demo] exit_status = {result['exit_status']}")
    print(f"[demo] 模型调用 {result['n_calls']} 次, 成本 ${result['cost']:.4f}")
    print(f"[demo] 容器 = {result['container']}  轨迹已保存 = {output}")
    if result.get("trace_id"):
        trace_id = result["trace_id"]
        tracing = result["tracing"]
        print(f"[demo] trace_id = {trace_id}  项目 = {tracing['project_name']}  REST = {tracing['base_url']}")
        # 导出该次执行轨迹并落盘（MCTS 采样的数据保存方式）。服务端不可达 /
        # 返回错误时 export_trace 直接抛异常（配置了追踪即假定 serve 已启动）。
        trace = export_trace(trace_id, **tracing)
        trace_path = output.parent / "traces" / f"{trace_id}.json"
        save_trace_json(trace, trace_path)
        print(f"[demo] 执行轨迹已导出: {trace_path}（spans={len(trace['spans'])}）")
    elif args.phoenix_tracing:
        raise RuntimeError(
            "Phoenix tracing was requested but no trace_id was produced; "
            "check that `phoenix serve` is running and "
            "PHOENIX_COLLECTOR_ENDPOINT is reachable."
        )
    print("=" * 60)
    print("项目概述（submission）:")
    print(result["submission"] or "(空)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
