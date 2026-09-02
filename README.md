# CodeAgentRL

面向 **GitHub Issue 修复（Issue Fix）Agent** 的强化学习（RL）研究项目。

**核心目的**：参考 [OmegaPRM（arXiv:2406.06592）](https://arxiv.org/abs/2406.06592) 的自动化过程监督方法，用 **MCTS（蒙特卡洛树搜索）自动生成"分步（step-level）正确性"标签**来训练一个 **Process Reward Model（PRM，过程奖励模型）**；再借助 PRM 对 Issue Fix Agent 的完整执行轨迹做**逐步骤质量评估与筛选**，收集高质量轨迹；最后通过 **KTO / DPO** 等偏好优化算法训练 Agent 策略，在不需要人工标注过程监督数据的前提下，提升 Agent 解决真实 GitHub Issue 的能力。

> 一句话概括：**MCTS 造过程监督数据 → 训练 PRM → PRM 筛高质量轨迹 → KTO/DPO 训练 Issue Fix Agent。**

---

## 1. 核心思路（整体研究路线）

整个项目分三个阶段，当前代码处于**阶段 0（Agent 执行层基建）**，主体已完成。

### 阶段 0：Agent 执行层（当前，已完成主体）

搭建"输入 `(repo, commit)` + Issue 描述 → 输出**完整 Agent 执行轨迹**"的流水线：

- `agent/init_env.py`：按 `(repo, commit)` 创建 / 缓存 / 复用 **Docker 执行环境**（容器内放入目标仓库并 checkout 指定 commit）；
- `agent/base_agent.py`：在指定容器中运行 **mini-swe-agent**（不修改其源码），完整复刻其轨迹格式、提交协议与序列化输出；
- `agent/shell_tool.py`：自主实现的**只读 shell 工具**（面向 codescout 风格只读分析 Agent，当前暂不使用）。

该层产出的**完整轨迹**（含每一步的观察与动作、最终提交结果）是后续所有阶段的数据基础。

### 阶段 1：MCTS 过程监督数据生成 + PRM 训练（规划中）

参考 OmegaPRM 的 **divide-and-conquer MCTS** 思想，把"数学解题步骤"替换为"Issue Fix Agent 的每一步动作"：

1. 以 Agent 轨迹的某个前缀为根节点展开搜索树，**节点 = Agent 的决策步骤**；
2. 通过 rollout 到终态（跑测试 / 对比 gold patch）得到**终态回报**（是否修复成功）；
3. 用**二分查找**快速定位"第一个错误步骤"，自动为每个中间步骤打上正确 / 错误标签；
4. 用收集到的分步标签训练 **PRM**，使其能对任意中间步骤给出质量分数（正确概率）。

### 阶段 2：PRM 引导的高质量轨迹采集 + KTO/DPO 训练（规划中）

1. 用训练好的 PRM 对海量 Agent 轨迹**逐步骤打分**，筛选出高质量轨迹 / 构造偏好对（chosen / rejected）；
2. 通过 **KTO / DPO** 对 Agent 策略做偏好优化训练，使 Agent 学会"在正确的步骤上前进"；
3. 在 SWE-bench 等基准上评估修复率提升。

---

## 2. 项目结构

```
CodeAgentRL/
├── main.py                      # PyCharm 默认示例脚本（占位，无实际功能）
├── README.md                    # 本文件
├── .env_template                # 全部环境变量模板与说明（复制为 .env 使用）
├── agent/                       # 核心代码：Agent 执行层
│   ├── __init__.py              # 包导出（EnvManager / RepoAgent / ShellTool / tracing 等）
│   ├── init_env.py              # 执行环境（Docker 容器）管理器：EnvManager
│   ├── base_agent.py            # 在指定容器中运行 mini-swe-agent：RepoAgent（含 Phoenix 追踪）
│   ├── tracing.py               # Phoenix 追踪接入：trace_id 注入 / 导出（可选，MCTS 数据来源）
│   ├── shell_tool.py            # 只读 shell 工具（暂不使用）
│   └── Dockerfile               # 通用 Agent 基础镜像（codeagentrl-agent:ubuntu24）
├── mcts/                        # 阶段 1 数据生成引擎（M0 数据预处理已完成）
│   ├── config.py / config.yaml  # 全部超参（数据/过滤/gold/split/env + M1–M5 占位）
│   ├── instances.py             # 数据读取 + 过滤 + 三粒度 gold 抽取（迁移 codescout 口径）
│   ├── splits.py                # 三层划分：60% 生成池 + repo 级 80/10/10（防泄漏）
│   └── env.py                   # 环境准备：无 commit 时不切换 / git apply patch
├── scripts/
│   ├── demo.py                  # 最小启动示例：让 agent 在仓库容器里生成项目概述
│   ├── run_demo.sh              # 实际验证过的 demo 运行命令（A800 本地 vllm）
│   ├── preprocess_data.sh       # 端到端数据预处理：原始 parquet → instances/splits（仓库缓存可选）
│   ├── batch_repo_pull.py       # 批量拉取数据集全部仓库到本地 zip 缓存（rollout 网络预热）
│   └── extract_pdf_text.mjs     # 论文 PDF 文本提取脚本
├── test/                        # pytest 测试（单元测试 mock docker，无需 Docker）
│   ├── conftest.py              # 集成测试标记 / 测试镜像解析 / github 连通性探测
│   ├── test_init_env.py
│   ├── test_base_agent.py
│   ├── test_shell_tool.py
│   ├── test_tracing.py          # Phoenix 追踪：trace 上下文注入 / REST 导出 / RepoAgent 集成
│   ├── test_instances.py        # M0 数据层：读取 / 过滤 / gold（含真实数据数量级对照）
│   ├── test_splits.py           # M0 三层划分：可复现 / 无 repo 泄漏
│   └── test_env.py              # M0 环境准备：SWE-Smith 无 commit 不切换
├── ref_papers/                  # 参考资料与参照实现（见 §7）
│   ├── OmegaPRM/                # OmegaPRM 实现（arXiv:2406.06592）
│   ├── mcts/                    # MCTS 经典实现 + 《MCTS 完整流程详解》
│   ├── mini-swe-agent/          # Agent 脚手架（本项目基座）
│   ├── codescout/               # CodeScout：代码搜索 Agent 的 RL 训练
│   ├── ReARTeR/                 # ReARTeR 资料（含 PRM_Data / FlashRAG / LLaMA-Factory / trl）
│   ├── AgentPro.pdf             # 相关论文 PDF
│   └── CodeScout.pdf            # CodeScout 论文 PDF
└── outputs/                     # Agent 轨迹输出目录（demo 轨迹默认落在这里）
```

## 3. 核心模块说明

### 3.1 `agent/init_env.py` —— 执行环境（Docker 容器）工厂

负责 `(repo, commit) → 就绪容器` 的全生命周期，**v2 语义：创建即用、用完即销毁，不复用、无缓存**：

- **创建**：`docker run -d <image> sleep infinity` 启动常驻容器，工作目录 `/repo`；容器内缺 git / unzip / ripgrep 时自动 apt 安装；
- **仓库放入容器**（两条路径，最终状态一致）：
  - **zip 缓存路径**（推荐）：宿主机 `git clone` 完整仓库（含 `.git` 历史）打成 `<owner>__<repo>.zip` 存入缓存目录 → `docker cp` 进容器 → 解压 → `git checkout <commit>`；
  - **容器内 clone 路径**（未配置缓存目录时的回退）：容器内 `git clone https://github.com/<repo>.git` → checkout；
  - 可选 `patch`（git diff 文本）在 checkout 后 `git apply` 应用（对齐 codescout 对 SWE-Smith 数据的处理）；
- **每次 `get_env` 都新建容器**（容器名含 uuid，天然互不冲突，不做缓存/复用/重启校验）；
  **`release_env` 幂等删除**（`docker rm -f`）；并发创建数量由调用方（`mcts.tasks.EnvFactory`）节流；
- 并发 rollout（MCTS 引擎）场景：每个 rollout 一个全新容器，完成后自动销毁，无状态串扰。

环境准备方式对齐 `ref_papers/codescout` 的 rollout 环境准备：zip 缓存、ripgrep 安装、SWE-Smith 快照仓库（`commit=None` 时保持 HEAD）。

CLI：`python -m agent.init_env {get|release|cache}`（get 打印新容器名，cache 预生成 zip 缓存）。

### 3.2 `agent/base_agent.py` —— 在指定容器中运行 mini-swe-agent

- **`AttachContainerEnvironment`**：继承 mini-swe-agent 的 `DockerEnvironment`，只覆写两处 —— `_start_container`（**不新建容器**，校验并复用 `container_name` 指定的容器）、`cleanup`（no-op，容器生命周期归 `EnvManager` 所有）。`execute` / 魔法字符串提交检测 / 轨迹序列化全部继承原实现，因此 **Agent 侧（DefaultAgent、轨迹格式、提交协议）与官方实现完全一致**；
- **`RepoAgent`**：编排器。`run()` = `get_env` 获取（或复用传入的）容器 → 运行 `DefaultAgent` → `finally` 中 `release_env`（成功与异常路径都释放）。返回 `exit_status / submission / cost / n_calls / container / trajectory`（完整轨迹，见 `DefaultAgent.serialize()`）；
- 提示词与模型配置默认复用 mini-swe-agent 自带 `config/mini.yaml`，可用 `config_file` 切换（如 `swebench.yaml`）；
- 支持注入 `model` 实例（测试 / 自定义模型）、`model_name`（或 `$MSWEA_MODEL_NAME`）、共享 `EnvManager`、外部预取容器（`container=`）、`keep_container` 调试保留。

CLI：`python -m agent.base_agent run owner/repo --commit <sha> --task "fix ..." --model gpt-5 --output traj.json`。

### 3.3 `agent/shell_tool.py` —— 只读 shell 工具（暂不使用）

面向 codescout 风格**只读分析 Agent** 的 bash 工具（OpenAI function-calling schema，仅一个 `command` 参数）：

- **命令级只读校验**：硬性拦截（`chmod` / `pip install` / git 写操作 / `apt` 等）；文件变更命令（`rm/mv/cp/tee` 等）只允许写 `/tmp`；重定向目标检查；可选严格只读白名单（`MSWEA_READONLY_ALLOWLIST_ONLY`）；
- 在 `init_env` 提供的容器内 `docker exec` 执行，**无状态**（每次调用独立，无持久 shell 会话）；
- **环境管理完全委托 init_env**：`ShellTool(container=...)` 构造时绑定容器，本身不创建 / 挂载 / 删除任何容器。

**当前状态**：暂不使用（Agent 走 mini-swe-agent 自带 bash 工具）；设计上为后续"PRM 打分用只读分析轨迹"或独立只读 Agent 预留。其 docstring 明确了批量 rollout 的设计目标（40k instances / 128 repos / 高并发）。

### 3.4 `scripts/Dockerfile` —— 通用 Agent 镜像

```bash
docker build -t codeagentrl-agent:ubuntu24 -f agent/Dockerfile .
```

基于 Ubuntu 24.04，预装 git / unzip / ripgrep / jq / yq / python3 / build-essential 等，使 `init_env._ensure_tools` 的探测全部命中、创建容器时**跳过容器内 apt 安装**——既加快容器创建，也规避 A800 这类主机上 Debian 系镜像 apt 挂起的问题（IPv6 路由问题，见 `apt_force_ipv4`）。容器只承载"被 Agent 操作的目标仓库环境"，Agent 本体（litellm / mini-swe-agent 等 Python 依赖）运行在宿主机。

### 3.5 `scripts/demo.py` —— 最小启动示例

让 Agent 在指定仓库容器中生成 `PROJECT_OVERVIEW.md` 项目概述并提交（走完整 get_env → run → release_env 流程），用于验证全链路。支持本地 vllm（`--base-url`，无需 API key）与云端模型。

## 4. 关键设计约定

- **环境生命周期**：容器归 `EnvManager` 所有，`base_agent` 只借不建、由 `release_env` 统一删除；`AttachContainerEnvironment.cleanup` 为 no-op，防止误删外部容器；
- **提交协议**：与 mini-swe-agent 一致 —— Agent 输出 `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 即视为提交，提交后不能再继续；
- **轨迹格式**：`DefaultAgent.serialize()` 输出，与官方格式完全一致，是后续 PRM 数据 / DPO 训练数据的直接来源；
- **批量 rollout 设计**：容器**创建即用、用完即销毁**（每 rollout 一个全新容器，`EnvFactory` 限并发创建数）+ 并发限流（`MSWEA_MAX_CONCURRENT` / `mcts.max_concurrency`），为大规模并行采集轨迹做准备；
- **模型路由**：经 mini-swe-agent 的 litellm 路由，模型名需带 provider 前缀（如 `openai/gemma-4`，裸 `gemma-4` 会报 "LLM Provider NOT provided"）；本地服务建议 `cost_tracking: ignore_errors` 关闭成本核算；
- **`.env` 加载时机**：`demo.py` 必须在 `import agent` **之前** `load_dotenv()`，否则模块级默认 `EnvManager` 会固化默认镜像（`python:3.11-slim`）。

## 5. 环境配置

复制 `.env_template` 为 `.env`（或启动命令前内联设置）即可运行；所有变量都有代码内默认值。核心变量：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `CODEAGENTRL_IMAGE` | `python:3.11-slim` | 环境容器基础镜像（推荐 `codeagentrl-agent:ubuntu24`） |
| `CODEAGENTRL_WORKDIR` | `/repo` | 容器内仓库路径 / 工作目录（与 `MSWEA_WORKDIR` 保持一致） |
| `CODEAGENTRL_REPO_CACHE_DIR` | （空） | 仓库 zip 缓存目录；配置后跳过容器内 git clone |
| `CODEAGENTRL_INSTALL_RIPGREP` | `true` | 是否在容器内安装 ripgrep（codescout 风格 agent 的搜索依赖） |
| `CODEAGENTRL_APT_FORCE_IPV4` | `true` | 强制 apt 走 IPv4（规避 IPv6 路由导致 apt 挂起） |
| `CODEAGENTRL_NAME_PREFIX` | `codeagentrl` | 容器名前缀（形如 `codeagentrl-<repo>-<commit8>-<随机6位>`） |
| `MSWEA_MODEL_NAME` | （空） | mini-swe-agent 默认模型名（也可 `RepoAgent(model_name=...)` 传入） |
| `MSWEA_PACKAGE_SRC` | `ref_papers/mini-swe-agent/src` | minisweagent 包源码目录（未 pip 安装时加入 sys.path） |
| `MSWEA_TIMEOUT` | `30` | shell_tool 单条命令超时（秒） |
| `MSWEA_MAX_OUTPUT_CHARS` | `10000` | shell_tool 输出截断长度（head+tail，防上下文爆炸） |
| `MSWEA_MAX_CONCURRENT` | `4` | 同一容器最大并发 docker exec（0 = 不限） |
| `CODEAGENTRL_TEST_IMAGE` | 自动探测 | 集成测试用镜像（默认优先 `codeagentrl-agent:ubuntu24`） |

完整清单与逐项说明见 `.env_template`。

## 6. 快速开始（A800）

```bash
# Python 执行环境（conda）
conda activate CodeAgentRL

# 1) 构建通用 Agent 镜像（可选但推荐）
docker build -t codeagentrl-agent:ubuntu24 -f agent/Dockerfile .

# 2)（推荐）预生成仓库 zip 缓存，跳过容器内 git clone（github 网络不稳时）
CODEAGENTRL_REPO_CACHE_DIR=/data/repo_cache \
  python -m agent.init_env cache octocat/Hello-World

# 3) 运行 demo：A800 本地 vllm（8010 端口 gemma-4，无需 API key）
CODEAGENTRL_REPO_CACHE_DIR=/data/repo_cache \
  python scripts/demo.py octocat/Hello-World \
    --model openai/gemma-4 --base-url http://localhost:8010/v1 --keep-container
#    云端模型示例：python scripts/demo.py django/django --commit <sha> --model gpt-5

# 4) 其他 CLI
python -m agent.init_env get django/django --commit 6da8c1f0c46a8a0f1b8a   # 打印容器名
python -m agent.init_env release <container-name>
python -m agent.base_agent run django/django --commit <sha> \
  --task "fix ..." --model gpt-5 --output traj.json
```

## 7. Phoenix 观测平台（可选，Agent 轨迹可视化）

**Phoenix**（`arize-phoenix`）是 LLM / Agent 可观测平台：服务端 = 数据收集器 + Web UI，应用进程通过 OpenTelemetry OTLP 协议上报 span，可可视化 Issue Fix Agent 的模型调用与工具调用轨迹（调试、PRM 数据采集分析用）。

### 架构：为什么用独立环境

`arize-phoenix` 全量包依赖的 `pydantic-ai-slim[openai]`（2.34.x）要求 `openai>=3.0.0`，而 `litellm`（mini-swe-agent 的模型路由层，当前 1.97.0、最新 1.98.0 亦然）钉死 `openai>=2.20,<3.0.0` —— **二者无法在同一环境共存**。因此采用独立环境方案：

- **`phoenix` conda 环境**：运行 Phoenix 服务端（已建好：`arize-phoenix==19.18.0`，Python 3.12）；
- **`CodeAgentRL` 环境**：只装轻量 OTLP 客户端（已装 `arize-phoenix-otel==0.17.1` + `openinference-instrumentation-litellm==0.1.39`），Agent 进程将 litellm 调用轨迹上报服务端，不引入 openai 3.x 冲突。

### 服务端（phoenix 环境）

```bash
conda activate phoenix
phoenix serve
```

启动后各入口（实测输出）：

| 入口 | 地址 |
| --- | --- |
| Web UI | `http://localhost:6006` |
| REST / GraphQL API | `http://localhost:6006/v1` / `/graphql` |
| OTLP gRPC | `http://localhost:4317` |
| OTLP HTTP | `http://localhost:6006/v1/traces` |

默认**不启用认证**（本机使用无需 API key）。A800 上访问 UI 需 SSH 端口转发：`ssh -L 6006:localhost:6006 A800`。

### 客户端接入（CodeAgentRL 环境，依赖已装好）

```python
# 在 Agent 进程启动、调用模型之前执行一次
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:4317"  # 指向服务端
os.environ["PHOENIX_PROJECT"] = "codeagentrl"

from phoenix.otel import register
from openinference.instrumentation.litellm import LiteLLMInstrumentor

tracer_provider = register()  # endpoint 缺省读 PHOENIX_COLLECTOR_ENDPOINT，再缺省 gRPC localhost:4317
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
# 之后 RepoAgent.run() 中所有 litellm 模型调用自动上报，可在 Phoenix UI 按 project 查看
```

`register(auto_instrument=True)` 可自动 instrument 所有已安装的 OpenInference 库（本项目已装 litellm 插桩）。

### 与 mini-swe-agent 的兼容性（结论：官方方式可直接套用）

**结论：可以直接参考官方 litellm-tracing 方式接入，无需修改 mini-swe-agent 源码。** 依据（已核对源码与已装包）：

- mini-swe-agent 的 `get_model` 默认返回 `LitellmModel`（`minisweagent/models/__init__.py` 的 `get_model_class`），其 `_query()` 调用 **`litellm.completion()`**（同步，带 bash 工具 schema）；
- `openinference-instrumentation-litellm==0.1.39` 的 `LiteLLMInstrumentor._instrument` 会对 `litellm` 模块做 monkey-patch，包装 `completion` / `acompletion` / `responses` / `aresponses` / `completion_with_retries` / `embedding` 等函数（含 `litellm.anthropic.messages.create`）；
- 插桩发生在 litellm **模块对象**上，与 import 顺序无关——只要在**首次模型调用之前**执行一次 `register()` + `instrument()` 即可；
- 每个 Agent 决策步（一次 completion 调用）产生一个 LLM span，含输入消息、工具 schema、输出、token 数与成本，适合轨迹回放与调试；
- ⚠️ 多进程批量 rollout 时，**每个 worker 进程**都需自行 `register()` + `instrument()`（插桩是进程内状态）。

### 已落地实现（agent/tracing.py + RepoAgent 集成）

按上述结论已在项目中落地，无需手工写 register/instrument：

**1. RepoAgent 可选项开启追踪**（`agent/base_agent.py`）：

```python
from agent import RepoAgent

agent = RepoAgent(
    "owner/repo", "commit",
    task="fix ...",
    model_name="gpt-5",
    phoenix_tracing=True,   # 或 dict: {"endpoint", "project_name", "trace_id", "rest_base_url"}
)
result = agent.run()
trace_id = result["trace_id"]   # 本次执行的 trace id（未启用时为 None）
print(result["tracing"])        # {"project_name": ..., "base_url": ...}，导出 trace 用
```

也可在 `run()` 上按次覆盖：`agent.run(task, phoenix_tracing={"trace_id": "..."})`。

**2. 每次执行 = 一个独立 trace**：`run()` 内部注入 OTel 远程父上下文（`agent.tracing.start_trace_context`），期间所有 litellm 调用（mini-swe-agent 每个模型请求）共享同一 trace_id；支持外部传入 `trace_id`（例如与 MCTS 采样节点关联）。

**3. trace 导出 = 执行轨迹的保存 / 读取方式**（`agent/tracing.py`，可作后续 MCTS 采样数据）：

```python
from agent.tracing import export_trace, save_trace_json

trace = export_trace(
    result["trace_id"],
    base_url=result["tracing"]["base_url"],      # 缺省 http://localhost:6006
    project_name=result["tracing"]["project_name"],
)
# trace["spans"] 含 name / context(trace_id, span_id) / parent_id / attributes /
#               start_time / end_time / status_code / events，
#               可按 parent_id 重建 span 树 = 一次任务执行的完整轨迹
save_trace_json(trace, "outputs/traces/<trace_id>.json")   # 落盘保存
```

`agent/tracing.py` 完整 API：`setup_phoenix_tracing`（进程内幂等注册 + 插桩 litellm）、`start_trace_context` / `end_trace_context`（trace_id 注入 / 撤销）、`export_trace`（REST `GET /v1/projects/{project}/spans?trace_id=...`，自动翻页）、`save_trace_json`。全部**惰性导入**：未启用追踪时不产生任何开销。单元测试见 `test/test_tracing.py`。

**4. CLI / demo.py**：

```bash
# base_agent CLI
python -m agent.base_agent run owner/repo --task "fix ..." \
  --phoenix-tracing \
  [--phoenix-project P --phoenix-endpoint http://localhost:4317 --phoenix-rest-url http://localhost:6006]

# scripts/demo.py（同样支持；配置了追踪即假定 `phoenix serve` 已启动，
# 不检查、不启动；后续任何一步失败直接抛异常）
python scripts/demo.py owner/repo --model gpt-5 --phoenix-tracing
```

运行结束打印 summary（含 trace_id）；demo.py 还会自动把执行轨迹导出到
`outputs/traces/<trace_id>.json`（export_trace + save_trace_json）。

> 实测记录：`register()` + `LiteLLMInstrumentor` 在 CodeAgentRL（litellm 1.97.0）中运行正常；插桩后对本地 vllm 的 `litellm.completion` 调用正常返回；trace 上下文注入 / REST 导出逻辑有单元测试覆盖。span 上报依赖 `phoenix serve` 完成启动（本机实测负载高时启动变慢，正常时可 35s 内就绪）。

### 依赖版本约束（避坑）

| 包 | 版本 | 说明 |
| --- | --- | --- |
| `openai` | `2.54.0`（锁定） | litellm 要求 `>=2.20,<3.0.0`；**勿升 3.x**，否则 mini-swe-agent 模型调用报错 |
| `litellm` | `1.97.0` | 与 openai 2.x 配套；勿与 arize-phoenix 全量包同环境 |
| `mini-swe-agent` 依赖 | `datasets` / `prompt-toolkit` / `textual` | 三者缺失时 `pip check` 报依赖冲突 |
| `pydantic-ai-slim` | CodeAgentRL 中不装 | 由 arize-phoenix 全量包引入、要求 openai>=3，是冲突根源 |

### 环境搭建与依赖修复（可复现命令）

```bash
# ---- 服务端：phoenix 独立 conda 环境（一次性） ----
conda create -n phoenix python=3.12 -y
conda activate phoenix
pip install arize-phoenix        # 实测 19.18.0
phoenix serve                    # UI :6006 / OTLP gRPC :4317

# ---- 若 CodeAgentRL 环境再次出现依赖冲突，按序修复 ----
conda activate CodeAgentRL
# 1) 移除 phoenix 全量包（其 pydantic-ai-slim[openai] 要求 openai>=3）
pip uninstall -y arize-phoenix arize-phoenix-client arize-phoenix-evals \
  arize-phoenix-otel openinference-instrumentation-openai pydantic-ai-slim
# 2) 清理孤儿包（可选，均为 phoenix 生态专用）
pip uninstall -y pydantic-graph pydantic-monty pydantic-monty-runtime
# 3) 锁定 openai 2.x（litellm 要求 >=2.20,<3.0.0）
pip install openai==2.54.0
# 4) 补齐 mini-swe-agent 依赖
pip install datasets prompt-toolkit textual
# 5) 重装轻量观测客户端（与 openai 2.x 兼容）
pip install arize-phoenix-otel openinference-instrumentation-litellm
# 6) 校验：应输出 "No broken requirements found."
pip check
```

> ⚠️ 注意：`openai`、`litellm`、`pydantic-ai-slim` 三者版本互相咬死（见上表），升级其中任意一个都可能重新引入冲突；修复后请勿再在 CodeAgentRL 内 `pip install arize-phoenix`。

## 8. 测试

```bash
python -m pytest test/ -v                     # 单元测试（mock docker CLI，无需 Docker / 网络）
python -m pytest test/ -v --run-integration   # 含真实 Docker 集成测试（需 Docker daemon + github 可达）
```

单元测试通过可编程的 docker CLI 假实现（拦截 `subprocess.run`）走完整真实逻辑，不触碰真实 Docker；集成测试经 `EnvManager.get_env` 创建真实容器执行。

## 9. 参考资料（ref_papers）

| 目录 | 说明 |
| --- | --- |
| `OmegaPRM/` | [OmegaPRM（arXiv:2406.06592）](https://arxiv.org/abs/2406.06592) 实现：以 divide-and-conquer MCTS + 二分查找自动收集过程监督数据（约 150 万条、约 1/10 采样成本），训练的 PRM 在 MATH-500 上把 best-of-N 提升到 64.4%、把 Gemini 1.5 Pro 提升到 69.4% —— 本项目的方法论核心参考 |
| `mcts/` | MCTS 经典实现（源自 int8/monte-carlo-tree-search）+《MCTS 完整流程详解.md》（选择/扩展/模拟/回溯/决策逐阶段源码级讲解） |
| `mini-swe-agent/` | mini-swe-agent：本项目 Agent 脚手架基座，不改其源码，复用其环境 / 轨迹 / 提交协议 |
| `codescout/` | [CodeScout](https://arxiv.org/abs/2603.17829)：代码搜索 Agent 的 RL 训练（只读终端 agent、rg 依赖、SWE 数据集、rollout 环境准备的参考） |
| `ReARTeR/` | ReARTeR 资料：PRM_Data、FlashRAG、LLaMA-Factory、trl 等（偏好训练 / 评估相关） |
| `AgentPro.pdf` / `CodeScout.pdf` | 相关论文 PDF |

## 10. 路线图 / TODO

- [x] 阶段 0：Agent 执行层 —— Docker 环境管理（init_env）、mini-swe-agent 集成（base_agent）、轨迹输出
- [x] 基础设施：Phoenix 观测平台独立环境部署 + litellm 轨迹上报客户端（可选，见 §7）
- [x] 阶段 1 · M0：数据预处理 —— `mcts/` 包（instances 读取/过滤/gold、splits 三层划分、env 环境准备），
  产出 `outputs/mcts/{instances,splits}.parquet` + 数据报告（39,284 实例 / 131 repos，详见 PLAN.md）
- [ ] 阶段 1：MCTS 过程监督数据生成（把 OmegaPRM 迁移到 Issue Fix 域）→ PRM 训练
- [ ] 阶段 2：PRM 引导的高质量轨迹筛选 → KTO/DPO 训练 Issue Fix Agent → SWE-bench 评估
