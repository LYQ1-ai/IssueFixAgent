# Overall_PLAN —— CodeAgentRL 项目总览与阶段成果（Agent 背景知识）

> 本文档定位：**项目进展、已完成阶段工作与设计模式的概览**（功能均引用到具体代码文件），
> 供后续 Agent 开发/维护快速建立背景，**不是**逐模块开发细则——细节见
> `README.md`、`docs/mcts_engine_design.md`、`docs/reward_design.md`、
> `docs/mcts_fault_tolerance.md` 与施工文件 `docs/construction/00-05`。
> 历史规划文档原名为 PLAN.md（M0–M5 里程碑细节），其内容已并入本文与上述文档。
>
> 最后更新：2026-09-08

---

## 1. 项目一句话与研究路线

面向 **GitHub Issue 修复 Agent** 的强化学习（RL）研究项目（参考 OmegaPRM / ReARTeR / CodeScout）：

```
MCTS 自动生成过程监督标签（每步动作对/错）
  → 训练 PRM（过程奖励模型，逐步骤打分）
  → PRM 筛高质量轨迹 / 构造偏好对
  → KTO / DPO 训练 Issue Fix Agent（阶段 2）
```

阶段划分：**阶段 0** Agent 执行层（Docker 环境 + mini-swe-agent 集成，已完成）→
**阶段 1** MCTS 过程监督数据生成引擎 + PRM 训练数据采集（M0–M3 已完成，M4 PRM 训练未开始）→
**阶段 2** PRM 引导筛选 + KTO/DPO（未开始）。

## 2. 当前进度快照（2026-09-08）

| 里程碑 | 状态 | 关键数字 / 产物 |
| --- | --- | --- |
| M0 数据层 | ✅（2026-08-25/26） | 39,284 实例 / 131 repos；生成池 79 repos / 23,032 实例；`outputs/mcts/{instances,splits}.parquet` |
| M1 单 rollout | ✅（2026-08-27） | gemma-4 冒烟 reward 1.83/3.0；qwen3.5-4B demo 中位 2.25 |
| 提交协议改造 | ✅（2026-08-27） | submit_locations 结构化提交；gemma 实机 565 rollouts、提交率 1.0、标注 159 条 |
| v5 存储/驱动重构 | ✅（2026-09-03） | 三表 SQLite + 会话原子提交 + kill -9 恢复；单测 266 passed |
| M2/M3 实机批量 | ✅（2026-09-08） | **2000/2000 树 done**；rollouts 116,444、nodes 23,294、consumed 20,465、leaf 2,412 |
| M4 PRM 训练 | ⏳ 未开始 | 输入 = `outputs/batch500/state.db` 派生步级样本（见 §7） |

## 3. 已完成阶段与实现摘要（含代码文件引用）

### 3.1 阶段 0：Agent 执行层（`agent/`，不改 mini-swe-agent 源码）

| 能力 | 文件 / 类 | 要点 |
| --- | --- | --- |
| Docker 环境管理（v2：创建即用、用完即销毁） | `agent/init_env.py::EnvManager`（`get_env/release_env`）、`mcts/env.py::instance_env_params/prepare_env` | 每次 `get_env` 新建容器（uuid 命名）；zip 缓存（`CODEAGENTRL_REPO_CACHE_DIR`）+ 容器内 clone 双路径；SWE-Smith 单 commit → `commit=None` 免 checkout + `git apply` bug patch |
| Agent 运行 | `agent/base_agent.py::RepoAgent` / `AttachContainerEnvironment` | 复用 mini-swe-agent 的 DefaultAgent/轨迹/提交协议；返回 `exit_status/submission/cost/n_calls/trajectory` |
| 结构化提交 | `agent/submit_tool.py`（submit_locations schema+校验）、`agent/submit_model.py::SubmitLocationsModel`、`agent/submit_agent.py::SubmitAgent`（催收机制）、提示词 `config/mini_submit.yaml` | 提交 = 调用 submit_locations 工具（不再依赖 bash 魔法串）；撞轮上限未提交 → user 催收一次放行最后一轮；`mcts/steps.py::extract_exit` 识别 exit_status |
| 观测 | `agent/tracing.py`（Phoenix OTLP，可选） | 每次执行一个 trace，`save_trace_json` 落盘；`run_mcts --phoenix-tracing` |

### 3.2 M0 数据层（`mcts/` 数据部分 + `scripts/`）

- `mcts/instances.py`：读 parquet → `Instance` 归一化；过滤五规则；三粒度 gold（files/modules/entities，与 `mcts/reward.py` 复用同一解析口径）。CLI `python -m mcts.instances build`。
- `mcts/splits.py`：三层划分（60% 生成池 + PRM train/dev/test=80/10/10，repo 级防泄漏），seed=42 可复现。CLI `python -m mcts.splits build`。
- `mcts/config.py` / `mcts/config.yaml`：全部超参（N=5、θ、QU 系数、并发、预算、提交/奖励配置）。
- 脚本：`scripts/preprocess_data.sh`（端到端）、`scripts/batch_repo_pull.py`（131 仓库 zip 缓存预热，落 `/media/shared_e/lyq/repos`）。
- 测试：`test/test_{instances,splits,env}.py`、`test/test_init_env.py::TestNoCheckoutWhenCommitNone`。

### 3.3 M1–M3：MCTS 数据生成引擎（`mcts/` 引擎部分）

演进：v2（2026-08-27，任务驱动 + 四表快照持久化）→ **v5（2026-09-03，三表 + 会话原子提交）**。
设计总纲见 `docs/mcts_engine_design.md`（v2）与 `docs/construction/00-05`（v5 实施规格）。

| 模块 | 职责（关键类/函数） |
| --- | --- |
| `mcts/steps.py` | `Step`；轨迹→步序列 `split_steps`；前缀消息 `messages_for_prefix/steps_to_messages`；**内容寻址 `prefix_node_key`**（空前缀="root"，否则 sha1 前缀序列化） |
| `mcts/node.py` | `MCTSNode`（rollouts/visited/mc/`gated`）；QU：`compute_q_value`（α^(1−MC)·β^(len/max_len)）、`compute_u_value`；`select_best_node`（历史纯函数；v5 驱动内 `TreeDriver._select_candidate` 同口径实现） |
| `mcts/locate.py` | `locate_error`：对选中 rollout 续跑步二分，probe=父前缀+confirmed+左半；MC∈{1→停, (0,1)→向右并入 confirmed/expanded, 0→记 leaf 并向左收缩}；返回 (expanded, leaves) |
| `mcts/replay.py` | `ReplayRunner.run_free/run_probe`：确定性回放前缀 + 自由续跑（drift 容忍/硬失败开关） |
| `mcts/executor.py` | `AgentRolloutExecutor.run`：容器创建→回放/自由跑→结构化 reward→销毁（单 rollout 原子） |
| `mcts/reward.py` | `reward_from_trajectory_exit`（结构化 locations F1，提交才计分）/ `locations_localization_f1*`（layered）；`patch_localization_f1` 保留兼容；分层设计见 `docs/reward_design.md` |
| `mcts/tasks.py` | 并发/驱动：`RolloutTask`（最小调度单元）→ `TaskQueue`（asyncio 优先级+去重）→ `Worker`（重试+信号量）→ `Budget`（熔断/部分提交）→ `EnvFactory`（容器创建节流）→ `TreeDriver`（单树会话流）→ `MCTSPipeline`（多树并发+tqdm 进度条） |
| `mcts/store.py` | **v5 StateStore**：三表 `tree_instances/nodes/rollouts`；`commit_session`（会话原子提交）、`load_tree_state`（整树载入）、`upsert_rollout`（成功即落，不覆盖消费账本）、旧库检测（`LegacySchemaError` → `--fresh`） |
| `mcts/report.py` | `build_report/write_report`：tree_summaries + leaf 派生统计 → `rollout_report.{json,md}` |
| `mcts/run_mcts.py` | CLI：`--sample/--instances/--resume/--dry-run/--fresh/--max-rollouts*` 等 |
| `scripts/run_rollout.sh` | 一键：数据自检→抽样→Rollout→报告（包装 run_mcts）；`scripts/run_sglang_qwen4b.sh` 起本地 sglang |

测试：`mcts/tests/`（FakeExecutor 离线，ReARTeR docs/01 §11 数值示例端到端；含 v5 会话/恢复/孤儿用例），全量 **266 passed**（+ 根目录 `test/` 数据层 128 passed）。

## 4. 核心设计模式（背景知识 · 与代码对应）

### 4.1 步（Step）与轨迹
一步 = agent 一次 `step()`（一次 model query + 工具执行），对应 assistant 消息 + observation。
`mcts/steps.py::split_steps(trajectory["messages"])` 切分；`messages_for_prefix(messages,k)` 取前 k 步原始消息（probe 续跑输入）。

### 4.2 内容寻址（节点去重 / 复用 / 恢复的根基）
节点 = `(instance, 前缀步序列)`；`node_key = prefix_node_key(prefix_steps)`（sha1；空前缀="root"）。
- 同一前缀全局只有一份节点 → locate 探针/跨会话/跨 resume 自动复用 rollout，**不重复花 LLM**；
- DB 中 `nodes(instance_id,node_key)` PK、`rollouts` 行带 node_key → 按前缀查已算结果即复用；
- 前提：前缀序列化跨运行稳定（`Step.to_json` 确定性）。

### 4.3 MCTS 树逻辑（ReARTeR 移植 + agent 域适配）
- 根节点 N=5 次 rollout → `MC = 正确数/N` → 门控 `0<MC<1` 才进入 select/locate；
- **select（rollout 级）**：在池（root + 已入池 0<mc<1 节点）上对"未消费 rollout"取 QU 全局最大；
  消费账本 = `rollouts.is_consumed`（v5 会话结束提交，root 消费同样记录）；
- **locate（首个错误步二分）**：`locate.py::locate_error` 三分支（见 3.3 表），probe 结算后角色由 MC 派生：
  - `leaf（负样本） = 非 root ∧ mc==0`（前缀末 ≈ 首个错误步）；
  - `0<mc<1` = expanded（`nodes.in_pool=1`，PRM 连续标签来源）；
  - `mc==1` 全对探针仅留存复用；
- 轮数上限 `n_rounds`（非 root 消费轮，cap=20）。

### 4.4 会话原子提交与崩溃恢复（v5 最核心的模式）
**"事实随时落、判定会话结束时单事务落"**：
- 事实：节点行创建即落（`store.ensure_node`）、rollout 成功即落（`store.upsert_rollout`）、结算置 ready（`set_node_ready`）；
- 判定：一次 (select+locate) 会话成功结束 → `store.commit_session` 单事务提交
  `is_consumed=1 / visits / n_rounds / expanded.in_pool`；
- 崩溃语义：中途崩溃 = 判定未提交 = 该 (node, rollout) 未消费 → resume 重选重做，
  probe 结果内容寻址复用、只补在飞缺槽——**无需保存二分中间状态**；
- 失败不落库（缺槽 = 补跑信号）；resume 从 DB 整树载入（`TreeDriver.run` → `load_tree_state`），
  树状态 `done/failed` 跳过、`running/budget_exhausted/not_started` 续跑；
- 实现位置：`mcts/tasks.py::TreeDriver`（`run/_ensure_rollouts/_select_candidate/process_annotations/_locate_session`）+ `mcts/store.py::StateStore.commit_session/load_tree_state`。

### 4.5 高并发与可观测
- `MCTSPipeline.run`：每实例一个 `TreeDriver` 并发（asyncio），树间无上限；每节点 N 次 rollout 经
  `TaskQueue`+`Worker`（全局信号量）并发执行；容器创建 `EnvFactory` 节流；
- tqdm 树进度条（`MCTSPipeline.run`，stderr）；周期日志 `Stats`（吞吐/失败/回报/LLM 调用）；
- Phoenix 可选追踪（每次 rollout 一个 trace）。

### 4.6 可靠性
三层重试（Worker 退避 → 树级重跑 1 次 → resume 缺槽补跑）；预算熔断（`Budget`，部分提交语义）；
容器用完即销毁 + 泄漏检查；DB WAL + 单写锁。

## 5. 数据与产物现状（2000 树批次，2026-09-08）

- 实例与划分：`outputs/mcts/instances.parquet`（39,284）/ `splits.parquet`（生成池 23,032）；
- 本批产物目录：`outputs/batch500/` → `state.db`（v5，~5GB；trees 2000 全 done、nodes 23,294、
  rollouts 116,444、consumed 20,465、leaf 派生 2,412 覆盖 433 实例、失败 104(0.09%)、
  提交率 0.767、avg_reward ~0.59）+ `rollout_report.{json,md}`；
- 运行日志：`/media/shared_e/lyq/logs/mcts/batch2000.log`（含 tqdm 与周期进度）；
- 已知小遗留：① 10 个节点 rollout 槽 <5（失败槽未落库，树已 done，如需补齐须强制重跑对应实例）；
  ② DB 体积随 result_json 步文本增长较快，长跑/导出前建议关注。

## 6. 环境与运行（A800 事实）

- conda `CodeAgentRL`（依赖按 `README §7` 手工锁定：openai==2.54.0 / litellm==1.97.0）；
  sglang 服务环境 `sglang`；`pip install -e . --no-deps` 已注册包（见根 `pyproject.toml`）；
- 本地 LLM：sglang qwen3.5-4B `http://localhost:30000/v1`（GPU1，`scripts/run_sglang_qwen4b.sh`）；
  vLLM gemma-4 `http://localhost:8010/v1`（GPU0）备选；
- 仓库 zip 缓存 `/media/shared_e/lyq/repos`（131 个，`.env` 已配 `CODEAGENTRL_REPO_CACHE_DIR`）；
- 常用命令：
  - `python -m pytest mcts/tests/ test/ -q`（离线全量）
  - `bash scripts/run_rollout.sh --trees N --seed 42 --reward-threshold 0.6 --resume [--output DIR]`
  - `python -m mcts.run_mcts --sample N --resume --fresh`（旧库备份重建）
- git：远端 `github.com/LYQ1-ai/IssueFixAgent`（ssh 别名 `lyq1-ai`），分支 `main` / `refactor/mcts-v5`；
  代码改动按模块提交于 `refactor/mcts-v5`（v5 实现），已合入 `main`。

## 7. 下一步（M4 PRM 训练）与注意

- 输入：`outputs/batch500/state.db` → 按"node = 样本"派生步级数据
  （prompt = head+前缀[:-1]、目标步 = 前缀末步、标签 = mc 连续 / leaf 末步=0；简化版链式回填：
  leaf 前缀前 len-1 步=1、末步=0）；落盘 `outputs/prm/{train,dev,test}.parquet`；
- 训练：transformers + PEFT LoRA（方案 A：步尾 `<extra_0>` logit sigmoid；或方案 B 线性头）——细则见 §"PRM 训练"（原 PLAN §3，内容保留于 git 历史 / 本文第 7 节概要）；
- 遗留提醒：输出目录若无 parquet 需先拷贝 M0 产物；`run_rollout.sh` 自动 preprocess 固定写 `outputs/mcts`（不随 `--output`）。

## 8. 历史引用说明

仓库内旧代码/文档中出现的 "PLAN §2.x / §5" 等引用，对应旧规划文档章节（M0–M5 细节），
其内容现已并入本文（总览/模式）与 `docs/construction/00-05`（v5 实施规格）及
`docs/mcts_engine_design.md`（v2 引擎设计）；阅读时按文件索引查证即可。
