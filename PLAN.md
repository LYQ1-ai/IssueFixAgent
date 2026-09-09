# CodeAgentRL 阶段 1 实施计划：MCTS + Rollout 构建 PRM 训练数据

> 目标：参考 **OmegaPRM**（divide-and-conquer MCTS + 二分定位首个错误步）与 **ReARTeR**（MCTS Rollout 落地实现），
> 用 **CodeScout**（数据 / 环境 / 奖励设计）作为 Issue-Fix 域的适配蓝本，为 Issue Fix Agent（mini-swe-agent）
> 的**每步动作**自动打上过程监督标签，训练一个 **PRM（Process Reward Model）**，为阶段 2 的 PRM 引导轨迹筛选与 KTO/DPO 铺路。
>
> 本文档依据当前仓库状态（阶段 0 已完成：Docker 环境管理 `agent/init_env.py`、mini-swe-agent 集成 `agent/base_agent.py`、
> Phoenix 追踪 `agent/tracing.py`）与 `ref_papers/` 下三份参考资料拟定，分三个部分：**数据预处理 → MCTS 核心逻辑 → PRM 训练**。

---

## 📍 当前进度（更新于 2026-09-08）

| 状态 | 事项 |
| --- | --- |
| ✅ 已完成 | **规划文档 v2**：三部分结构（数据预处理 → MCTS 核心逻辑 → PRM 训练）及 M0–M5 里程碑、风险对策、参考资料对照 |
| ✅ 已完成 | **数据实测**：`swe_smith/{train,validation}.parquet` schema 与字段格式探明 —— `repo` 列 = `swesmith/{owner}__{repo}.{commit8}`（单 commit 合成仓库，本身即 GitHub 仓库地址，环境准备免 checkout）；`base_commit` 恒为 None；`use_patch` 恒为 True；`file_changes`/`target` 为三粒度 gold 定位目标 |
| ✅ 已完成 | **codescout 代码定位**：数据加载（`src/build_dataset.py`、`tests/test_single_file_localization.py` 等 `pd.read_parquet` 直读）、GT 构建（`src/rewards/file_localization/file_localization.py::multilevel_localization_f1_reward` 解析口径）、环境准备（`src/utils/instance.py::clone_instance`：clone 免 checkout + `git apply` patch）—— 均已标注到 §1、§2.2、§7 |
| ✅ 已完成 | **M0 数据层实现**（2026-08-25）：`mcts/{config.yaml,instances.py,splits.py,env.py}` —— 读取 39,287 行（train 39,187 + validation 100）→ 过滤后 39,284（仅剔除 3 条超长语句，131 repos 不变）→ 三粒度 gold 抽取（39,284 实例全有 files，38,680 有 entities）；三层划分：60% 生成池 = 79 repos / 23,032 实例，PRM train/dev/test = 63/8/8 repos（17,208/3,881/1,943 实例，同 repo 不跨 split）；环境准备：SWE-Smith `base_commit=None` → `commit` 传 `None`，**无 commit 时不切换**（跳过 `git checkout`，两条路径 zip/clone 均有单测）。产出 `outputs/mcts/{instances,splits}.parquet` + 数据报告；单测 `test/test_{instances,splits,env}.py` + `test_init_env.py::TestNoCheckoutWhenCommitNone` |
| ✅ 已完成 | **数据预处理流水线脚本化**（2026-08-26）：`scripts/preprocess_data.sh` 端到端串联（解析过滤 → 三层划分 → 可选仓库缓存，`--data-dir` 默认 `data/swe_smith`，支持 `--skip-*`/`--dry-run`/缓存参数）；`scripts/batch_repo_pull.py` 批量拉取数据集中全部仓库到本地 zip 缓存（幂等 / 并发 / 失败重试与续拉）；两步均实跑：39,287 → 39,284 实例 / 131 repos，splits 划分数字与 M0 一致（无泄漏、可复现） |
| ✅ 已完成 | **测试基建加固**（2026-08-26）：`test/conftest.py::resolve_test_image` 默认**固定** `codeagentrl-agent:ubuntu24`（移除 `python:3.11-slim` 兜底），预构建镜像缺失时 fail-fast 提示先 `docker build` —— 集成测试不再触发容器内 apt 安装（避免挂起 / 超时）；`--run-integration` 离线路径实跑通过（`test_zip_hit_local_repo_offline`） |
| ✅ 已完成 | **仓库缓存全量预热**（2026-08-26）：`batch_repo_pull` 拉取**全部 131 仓库**成功（0 失败，约 103s），zip 缓存落 `/media/shared_e/lyq/repos`（131 个 zip，含完整 `.git` 历史）；`.env` 已配置 `CODEAGENTRL_REPO_CACHE_DIR=/media/shared_e/lyq/repos`，rollout 时 `EnvManager.get_env` 直接走 docker cp + 解压的离线路径，不再依赖实时外网 |
| ✅ 已完成 | **MCTS 高并发 Rollout 引擎设计与实现**（2026-08-27，对应 M1–M3 核心）：设计文档 `docs/mcts_engine_design.md`（任务驱动：任务提交 → 优先级任务队列，多树并发 + 节点内 N 次 rollout 并发）；实现 `mcts/{steps,reward,llm,replay,node,locate,tasks,executor,run_mcts}.py` + `mcts/tasks.py`（TaskQueue / ContainerPool / Worker / TreeDriver / MCTSPipeline / 预算 / 断点续跑）+ `mcts/tests/` 56 个单测全部通过（含 ReARTeR docs/01 §11.1–11.9 数值示例端到端复现、并发语义、预算熔断、续跑复用）。详见 §2.3/§2.4 与设计文档 |
| ✅ 已完成 | **A800 实机验证（M1 冒烟 + dry-run）**（2026-08-27）：`mcts.run_mcts --dry-run` 3 实例 90 rollouts（0 失败，树搜索正常触发）；**真实 rollout** 1 实例 `pallets__markupsafe`（zip 缓存容器 + vLLM gemma-4，6 步 / 6 次 LLM 调用，终态回报 **1.83/3.0**，correct），**容器无泄漏**；测试套件在 A800 通过 |
| ✅ 已完成 | **v2 并发架构重构（评审确认后落地，2026-08-27）**：① **容器不复用** —— `agent/init_env.py::EnvManager` 改写为"创建即用、用完即销毁"（无缓存/无复用/无 repo+commit key，保留 zip 缓存与环境准备），`mcts.tasks.EnvFactory` 只限**并发创建数**（`creation_concurrency`，默认 8）不限总次数；② **并发解耦** —— 树只管提交任务，任务队列 + 全局信号量执行并回传（保持 asyncio + to_thread）；③ **SQLite 持久化** —— 新增 `mcts/store.py::StateStore`（`outputs/mcts/state.db`，WAL：rollouts/nodes/annotations/instances 四表），树与结果常驻内存、rollout 完成即落 + 树定期快照（崩溃恢复代码后置，schema 就绪）；单测扩至 62 个（新增 store 往返/并发写），A800 全量通过；**注意**：按评审决定**不加活跃树上限**（全部树常驻，内存风险已记录于设计文档 §1） |
| ✅ 已完成 | **PRM 训练数据 Rollout 完整脚本 + 数据报告**（2026-08-27）：`scripts/run_rollout.sh`（数据缺失自动 preprocess → `--seed`+`--trees` 抽样（= MCTS 树数）→ 真实 Rollout → SQLite 落库 + 报告）；`mcts/report.py` 生成 `outputs/mcts/rollout_report.{json,md}`（吞吐/失败率/回报/标注统计，M3 验收口径；`--report-only` 可只读报告）；单测扩至 65 个（新增 report 结构与内容断言），A800 通过 |
| ✅ 已完成 | **sglang Qwen3.5-4B 本地后端**（2026-08-27）：`scripts/run_sglang_qwen4b.sh` 硬编码快捷启动（GPU1 :30000，`--reasoning-parser qwen3` + `--tool-call-parser qwen3_coder` 官方解析器；修复了 0.5.18 已移除的 `--enable-paddings/--enable-chunked-prefill` 与 A800 不支持的 fp8 KV cache；CUDA graph 崩溃应急 `--disable-cuda-graph`）；`test/test_sglang_server.py` 6 个手动验证用例（健康检查/纯文本/tool-call/工具往返/litellm 同路径，设 `SGLANG_BASE_URL` 即启用，否则跳过）；**手动测试已通过** |
| ✅ 已完成 | **10 样本真实 rollout demo**（2026-08-27，qwen3.5-4B @ sglang）：`scripts/run_rollout.sh --trees 10 --seed 42 --concurrency 40` —— 10 树 × N=5 = **50 次真实 rollout / 0 失败 / 77s**，reward 中位 2.25/3.0，全链路（读数据→抽样→真实 Rollout→SQLite→报告）端到端通畅；**发现**：①50/50 全部 correct（θ=0.5 过宽 + qwen 定位强）→ root_mc 全 1.0 → **门控未触发 → 标注 0 条**（无 best/leaf/add，需调 `--reward-threshold` 产出 PRM 分步标注）；②qwen3.5-4B 从不提交（exit_status 全 LimitsExceeded，7–8 步撞 step_limit=8，submission 空）→ 需提高 `--step-limit` 观察提交行为 |
| ✅ 已完成 | **提交协议改造：submit_locations 结果提交工具**（2026-08-27，仿 CodeScout `localization_finish`）：① 新工具 `submit_locations`（schema/校验/规则提示词在 `agent/submit_tool.py`，OpenAI function-calling，`{file(必填), class_name, function_name}`，对齐 CodeScout TOOL_DESCRIPTION 规则 1–4 + IMPORTANT）；② 接入 mini-swe-agent **不改其源码**——`agent/submit_model.py::SubmitLocationsModel`（继承 LitellmModel，`tools=[bash, submit_locations]`，经 `get_model(model_class=...)` 注入，根 rollout 与 probe 回放同路径）+ `agent/submit_agent.py::SubmitAgent`（继承 DefaultAgent：`execute_actions` 拦截 submit 动作 → 抛 `Submitted`，**任务结束标志同步修改**：exit_status="Submitted"、submission=结构化 locations JSON，`mcts/steps.py::extract_exit` 无需改动）；③ **默认轮次上限 8→20**（config.yaml / executor / CLI / run_rollout.sh）；④ **催收机制**：到达轮次上限或提前结束（LimitsExceeded/TimeExceeded/RepeatedFormatError）但未提交 → 移除 exit、注入 user 催收消息要求用 submit_locations 输出结果、放行最后一轮（`_agent_loop` 统一循环，probe 续跑复用），催收只发一次；⑤ **完全结构化判定**：reward 只认有效提交（`mcts/reward.py::reward_from_trajectory_exit`，无提交/无效提交 → 0，对齐 CodeScout），旧 git-diff 判定保留为兼容函数；⑥ bash 魔法串提交**默认禁用**（`AttachContainerEnvironment.magic_submit` 开关，config `mcts.submit.magic_submit`，CLI `--magic-submit/--no-magic-submit`）；⑦ 新提示词 `config/mini_submit.yaml`（纯定位模式，对齐 CodeScout system_prompt_custom_finish）；⑧ 报告加提交率（store rollouts.submitted 聚合列 + report rollout_stats.submitted/submit_rate）；⑨ 单测 +38（submit 工具解析/SubmitAgent 催收/结构化 reward），**A800 全量 235 passed / 17 skipped**，dry-run 与旧库 report-only 迁移验证通过 |
| ✅ 已完成 | **提交协议改造实机验证（gemma-4 @ vLLM，2026-08-27）**：`scripts/run_rollout.sh --trees 10 --seed 42 --concurrency 40 --reward-threshold 2.0 --model openai/gemma-4 --base-url http://localhost:8010/v1 --output outputs/mcts_gemma` —— **565 rollouts / 0 失败 / 1068s（吞吐 ~31/min，受共享机负载拖累）**；**提交率 1.0（565/565 exit_status="Submitted"，submit_locations 结构化提交全部生效）**；完全结构化判定 reward 分布 {1.0×248, 3.0×290, 部分 1.5–2.67}，θ=2.0 下 4/10 树 MC∈(0,1) **门控触发 → 标注 159 条（best 84 / leaf 20 / add 55，覆盖 4 实例）**；**催收实机生效**：15 条 rollout 撞 20 轮上限（n_calls=21 = 20+1 grace）收到 user 催收消息后全部在最后一轮提交成功；对比旧 demo（qwen θ=0.5：0/50 提交、0 标注）全链路目标达成 |
| ✅ 已完成 | **v5 存储/驱动重构（2026-09-03）**：SQLite 三表（tree_instances/nodes/rollouts，删 annotations 表与整树快照）+ **会话原子提交**（一次 (select+locate) 判定单事务：is_consumed/visits/n_rounds/in_pool；失败不落库）+ TreeDriver 会话式执行流（无 DB 从零注册、DB 整树恢复、中断会话重做）+ report/CLI 适配（`--fresh` 旧库备份重建）；施工文件 `docs/construction/00-05`；单测全量 266 passed；A800 真机旧库迁移 + **kill -9 断点恢复验收**（done 树跳过零重写、running 树续跑、容器无泄漏） |
| ✅ 已完成 | **M2/M3 实机批量 2000 树（2026-09-08，qwen3.5-4B @ sglang）**：`scripts/run_rollout.sh --trees 2000 --seed 42 --reward-threshold 0.6 --concurrency 40 --resume`（输出 `outputs/batch500`，全程 ~4.35 天）—— **2000/2000 树 done（0 failed / 0 budget_exhausted）**；rollouts **116,444**（失败 104 / 0.09%）、nodes 23,294、**consumed 会话 20,465**、**leaf 派生 2,412（433 实例）**、root 门控 964 树；提交率 0.767、correct 0.594、avg_reward 0.59；LLM 调用 ~48 万、容器创建 86,065 无泄漏、DB ~5GB；验收达标（失败 <5%、标注非零、`--resume` 可续、容器无泄漏） |
| ⏳ 未开始 | **M4–M5**：PRM 训练（PEFT LoRA，输入 = outputs/batch500 的 v5 state.db 派生步级样本，见 §3.1）→ 交付报告（SQLite 崩溃恢复已在 v5 完成） |

> 后续每个里程碑完成后在此表中更新状态并记录日期。

---

## 0. 现状盘点与设计前提

| 已有资产 | 说明 |
| --- | --- |
| `agent/init_env.py`（EnvManager v2） | **创建即用、用完即销毁**（不复用、无缓存）；每次 `get_env` 新建容器，`release_env` 幂等删除；支持 zip 缓存、容器内 apt、`git apply` patch |
| `agent/base_agent.py`（RepoAgent） | 在指定容器内原样运行 mini-swe-agent；返回 `exit_status / submission / cost / n_calls / trajectory` |
| `agent/tracing.py`（Phoenix） | 可选 OTLP 上报每次执行的 litellm 调用轨迹，`save_trace_json` 落盘 |
| 本地 LLM 后端 | **vLLM gemma-4** `http://localhost:8010/v1`（GPU0，tool-calling，litellm 用 `openai/gemma-4` 前缀）；**sglang Qwen3.5-4B** `http://localhost:30000/v1`（GPU1，`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`，启动脚本 `scripts/run_sglang_qwen4b.sh`，litellm 用 `openai/qwen3.5-4B`）—— 当前 rollout 主路径已切到 qwen3.5-4B |
| 数据 | `ref_papers/codescout/data/swe_smith/train.parquet`（39,187 行）+ `validation.parquet`（100 行）；`qwen3_1.7b_data/`（34,713 行，过滤后子集）。`repo` 列 = `swesmith/{owner}__{repo}.{commit8}`，**本身即一个 GitHub 仓库地址**（`https://github.com/swesmith/{owner}__{repo}.{commit8}`，swesmith 组织下的 SWE-Smith 合成快照仓库，**单 commit**，`base_commit` 恒为 None，`use_patch` 恒为 True） |
| 硬件 | A800 80GB ×2：GPU0 = vLLM gemma-4，GPU1 = sglang qwen3.5-4B（PRM 训练需另择时机/换卡）；conda 环境 `CodeAgentRL`（+ `sglang` 服务环境） |

**关键设计决策（D1–D6）**：

- **D1 步（step）定义**：一步 = agent 一次 `step()`（一次 model query + 随后的工具执行），对应轨迹 `messages` 中
  一条 assistant 消息及其后续 observation 消息。MCTS 节点 = `(instance, 轨迹前缀)`。与 ReARTeR 的
  `Node(question, partial_answer, correct_answer)` 一一对应（`partial_answer` → 动作前缀列表）。
- **D2 终态回报（terminal reward）**：主用 **patch-vs-gold 定位 F1**（agent 提交的 diff 解析出 file/module/function
  改动位置，与 gold `file_changes` 算三层 F1，加和 ∈ [0,3]）——离线、无测试执行、**梯度式**回报（产生更多 0<MC<1 节点，
  是 MCTS 可定位性的前提，同 CodeScout §3.3 + OmegaPRM 门控 `0<MC<1`）；备选 **test-based 二值回报**（SWE-bench 类数据，
  跑 FAIL_TO_PASS，作为严格版开关）。
- **D3 前缀续跑（probe 机制）**：从任意前缀续跑 = **replay 到前缀 + 自由续跑**。将已记录步骤的 action 按序在
  新容器里确定性重放（注入记录到的 observation），精确还原前缀容器状态后切换自由生成。无需修改 mini-swe-agent 源码。
- **D4 并发模型（v2）**：单机多 worker（asyncio 驱动 + 线程池执行 litellm/docker exec 阻塞调用），
  **容器不复用**（每 rollout 新建、完成即销毁，`EnvFactory` 限并发创建数），全局信号量限流，
  SQLite 持久化（rollout 完成即落；v5 起三表 + 会话原子提交 + 整树恢复，见施工文件 docs/construction/01–04）。
- **D5 PRM 训练**：transformers + **PEFT（LoRA）** + 分类头（Qwen2.5-Math-PRM 风格：步尾 token 的 logit 做 sigmoid 二分类；
  或线性头），二分类 CE loss，单卡 A800，bf16。
- **D6 数据防泄漏**：PRM 各 split 按 **repo 级**划分（同仓库实例不跨 split，对齐 CodeScout §4.1 的 128 repos 无重叠原则）。

---

## 1. 数据预处理（Data Preprocessing）

### 1.1 数据读取

- **主数据源**：`ref_papers/codescout/data/swe_smith/{train,validation}.parquet`（schema 已实测探明）：
  `instance_id, file_changes, repo, base_commit, problem_statement, patch, target, prompt, use_patch`。
- **字段格式（实测）**：
  - `repo`：`swesmith/{owner}__{repo}.{commit8}` —— **直接代表一个 GitHub 仓库**（克隆地址即
    `https://github.com/{repo}.git`，如 `https://github.com/swesmith/kurtmckee__feedparser.cad965a3`）。
    SWE-Smith 合成仓库，**每个仓库只有一个 commit**（快照即目标状态），故环境准备**无需 checkout commit**；
  - `base_commit`：恒为 `None`（单 commit 仓库无独立 base commit）；
  - `use_patch`：恒为 `True` —— 该行的 `patch` 是**引入 bug** 的 diff（见 §2.2 环境准备）；
  - `instance_id`：`{owner}__{repo}.{commit8}.{task_type}__{task_id}`（如 `theskumar__python-dotenv.2b8635b7.func_basic__n6cxbsay`），全局唯一键；
  - `problem_statement`：纯文本 issue 描述（GitHub issue 风格，Markdown 标题 + 复现步骤）；
  - `patch`：标准 `git diff` 文本（SWE-Smith 的 bug 引入补丁，apply 后得到 pre-PR 状态）；
  - `file_changes`：**gold 定位目标**（结构见 §1.2 GT 构建）；
  - `target`：`build_dataset.py` 从 `file_changes` 复制的等值副本；
  - `prompt`：`[{"role": "user", "content": problem_statement}]` 单轮消息（codescout 训练时再叠加系统提示词渲染，见 `src/prompts/prompt_builder.py`）。
- **加载代码位置（codescout 内已有，直接复用/对齐）**：
  - `ref_papers/codescout/src/build_dataset.py`：`load_dataset("adityasoni17/SWE-smith-py-code-search", split).to_pandas()` → 加工 → `to_parquet`（即本地 parquet 的**生成代码**，含 shuffle(seed=42)、尾部 100 行切验证集）；
  - `ref_papers/codescout/tests/test_single_file_localization.py`、`tests/test_single_prompt.py`：`pd.read_parquet("data/.../train.parquet")` 的**直接读取**样例；
  - 本项目采用 `pd.read_parquet` 直接读本地文件，无需联网；如需原始测试字段（`PASS_TO_PASS/FAIL_TO_PASS`，当前 parquet 已丢弃），再走 HF `adityasoni17/SWE-smith-py-code-search`。
- 读入后统一为内部 `Instance` 记录（dataclass / 列式 parquet），字段：`instance_id, repo_url, owner, name, commit8,
  problem_statement, patch, use_patch=True, gold: {files[], modules[], entities[]}, source`。

> ✅ **已实现（M0）**：`mcts/instances.py::read_instances` —— `pd.read_parquet` 直读 + `Instance` 归一化
> （`parse_repo` 拆 `owner / name / commit8`）；实测读取 train 39,187 行 / validation 100 行。

### 1.2 数据预处理（过滤与清洗）与 GT 构建

- **过滤规则**（对齐 CodeScout §3.1 与 `build_dataset.py`）：
  1. 丢弃空 / 空白 `problem_statement`；
  2. 丢弃 gold patch **新建或删除文件**的实例（agent 无法预测新建文件名，删除文件无 ground truth）；
  3. 忽略非 Python 文件（README 等）在 ground truth 中的条目；
  4. 去除重复 `instance_id`，格式非法行剔除；
  5. 统计 repo 内实例数分布，`problem_statement` 长度上下限（防过短无信息 / 超长超预算）。
- **GT 构建（三粒度 ground truth，标注 codescout 源码位置）**：
  - 数据侧：`ref_papers/codescout/src/build_dataset.py` 第 11–13 行 —— `dataset["target"] = dataset["file_changes"]`（**直接复制**，不做解析）；
  - 奖励侧（解析口径的真正定义者）：`ref_papers/codescout/src/rewards/file_localization/file_localization.py`
    `multilevel_localization_f1_reward()` —— 从 `instance["file_changes"]` 逐条解析：
    - **files**：每条 `change["file"]`（`src/pptx/chart/plot.py`）；
    - **modules**：`change["changes"]["edited_modules"]`（`src/pptx/chart/plot.py:PlotTypeInspector`）；
    - **entities**：`change["changes"]["edited_entities"]`（`src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type`）；
    - `added_entities / added_modules` 与 `edited_*` 一并计入（解析代码含 None 兜底）；
  - 本项目按同一口径实现 `mcts/instances.py` 的 `gold` 抽取（`path:name` 规范形式），并与 `mcts/reward.py`
    的 agent patch 解析**复用同一解析器**（见 §2.4），保证 GT / 预测可比；
  - 论文侧依据：CodeScout §3.1 "extract ground truth by using patch-processing scripts from LocAgent and enhance them
    to (i) detect additions of member functions/class attributes (ii) capture import/global changes (iii) ignore docstring edits"。
- **产出**：`outputs/mcts/instances.parquet`（含过滤后全部实例，每行附解析好的 `gold`）+ 一份过滤报告（原数量 / 保留数量 /
  各规则剔除数 / repo 数，验收时对照 CodeScout 的 39K / 128 repos 数量级）。

> ✅ **已实现（M0）**：`mcts/instances.py` —— `extract_gold`（三粒度，`added_*` 与 `edited_*` 一并计入、None 兜底、
> 非 Python 条目忽略）、`filter_instances`（五条规则，配置见 `mcts/config.yaml`）、`patch_creates_or_deletes_files`
> （git diff 头解析）。实测：39,287 → 39,284（仅剔除 3 条 `statement_too_long`，131 repos 不变；本数据集中
> 新建/删除文件、空语句、重复、非 Python gold 均为 0 条）。CLI：`python -m mcts.instances build`，
> 产出 `outputs/mcts/{instances.parquet,data_report.json,data_report.md}`。

### 1.3 PRM 数据划分（分三层）

1. **实例层**（rollout 执行范围）：按 repo 分组后，取 60% repo 的实例作为**数据生成池**（阶段 1 先采样 500–2000 实例
   跑通全链路，再扩至 5K–10K）；
2. **repo 级防泄漏划分**（PRM 数据集）：`PRM-train : PRM-dev : PRM-test = 80% : 10% : 10%`（按 repo 分组，同 repo 不跨 split）；
3. **步级样本划分**：MCTS 标注展开成步样本后，按 `instance_id` 分桶（同实例的步全部落入同一 split），
   dev/test 各保留 500–1000 条标注步，用于 PRM 步级评估与 best-of-N 下游验证（D6）。
- 划分配置写入 `mcts/config.yaml`，随机种子固定，输出 `splits.parquet`（instance_id → split 映射）。

> ✅ **已实现（M0）**：`mcts/splits.py::make_splits` / `build_splits` —— 固定 seed=42 可复现；实测 131 repos →
> 生成池 79 repos / 23,032 实例，PRM train/dev/test = 63/8/8 repos（17,208/3,881/1,943 实例）；
> 校验：同 repo 不跨 split（0 泄漏）、`prm_split` 只出现在生成池内、seed 重放结果逐行一致。
> CLI：`python -m mcts.splits build`，产出 `outputs/mcts/{splits.parquet,splits_report.*}`。

### 1.4 数据预处理流水线（可执行脚本，2026-08-26）

M0 除 `mcts/` 包外，另提供一键执行与运维脚本（均已在 A800 实跑验证）：

| 脚本 | 作用 | 关键参数 |
| --- | --- | --- |
| `scripts/preprocess_data.sh` | 端到端串联：①解析过滤（`mcts.instances build`）→ ②三层划分（`mcts.splits build`）→ ③**可选**仓库缓存（`batch_repo_pull`） | `--data-dir`（**默认 `data/swe_smith`**）、`--cache-dir`、`--skip-instances/--skip-splits`、`--cache-workers/--cache-retries/--cache-limit`、`--cache-ignore-failures`、`--dry-run` |
| `scripts/batch_repo_pull.py` | 批量拉取数据集中全部仓库到本地 zip 缓存（rollout 网络预热；幂等、并发、失败重试与续拉，退出码 1 表示有失败） | `--instances` / `--data-dir` / `--repos`（三选一，优先级递降）、`--cache-dir`、`--workers`、`--retries`、`--limit`、`--report` |

- **数据位置**：默认 `data/swe_smith/{train,validation}.parquet`（A800 上已就位，与 `ref_papers/codescout/data/swe_smith` 同一份数据），可用 `--data-dir` 指向其它位置；
- **产物**：`outputs/mcts/{instances,splits}.parquet` + `data_report.*` / `splits_report.*` /（可选）`repo_cache_report.json`；
- **幂等性**：三步均可重跑 —— instances/splits 确定性（seed=42 固定）；缓存 zip 已存在自动跳过、失败仓库原样重跑续拉；
- **验收数字（实跑）**：39,287 → 39,284 实例 / 131 repos；生成池 79 repos（23,032 实例）；PRM train/dev/test = 63/8/8 repos（17,208/3,881/1,943 实例）；无 repo 泄漏；
- **仓库缓存状态**：✅ **全量预热已完成**（2026-08-26）—— 131/131 仓库拉取成功（0 失败，约 103s），
  zip 缓存目录 `/media/shared_e/lyq/repos`（131 个 zip）；`.env` 已配置
  `CODEAGENTRL_REPO_CACHE_DIR=/media/shared_e/lyq/repos`，rollout 直接走 docker cp + 解压离线路径；
  增量补拉 / 续拉：`bash scripts/preprocess_data.sh --cache-dir /media/shared_e/lyq/repos --cache-workers 4`；
- **测试基建**（配套）：`test/conftest.py::resolve_test_image` 默认固定 `codeagentrl-agent:ubuntu24`（移除 `python:3.11-slim` 兜底、缺镜像 fail-fast），集成测试不再触发容器内 apt 安装。

---

## 2. MCTS 核心逻辑（数据生成引擎）

### 2.1 LLM 实例获取（模型路由层）

- **主路径（当前，2026-08-27 起）**：A800 本地 **sglang Qwen3.5-4B**（`http://localhost:30000/v1`，GPU1），
  启动脚本 `scripts/run_sglang_qwen4b.sh`（`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`，
  硬编码快捷启动）；litellm 路由 `openai/qwen3.5-4B --base-url http://localhost:30000/v1`。
  实测：10 样本 demo 50 次真实 rollout / 0 失败 / ~1.5s 每次（并发 40）。
- **备路径（离线）**：A800 本地 vLLM `gemma-4`（`http://localhost:8010/v1`，GPU0），
  模型名 `openai/gemma-4`；已启用 tool-calling + gemma4 parser（M1 冒烟用）。
- **备路径（外部 API）**：`gpt-5` 等云端模型（litellm 直连），用于小样本对标与疑难实例重标注（对齐 ReARTeR
  `my_config_2.yaml` 的强生成器重标注思路）。
- **封装**：`mcts/llm.py` 提供 `build_model_config`（base_url / temperature / cost_tracking，Agent 路径统一取用）
  与备用 `CompletionClient`（纯文本补全，指数退避重试）；版本约束 `openai==2.54.0` / `litellm==1.97.0` 锁定不升级。
- **sglang 服务手动验证**：`test/test_sglang_server.py`（6 用例：健康检查 / 纯文本 / tool-call /
  工具往返 / litellm 同路径；`SGLANG_BASE_URL` 未设置时自动跳过）。

### 2.2 Agent 实例创建与环境准备

- **Agent 实例**：复用 `agent.base_agent.RepoAgent`（含 `AttachContainerEnvironment`），不改 mini-swe-agent 源码。
  每实例配置：`model_name=openai/gemma-4`、`base_url`、`config_file=config/mini.yaml`（issue-fix 提示词）、
  `step_limit`（回合上限，CodeScout 用 4–6 回合，此处默认 8，可配）、`temperature`（rollout 随机采样 0.7–1.0，对齐 ReARTeR 的多样性来源）。
- **环境准备**（对齐 codescout `src/utils/instance.py::clone_instance` 的 SWE-Smith 分支 + 现有 EnvManager）：
  - **仓库克隆**：`repo` 列本身即仓库地址 —— `git clone https://github.com/{repo}.git`（或按
    `swesmith/{owner}__{repo}.{commit8}` 生成 zip 缓存，`EnvManager` 缓存键直接用该字符串）；
  - **无需 checkout commit**：SWE-Smith 合成仓库为**单 commit 快照**，且 `base_commit=None`；
    `EnvManager.get_env(repo_url, commit=None)`（commit 传 None，对齐 `clone_instance` 的 `commit_id=None` 跳过 checkout 分支）；
  - **apply patch 得到 pre-PR（bug）状态**：`use_patch=True` 恒成立 —— clone 后 `git apply` 应用 `patch`
    （bug 引入 diff；对齐 `clone_instance` 中 `patch is not None → git apply` 与 `build_dataset.py --use_patch` 注释
    "SWE-Smith whose patch actually introduces the bug"）；`init_env` 的 patch 参数已支持，需验证对 SWE-Smith diff 格式（统一 git apply，无需 `--3way`）；
  - 容器内预装 ripgrep（`CODEAGENTRL_INSTALL_RIPGREP` 已默认 true）；
  - 容器池上限与 LRU 淘汰（128 repos 规模，防容器数失控）；`MSWEA_MAX_CONCURRENT` 控制单容器并发 docker exec。

> ✅ **已实现（M0）**：`mcts/env.py::instance_env_params` / `prepare_env` —— SWE-Smith 分支：
> `commit = instance.base_commit`（恒为 None）→ `get_env(repo, commit=None, patch=patch)`，
> **无 commit 时不切换**（EnvManager 在 commit 非空时才 `git checkout`；zip 缓存与容器内 clone
> 两条路径均有单测锁定，见 `test/test_init_env.py::TestNoCheckoutWhenCommitNone`）；
> `use_patch=True` 恒成立 → patch（bug 引入 diff）克隆后 `git apply`。对齐 codescout `clone_instance`。
- **轨迹与步解析**：`RepoAgent.run()` 的 `trajectory["messages"]`（`trajectory_format: mini-swe-agent-1.1`）→
  切分为步序列 `steps = [(action_msgs, observation_msgs)]`；提交后不允许继续；异常出口
  （`LimitsExceeded / TimeExceeded / RepeatedFormatError`）轨迹同样入库并标记。
- **提交协议（2026-08-27 改造，仿 CodeScout）**：提交 = 调用 **`submit_locations` 工具**
  （`agent/submit_tool.py`：schema + 严格校验 + 规则提示词，对齐 CodeScout `localization_finish`），
  `agent/submit_model.py::SubmitLocationsModel`（`tools=[bash, submit_locations]`，经
  `get_model(model_class=...)` 注入）+ `agent/submit_agent.py::SubmitAgent`（`execute_actions`
  拦截 submit 动作 → 抛 `Submitted`：**exit_status="Submitted"、submission=locations JSON**，
  即新的任务结束标志）；**bash 魔法串提交默认禁用**（`AttachContainerEnvironment.magic_submit`
  开关，mcts config `submit.magic_submit`）；**轮次上限默认 20**；到达上限或提前结束但未提交时
  **注入 user 催收消息**要求用 submit_locations 输出结果，放行最后一轮（催收一次）。

### 2.3 Rollout 高并发实现

**目标**：10 万+ 次 rollout 的可扩展采集管道（并发 40–50），具备 SQLite 断点恢复、限流、可观测、成本控制。

> ✅ **已实现（v2，2026-08-27）**：完整设计见 `docs/mcts_engine_design.md` —— **任务驱动架构**：
> **RolloutTask（最小调度单元，payload 自包含）→ TaskQueue（asyncio.PriorityQueue + future
> 注册表，按 task_id 去重幂等）→ Worker 池（`asyncio.to_thread` 执行阻塞的容器+Agent 调用，
> 全局信号量限流）→ TreeDriver（每实例一棵树，`await gather(*submit(N))` 实现**节点内 N 次
> rollout 并发**；多 TreeDriver 同时跑 ⇒ **多树并发**）→ EnvFactory（容器**创建即用、用完即销毁**，
> 不复用、无 key、不限总创建次数，只限并发创建数）→ StateStore（SQLite：树与结果常驻内存、
> rollout 完成即落 + 会话原子提交）。模块：`mcts/{tasks,executor,replay,steps,reward,llm,node,
> locate,store,run_mcts}.py`；入口 `python -m mcts.run_mcts`（`--sample/--instance-ids/
> --concurrency/--create-concurrency/--max-rollouts/--resume/--dry-run/--phoenix-tracing`）。
> 单测 `mcts/tests/` 62 个全部通过（本地 + A800）。

- **调度架构**（单机即可，对齐 CodeScout 的 SkyRL 异步思想但轻量实现）：
  - 主进程 = asyncio 事件循环 + 任务队列（优先级：root rollout > probe rollout，probe 依赖其父 rollout 完成）；
  - worker 池：每个 worker 一个 asyncio Task，内部用 `asyncio.to_thread` / 线程池执行阻塞的 litellm 同步调用与 docker exec；
  - 全局并发信号量（目标 40–50，随 vLLM 吞吐实测调优）+ 容器**并发创建**节流（`creation_concurrency`，默认 8）；
  - **容器不复用**（v2）：`EnvManager` 每次 `get_env` 新建容器（uuid 命名互不冲突）、
    `release_env` 幂等删除；executor 在 `finally` 中销毁（成功/失败都删），
    **任务完成归还自动销毁**，无池化、无状态复位；
- **持久化（v5，SQLite 三表，2026-09-03 起）**：
  - `outputs/mcts/state.db`（WAL）三表：`tree_instances`（status：not_started/running/budget_exhausted/done/failed + n_rounds + head）、`nodes`（prefix_json/mc/visits/in_pool；创建即落、结算置 ready）、`rollouts`（成功即落、`is_consumed` 消费账本、失败不落库）；**删 annotations 表与整树快照**（leaf 由 非root∧mc==0 派生）；
  - **会话原子提交**：一次 (select+locate) 的判定（is_consumed/visits/n_rounds/expanded.in_pool）在会话成功结束时**单事务**落库 —— 崩溃中途 = 判定未提交 = 该 (node, rollout) 未消费 → resume 重选重做（probe 结果内容寻址复用，仅补在飞缺槽），**无需保存二分中间状态**；
  - **断点续跑（v5）**：run() 从 DB 整树载入重建；树 `status ∈ {done, failed}` 跳过，running/budget_exhausted/not_started 续跑；缺槽（含失败/预算部分）一律按 rollouts 行数 lazy 补跑；`kill -9` 恢复经 A800 真机验收；
- **可靠性**：
  - **重试与降级**：单次 rollout 失败重试（退避）；仍失败记为 failed（不计入 N，对齐 ReARTeR
    bad_gen 语义）；单实例全部失败标记 `failed` 保留已有数据；
  - **预算熔断**：全局 / 每实例 rollout 数 + LLM 调用软预算；预算将尽时节点**部分提交**
    （只跑允许数量的 rollout，MC 照常计算），超限停止扩展（保留已生成数据）；
  - **可观测**：Stats（submitted/done/failed/吞吐/平均回报/平均步数/成本/容器创建数/DB 计数）+ 周期日志；
    Phoenix 可选（每次 rollout 一个 trace，trace_id 入结果，`setup` 进程内一次）；
  - **容器防泄漏**：用完即销毁 + 冒烟校验（`docker ps` 无残留）。
- **性能基线（验收）**：并发 40–50 时单 rollout 平均耗时与吞吐记录在案；目标 500 实例 × 5 rollouts + MCTS 扩展
  在预算内跑完（详见 §5 里程碑 M3）。

### 2.4 MCTS 树逻辑（OmegaPRM divide-and-conquer 的 agent 域适配）

核心实现参考 `ref_papers/ReARTeR/PRM_Data/module.py`（Node / perform_rollouts / calculate_mc_score /
select_best_node / locate_error / best-leaf-add 标注）与 OmegaPRM 论文；差异点在于**续跑机制与回报函数**。

> ✅ **已实现（2026-08-27）**：`mcts/node.py`（MCTSNode / MC / QU 选择，纯函数）、
> `mcts/locate.py`（`locate_error` 二分，返回 expanded/leaf，异步任务驱动）、`mcts/steps.py`
> （轨迹 → 步序列）、`mcts/replay.py`（D3 前缀回放 + 自由续跑）。树逻辑通过任务队列与
> rollout 执行解耦（`get_node` / `perform_rollouts` 注入），ReARTeR docs/01 §11.1–11.9
> 数值示例端到端复现（单测锁定 QU 数值与标注顺序）。

- **节点**：`MCTSNode(instance, prefix_steps[], mc_score, visits, rollouts[], visited_flags)`；根节点 prefix 为空；
  **内容寻址**（`node_key = sha1(前缀步序列化)`）——同一前缀的探测节点共享 rollout（不重复计算 MC）。
- **rollout(node)**：replay 前缀 + 自由续跑 → 完整轨迹 + 终态回报 `r ∈ [0,3]`（D2），二值化 `correct = (r ≥ θ)`，θ 默认 0.5。
- **MC 分数**：`MC(node) = Σ correct / N`，N=5（ReARTeR / OmegaPRM 同值；probe 节点同样 N 次）。
- **门控**：`0 < Σcorrect < N`（即 0<MC<1）才进入树搜索（对齐 ReARTeR 与 OmegaPRM：全对 / 全错不再定位）。
- **选择**：QU 值（对齐 ReARTeR `select_best_node`）：`QU = α^(1−MC) · β^(len/max_len) + c·√(Σvisits)/(1+visits)`，
  默认 α=0.5, β=0.9, c=0.125, max_len=6；在"全部 0<MC<1 节点 × 未访问 rollout"上全局取最大；选中节点 `visits += 1`（回传）。
- **定位首个错误步**（`locate_error`，二分）：对选中的 rollout 步序列 `[s1..sk]`，反复试探：
  - 构造 probe 节点 = 已确认正确前缀 + 左半段步序列，对其做 N 次 rollout（**并发**）得 MC；
  - `MC==1` → 前缀已稳，停止（错误在右半之外，无需再探）；
  - `0<MC<1` → 错误在右半，把左半并入正确前缀继续二分；
  - `MC==0` → 错误在左半，收缩左半；步数 < 2 时终止，记录 **leaf**（首个错误位置候选）。
- **标注产出（v5 派生口径，2026-09-03 起；评审 D8）**：不再落盘 best/leaf/add（删表）——
  - 节点角色由 MC 派生：`leaf`（首个错误步负样本）= **非 root ∧ mc==0**；`0<mc<1` = 扩展节点（in_pool，PRM 连续标签来源）；`mc==1` 全对探针仅留作内容寻址复用；root 无前缀不产标注；
  - select 消费账本 = `rollouts.is_consumed`（会话结束提交，root 消费同样记录）；
  - 下游 PRM 展开（§3.1）按 `(instance_id, node_key)` join `nodes.prefix_json` + head 还原消息；
- **终态回报实现（奖励模块 `mcts/reward.py`）**：
  - **结构化判定（主路径，2026-08-27 起）**：`reward_from_trajectory_exit` —— 仅
    `exit_status == "Submitted"` 且 submission 为合法 locations JSON 才计 reward；
    `locations_localization_f1(locations, gold)` 把结构化 locations 映射为
    files/modules/entities 三集合（对齐 CodeScout `parse_structured_outputs`）与 gold
    三层集合算 F1，加权和 ∈ [0,3]；**无有效提交 → 0**（对齐 CodeScout 未调用 finish 即 0 分）；
  - `patch_localization_f1(agent_patch, gold)`（保留，兼容/对照）：解析 agent 提交 diff 的
    改动位置（files 取 `+++/---` 头、modules/entities 取 hunk 的 git function context +
    体内 class/def，`path:Class` / `path:Class.method` 命名与 gold 同一口径；类上下文
    **逐 hunk 重置**，跨 hunk 不沿用），与 gold 三层集合算 F1（复用 CodeScout
    `compute_file_f1_score` 的公式），加权和 ∈ [0,3]；旧路径原为容器内
    `git add -N . && git diff` 取 agent 改动，新流程不再执行（executor 已移除该 docker exec）；
  - `test_based_reward`（备选开关）：SWE-bench 式测试执行（阶段 1 不启用）。
- **合成验证**：ReARTeR docs/01 的数值示例（11.1–11.9）构造 mock 数据端到端单测（QU 选择、二分方向、
  标注顺序、并发提交），全部通过；真实 agent 接续在 M1 冒烟验证。

---

## 3. PRM 训练（PEFT 微调）

### 3.1 PRM 数据构建

- **输入**：nodes/rollouts（v5 派生：leaf = 非 root∧mc==0 负样本、0<mc<1 连续标签）+ prefix_json + head 还原的轨迹前缀。
- **样本展开**：每条标注 → 步级样本 `(instance_id, step_idx, prompt=完整对话到 step_idx-1, step_content, label)`：
  - **主标签（二值，OmegaPRM 式）**：由 leaf 定位结果回填——`leaf` 前的步正确（1）、`leaf` 所在步错误（0）、之后步不参与；
    `add/best` 的 MC 作为连续辅助标签（保留用于回归式训练对照）；
  - **连续标签（ReARTeR 式）**：`label = MC > 0.5`（对齐其 PRM 阈值 0.5；KTO 用 0.4 属阶段 2，不混用）；
- **格式**：统一为 LLaMA-Factory 兼容的对话式 JSONL / parquet（`messages + label`），并支持自定义 PRM Trainer 直读；
  - 长轨迹截断：prompt 按 token 截断（默认 8K / 16K 两档），步 content 单独保留；
  - **类别平衡**：负样本（首个错误步）天然少 → 对 leaf 步过采样 / 加权 CE；按步位置分桶统计分布。
- **划分**：按 §1.3 的 instance 级 split 落盘 `outputs/prm/{train,dev,test}.parquet`。

### 3.2 训练（PEFT LoRA）

- **基座**：Qwen3-1.7B / 4B（与 CodeScout 模型族一致；1.7B 起跑，4B 留作对比）；也可选 agent 同款 gemma-4 作对照。
- **结构（两方案，主推 A）**：
  - **A（special-token logit，Qwen2.5-Math-PRM 风格）**：每步步尾插入 `<extra_0>`，取该 token 的 logits 做 sigmoid 二分类；
    LoRA 只作用于 attention/MLP，无需新增参数，推理时逐步滑动窗口打分；
  - **B（线性分类头）**：从最后 step token 的 hidden state 接 2 层 MLP → sigmoid（参考 OmegaPRM `ProcessRewardModel` 思路）；
  - 训练目标均为二分类 **CE loss**（混合 3.1 的连续标签时用平滑 CE / MSE 辅助）。
- **PEFT 配置**：`peft.LoraConfig`（r=16/32, α=32/64, target_modules=q/k/v/o/gate/up/down, dropout 0.05）；
  冻结基座，仅训练 LoRA 适配器 + 分类头。
- **训练参数（初版）**：bf16，lr 1e-5（余弦 + warmup 0.1），per-device batch 8–16 + grad accum 至全局 64–128，
  max_seq_len 8K–16K，epochs 1–3（早停看 dev loss / 步级准确率），单卡 A800（GPU1），`transformers.Trainer` + `peft`，
  checkpoint 每 500 步保存，`PEFT` adapter 与 head 分开导出。
- **训练入口**：`prm/train_prm.py --data outputs/prm --base Qwen3-1.7B --lora ... --output outputs/prm/ckpt`。

### 3.3 评估

1. **步级质量**：dev/test 上 step 二分类准确率 / AUC / F1；按步位置分桶（第 1、中间、末段）报告 → 检验 early-step bias
   （ReARTeR 指出 PRM 对早期步打分虚高的问题）；
2. **与终态回报相关性**：PRM 对完整轨迹各步打分，轨迹级 PRM 分数（如 min/mean/最后步）与终态回报的 Spearman 相关；
3. **下游 best-of-N**：同一实例 N=5 条 rollout，PRM 选最高分 vs 随机选择 vs 真实回报最优，
   对比 resolution/F1 提升（对齐 OmegaPRM 的 best-of-N 报告口径）；
4. 输出 `outputs/prm/eval_report.md`（含混淆矩阵与样例）。

---

## 4. 新增代码结构

```
CodeAgentRL/
├── mcts/                      # 阶段 1 数据生成引擎（新建包）
│   ├── config.py / config.yaml  ✅ 全部超参（N、θ、α/β/c、并发、容器池、预算、split 种子）
│   ├── instances.py           ✅ 数据读取 + 过滤 + gold 抽取（§1.1–1.2，M0）
│   ├── splits.py              ✅ 三层划分（§1.3，M0）
│   ├── env.py                 ✅ 环境准备：无 commit 不切换 / git apply patch（§2.2）
│   ├── llm.py                 ✅ 模型配置构建（base_url/temperature/cost_tracking）+ CompletionClient（备用）
│   ├── reward.py              ✅ 终态回报：diff 解析 + patch-vs-gold 定位 F1（主）/ test-based（备）
│   ├── steps.py               ✅ 轨迹 → 步序列解析 / 出口识别 / 前缀消息裁剪 / node_key（D1）
│   ├── node.py                ✅ MCTSNode / MC / QU 选择（纯函数，ReARTeR 移植）
│   ├── locate.py              ✅ locate_error 二分 + expanded/leaf 返回（会话驱动，无标注落盘）
│   ├── replay.py              ✅ 前缀回放 + 自由续跑（D3，含回放一致性校验）
│   ├── tasks.py               ✅ 任务驱动高并发引擎：TaskQueue / EnvFactory / Worker /
│   │                              TreeDriver / MCTSPipeline / 预算 / SQLite 续跑（D4，见设计文档）
│   ├── store.py               ✅ v5 SQLite 三表：tree_instances/nodes/rollouts（WAL+单写锁；
│   │                              commit_session 会话原子提交 / load_tree_state / 旧库检测）
│   ├── executor.py            ✅ 生产 rollout 执行器：create_env → 回放/自由跑 → 回报 → destroy_env
│   ├── run_mcts.py            ✅ 入口 CLI（--sample/--instances/--resume/--concurrency/
│   │                              --create-concurrency/--max-rollouts/--dry-run/--phoenix-tracing）
│   └── tests/                 ✅ 树逻辑合成测试（ReARTeR 数值示例端到端）+ 会话/恢复/孤儿用例；全量 266 passed
├── prm/                       # PRM 训练（新建包）
│   ├── build_dataset.py       # 标注 → 步级样本（§3.1）
│   ├── train_prm.py           # PEFT LoRA + 分类头 CE（§3.2）
│   └── eval_prm.py            # 步级指标 / 相关性 / best-of-N（§3.3）
└── outputs/
    ├── mcts/                  # ✅ instances/splits.parquet + 数据报告；state.db（v5 三表）+ 派生标注（M1+）
    └── prm/                   # train|dev|test.parquet / ckpt / eval_report.md
```

> M0 单测落根目录 `test/`（与既有测试同一套件）：`test/test_instances.py`、`test/test_splits.py`、
> `test/test_env.py`、`test/test_init_env.py::TestNoCheckoutWhenCommitNone`（v2 语义：每次 get_env 新建容器）；
> M1–M3 的树逻辑 / 并发 / 续跑 / store / report 测试放 `mcts/tests/`（65 个，FakeExecutor 离线可跑，见设计文档 §8）；
> sglang 后端手动验证测试 `test/test_sglang_server.py`（6 用例，`SGLANG_BASE_URL` 启用）。
>
> 配套脚本（✅ 已落地，见 §1.4）：`scripts/preprocess_data.sh`（端到端流水线，`--data-dir` 默认
> `data/swe_smith`，仓库缓存可选）、`scripts/batch_repo_pull.py`（仓库缓存预热）、
> `scripts/run_rollout.sh`（PRM 训练数据 Rollout 完整脚本：抽样 → 真实 Rollout → SQLite → 报告）、
> `scripts/run_sglang_qwen4b.sh`（sglang Qwen3.5-4B 快捷启动）；测试基建：
> `test/conftest.py::resolve_test_image` 默认固定 `codeagentrl-agent:ubuntu24`。

---

## 5. 里程碑与验收标准

| 里程碑 | 内容 | 验收标准 |
| --- | --- | --- |
| **M0 数据层** ✅（2026-08-25，脚本化 08-26） | §1 全部脚本 + 数据报告 + 流水线脚本（`preprocess_data.sh` / `batch_repo_pull.py`） | 过滤数量级对齐 CodeScout（39,284 / 131 repos）；split 无 repo 泄漏（0）；gold 抽取得出三层 F1 可复算（38,680/39,284 实例含 entities，单测覆盖 added_* 并入）；一键重跑可复现 |
| **M1 单 rollout** ✅（2026-08-27 实机验证） | 1 实例 × N=5 rollouts 全链路（env → agent → 回报） | 轨迹格式正确；patch 解析与回报可解释（gemma-4 冒烟 reward 1.83、qwen3.5-4B demo reward 中位 2.25）；replay 前缀一致性校验通过；容器无泄漏 |
| **M2 单实例 MCTS** ✅（2026-09-08 实机批量验证） | rollout → MC 门控 → 选择 → 二分定位（标注 v5 派生：leaf/连续标签） | 合成测试（ReARTeR 数值示例）通过 ✅；2000 树实测：964 树门控、leaf 派生 2,412（433 实例）、消费会话 20,465，标注语义正确 |
| **M3 并发管道** ✅（2026-09-08 实机批量完成） | 500–2000 实例批量生成，断点续跑 | 引擎单测覆盖并发/预算/续跑 ✅；10 样本 demo 50 rollouts / 77s / 0 失败（吞吐基线 ~40 rollouts/min @ 并发 40）；`--resume`/`kill -9` 恢复验证通过、容器无泄漏（2000 树 / 116,444 rollouts / 0 失败树） |
| **M4 PRM 训练** | 数据集 + PEFT LoRA 训练 + 评估 | dev 步级准确率 ≥ 基线（随机/多数类）；best-of-N 相对随机选择有提升 |
| **M5 交付** | 全量数据 + 评估报告 + 本文档更新 | 报告包含：数据统计、训练曲线、相关性、样例；为阶段 2（PRM 引导筛选 + KTO/DPO）提供直接输入 |

**预算提示**：阶段 1 建议 2,000 实例 ×（1 根 rollout×5 + 平均 3–4 个 probe×5）≈ 4–5 万次 rollout 调用；
先以 500 实例验证每实例 token 均值（日志记录），再按预算上限放大。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 前缀回放状态漂移（非确定性命令 / 网络 / 时间戳） | MC 失真 | 回放后校验 observation 哈希与期望一致，不一致则该节点标记重rollout；优先只读类搜索命令轨迹 |
| token 成本超预算 | 无法收敛完成 | 每实例 / 每日预算熔断；小样本先行；temperature 0.7–1.0 采样上限；截断控制 |
| patch 解析与 gold 对齐口径差异 | 回报噪声 | 同一解析器双用（gold 与 agent patch）；人工抽检 20 条比对 |
| 类别不平衡（正样本远多于负） | PRM 偏向全对 | leaf 过采样 / 加权 CE；按步位置分桶评估 |
| PRM 早期步虚高（ReARTeR 指出） | 下游筛选失效 | 分桶评估暴露；后续引入 TD-lookahead / 平衡标注（v2 预留） |
| **θ=0.5 过宽 → 全部 correct → 门控 0<MC<1 不触发 → 无标注**（10 样本 demo 实测：50/50 correct、标注 0 条） | 跑再多也产不出 PRM 分步标签 | 提高 `--reward-threshold`（建议 2.0，让"强匹配"才算对）；或换更难实例 / 压缩温度多样性；按实例筛选 0<MC<1 比例监控门控触发率 |
| **模型从不提交（exit_status 全 LimitsExceeded，撞 step_limit）**（qwen3.5-4B 实测） | 提交协议语义未走通；rollout 成本被 step_limit 固定 | 提高 `--step-limit`（8→12）；reward 不依赖 submission（容器 git diff），语义仍有效；后续可评估"强制提交模板"提示词 |
| vLLM / sglang 吞吐瓶颈 | 并发上不去 | 实测调并发；sglang `--max-running-requests` 已配 40；A800 GPU1 已起 qwen3.5-4B |
| sglang 启动问题（CUDA graph 捕获崩溃 / 旧 flag / fp8 不支持） | 服务起不来 | 脚本已修复（`--disable-cuda-graph` 应急；`--kv-cache-dtype auto`；移除 0.5.18 已删除的 flag）；`test/test_sglang_server.py` 手动验证 |
| 依赖版本互锁（openai/litellm/phoenix） | 环境崩坏 | 遵守 README §7 依赖表，新装包前 `pip check` |

---

## 7. 参考资料对照

| 参考 | 本项目对应 |
| --- | --- |
| OmegaPRM（arXiv:2406.06592）：divide-and-conquer MCTS + 二分首个错误步 + 步级标签训练 PRM | §2.4 树逻辑、§3.1 步级标签 |
| ReARTeR `PRM_Data/module.py`：Node / perform_rollouts(N=5) / MC / QU 选择 / locate_error / best-leaf-add / 阈值 0.5 | §2.4 全部结构与默认参数 |
| ReARTeR `docs/01_mcts_rollout.md`：数值示例与实现细节（temperature 随机、bad_gen 跳过、死循环防护） | 合成测试用例、rollout 多样性、格式失败不计入 N |
| CodeScout：SWE-Smith 数据与过滤、rollout 环境准备（zip 缓存 / ripgrep / patch 应用）、三粒度 F1 奖励、OpenHands-Bash 提示词 | §1 数据、§2.2 环境、§2.3 异步管道（SkyRL 思想轻量落地）、§2.4 回报设计 |
| CodeScout reward 配置（`multilevel_localization_f1_reward` + `multiturn_reward`） | `mcts/reward.py` 的回报组合开关 |
| **codescout 直接加载数据的代码**：`src/build_dataset.py`（HF→parquet 生成）、`tests/test_single_file_localization.py` / `tests/test_single_prompt.py`（`pd.read_parquet` 直读） | §1.1 数据读取实现对齐 |
| **codescout GT 构建代码**：`src/build_dataset.py`（`target=file_changes` 复制）、`src/rewards/file_localization/file_localization.py::multilevel_localization_f1_reward`（files/modules/entities 三粒度解析口径） | §1.2 GT 构建（复用同一解析器） |
| **codescout 环境准备代码**：`src/utils/instance.py::clone_instance`（`git clone https://github.com/{repo}.git` → `commit_id=None` 跳过 checkout → `git apply` patch）、`src/prompts/prompt_builder.py`（提示词渲染） | §2.2 环境准备（单 commit 仓库免 checkout） |
