# MCTS 高并发 Rollout 引擎设计（v2：任务驱动 + 容器不复用 + SQLite 持久化）

> 对应 Overall_PLAN.md「阶段 1 引擎与设计模式」（mcts/{node,locate,replay,tasks,executor,store,run_mcts}.py），
> 里程碑 M1–M3。本文档是"高并发 MCTS 数据生成引擎"的权威设计；代码以本设计为准。
>
> **v2 变更（2026-08-27 评审确认）**：
> 1. **容器不复用**：每个 rollout 新建容器、完成后自动销毁（创建 3–8s vs rollout 1–5min，
>    复用收益低且引入状态污染；与 D3 probe 回放"从干净状态重放"天然匹配）；
> 2. **并发模型**：树结构只管并发提交任务，任务队列 + 全局信号量负责并发执行并回传结果
>    （保持 asyncio + to_thread）；
> 3. **SQLite 持久化**：树结构与 rollout 结果**常驻内存**，定期/完成即写 SQLite
>    （取代 v1 的每 rollout 一个 JSON 文件）；断点恢复代码后置实现，schema 现在就定。

## 0. 结论先行：为什么不用现成库

调研了候选方案（2026-08）：

| 候选 | 结论 |
| --- | --- |
| [mcts-reasoning](https://github.com/queelius/mcts-reasoning)（LLM 文本 MCTS） | 面向"文本续写采样"，无 Agent/容器/环境状态概念，无法承载"docker 容器内执行 bash 修改仓库"的 rollout；不可用 |
| [mctx (Google DeepMind)](https://github.com/google-deepmind/mctx)（MCTS-in-JAX） | 面向 RL 策略优化（固定动作空间、可向量化 rollout 函数），与"LLM Agent 在容器内自由执行"不匹配；不可用 |
| Ray / Dask / Celery / RQ | 通用分布式/任务框架，功能过剩：单机、Docker 容器绑定执行、无 broker 基础设施；且引入新依赖与 README §7 的依赖互锁（openai/litellm/phoenix）有冲突风险；不可用 |
| **asyncio + asyncio.PriorityQueue 自研轻量任务引擎** | **采用**：与 PLAN D4 一致，零新增依赖，天然支持"多树并发 + 节点内多 Rollout 并发 + 优先级 + 断点续跑"，容器/LLM 阻塞调用经 `asyncio.to_thread` 落入线程池 |

> 结论：**没有现成库同时满足"Issue-Fix Agent（docker 容器执行）"+"每个树节点多次 Rollout 高并发"**。
> 采用轻量自研：单进程 asyncio 事件循环 + 优先级任务队列 + 线程池执行阻塞调用（docker exec / litellm），
> 与 PLAN D4"asyncio 驱动 + 线程池执行阻塞调用 + 全局信号量限流"完全一致。

---

## 1. 目标与约束

- **目标**：10 万+ 次 rollout 的可扩展采集管道；并发 40–50；多棵树并发；单节点 N 次 rollout 并发；SQLite 断点恢复、限流、可观测、预算熔断。
- **约束**：
  - 单机（A800 双卡，vLLM 占 GPU0，容器执行占 CPU/磁盘）；
  - rollout = "在 docker 容器里跑 mini-swe-agent 到终态"（阻塞、1–5 分钟）；
  - **容器不复用**：每个 rollout 新建容器、完成后销毁（创建 3–8s，占比 ≤5%；且 D3
    probe 回放天然要求"从干净 bug 状态开始"）；
  - 每实例一个 MCTS 树；树内节点间有依赖（probe 依赖父 rollout 完成），树间无依赖；
  - **内存**：树结构与 rollout 结果常驻内存（决策记录：**不加活跃树上限**，全部树常驻；
    内存账本 ≈ 活跃树数 × 单树 ~5–25MB，2,000 实例全活跃时 ~10–50GB —— 已作为已知风险
    记录，若实测 OOM 再引入 `max_active_trees` 信号量，接口已预留）。

## 2. 任务模型（任务驱动核心）

**RolloutTask** —— 最小调度单元：

```python
@dataclass(frozen=True)
class RolloutTask:
    task_id: str          # f"{instance_id}:{node_key}:{rollout_idx}"
    instance_id: str
    node_key: str         # 前缀内容哈希（"root" 为空前缀）—— 同前缀节点共享
    rollout_idx: int      # 节点内第几次 rollout（0..N-1）
    kind: str             # "root" | "probe"
    priority: int         # 0=root（最高），1=probe
    payload: dict         # instance / prefix_steps / prefix_messages / prefix_actions / temperature
```

- **node_key**：`"root"` 或 `sha1(前缀各步内容序列化)`。**按内容寻址** ⇒ 同一棵树里不同 rollout 产生的相同前缀 probe 节点共享同一批 rollout（ReARTeR 的 `(question, partial_answer)` 聚合在树内提前完成，rollout 不重复）。
- 每个任务携带**完整执行上下文**（payload 自包含），worker 无需访问树结构 —— 这是"任务驱动"的核心：树逻辑（驱动者）与执行细节（worker）解耦。

## 3. 架构总览（单进程 asyncio）

```
主进程 asyncio event loop
├── TreeDriver(instance) × 全部实例            ← "树"的并发（内存常驻）
│     每个驱动者 = 一个 asyncio Task：
│     root rollout × N ─► TaskQueue ─► Worker ─► to_thread(executor.run)
│     MC 门控（0<MC<1）                         │   │      ├─ EnvFactory.create_env(instance)
│     select_best_node（QU）                    │   │      ├─ ReplayRunner（前缀回放+自由续跑）
│     locate_error（二分，probe rollout × N 并发）│  │      ├─ reward（patch-vs-gold 定位 F1）
│     best/leaf/add 标注 + 树快照 ──────────────│   │      └─ finally: destroy_env(container)
│                                              │   │
│     └─► SQLite state.db（完成即落 + 定期快照） │   │
└── WorkerPool: K 个 asyncio Task 消费队列      │   │
    全局信号量 max_concurrency 限流（40–50）────┘   │
    EnvFactory: 容器创建即用/用完即销毁 + 创建节流信号量（8–16）────┘
```

- **队列**：`asyncio.PriorityQueue`，元素 `(priority, seq, task)`；`TaskQueue.submit(task) -> Future`（按 task_id 去重，重复提交返回同一 Future，幂等 —— 断点续跑/并发扩展的安全保证）。
- **Worker**：`get() → await sem.acquire() → await asyncio.to_thread(executor.run, task) → sem.release() → future.set_result(result)`。
- **TreeDriver**：`await asyncio.gather(*[queue.submit(t) for t in 节点 N 个 rollout 任务])` —— **单节点 N 次 rollout 并发**由此实现；多个 TreeDriver 同时 submit ⇒ **多树并发**。

## 4. 并发控制与容器工厂（v2）

| 层 | 机制 | 配置 |
| --- | --- | --- |
| 全局 rollout 并发 | asyncio.Semaphore | `mcts.max_concurrency`（目标 40–50，随 vLLM 吞吐实测调优） |
| 容器**并发创建** | threading.Semaphore（只限同时创建数，不限总次数） | `mcts.creation_concurrency`（默认 8） |
| 单容器并发 docker exec | 继承 `MSWEA_MAX_CONCURRENT`（mini-swe-agent 层） | 环境变量 |

**EnvFactory（`mcts/tasks.py`，取代 v1 ContainerPool）**：
- `create_env(instance)`：每次调用都新建一个**全新**容器（EnvManager v2 不复用语义），
  完成环境准备（zip 缓存 docker cp + 解压 → 免 checkout → git apply patch = bug 状态）；
- `destroy_env(container)`：`docker rm -f` 幂等删除 —— executor 的 `finally` 保证
  成功/失败都销毁（**任务完成归还自动销毁**）；
- **无 repo+commit key**：容器互不共享，无需 key 分组；总创建次数不限；
- 创建并发节流：`creation_concurrency` 信号量包住 `get_env`（docker run + cp + unzip），
  防 40–50 路同时解压的瞬时 IO 风暴；
- `exec(container, cmd)`：容器内执行命令（rollout 后取 patch 用）。

> v1 的 `reset_container`（git checkout/clean/重放 patch）随容器复用一起移除。

## 5. Rollout 执行（executor，阻塞段在线程池）

```
executor.run(task):
  container = env_factory.create_env(instance)          # 每次全新容器（bug 状态）
  try:
    temperature = uniform(0.7, 1.0)                     # ReARTeR 多样性来源
    if task.kind == "root":                             # 无前缀：自由跑
        trajectory = replay.run_free(instance, container)
    else:                                               # probe：前缀回放 + 自由续跑（D3）
        trajectory, drift = replay.run_probe(instance, container, task.payload)
    reward, details = reward_fn(instance, trajectory, container)
    return RolloutResult(...)                           # 落库由 TreeDriver 统一负责
  finally:
    env_factory.destroy_env(container)                  # 用完即销毁（成功/失败都删）
```

**D3 前缀续跑（replay.run_probe）**：
1. `AttachContainerEnvironment` 挂到容器（不新建）；
2. **确定性回放**：按序重放前缀各步的 action（`env.execute`），与记录 observation 的 `raw_output` 比对（一致 ⇒ 状态还原；不一致 ⇒ `replay_drift=True` 标记，可配置 `replay_require_match` 硬失败）；
3. **自由续跑**：`DefaultAgent` 预填 `messages = 记录到第 k 步的原始消息`（system/user/assistant/tool 完整对话），然后复刻 `DefaultAgent.run()` 的循环（FormatError 恢复 / LimitsExceeded 出口）逐 `step()` 直到 exit；
4. 续跑步预算 = `step_limit - len(prefix_steps)`。

**终态回报（reward_fn，D2）**：容器内 `git add -N . && git diff` 取 agent 改动（含新建文件；submission 为 diff 时优先用 submission）→ 解析 files/modules/entities（`path:Class` / `path:Class.method` 命名，与 gold 同一口径）→ 三粒度 F1 加权和 ∈ [0,3]（codescout `multilevel_localization_f1_reward` 公式）→ `correct = (reward ≥ θ)`，θ=0.5。

## 6. 树逻辑（OmegaPRM / ReARTeR 移植，见 §2.4）

- `MCTSNode(instance, prefix_steps, mc_score, visits, rollouts[], visited_flags)`；节点按 `node_key` 注册在树内 registry（同前缀共享）；
- `perform_rollouts(node, N)` → 异步：submit N 个任务并 gather（并发）；
- MC = Σcorrect / N；门控 `0<MC<1` 才进入标注循环（全对/全错不定位）；
- `select_best_node`：QU = `α^(1−MC)·β^(len/max_len) + c·√(Σvisits)/(1+visits)`（α=0.5, β=0.9, c=0.125, max_len 可配），全局最大、跳过已访问 rollout；
- `locate_error`（二分）：试探节点 = 已确认前缀 + 左半段，N 次 rollout 得 MC；`==1` 停 / `>0` 向右 / `==0` 向左记 leaf；
- `process_annotations`：select → best → visits+1 → locate_error → 扩展，最多 20 轮；收尾写 leaf + add；
- 标注与树快照写入 SQLite（`mcts/store.py`），内存树是权威、DB 是镜像。

## 7. 可靠性

- **SQLite 持久化（`outputs/mcts/state.db`，WAL）**：`mcts/store.py::StateStore`
  - `rollouts` 表：rollout 完成即 upsert（单条事务，崩溃安全最小粒度）；result_json 存
    steps + 回报 + submission（**不含大块轨迹**，控制 DB 体积，轨迹可从 steps 推导）；
  - `nodes` 表：树结构**定期快照**（每次 select/locate 一轮 + 实例完成时）；
  - `annotations` 表：best/leaf/add；`instances` 表：状态 + root_mc + messages_head
    （system+user 头部，probe 前缀拼接与恢复用）；
  - 写并发：WAL + 单写锁（40–50 worker 并发单条事务可承受）；
- **断点续跑（--resume，v2 已实现部分）**：
  - 实例级：`instances.status ∈ {done, failed}` ⇒ 跳过；
  - 节点级：`load_node_rollouts` 从 rollouts 表读回节点已有结果，缺失 idx 才重新提交
    （幂等、不浪费已产生数据）；
  - **崩溃恢复（重建树状态机继续 select/locate）**：后置里程碑实现（schema 已就绪）；
- **重试与降级**：executor 异常（docker 无响应 / LLM 失败）⇒ 该 rollout 重试 ≤3 次（退避）；仍失败 ⇒ 记为 failed（不计入 N，对齐 ReARTeR bad_gen 语义）；单实例全部失败 ⇒ 实例标记 `failed`，保留已有数据；
- **预算熔断**：全局 rollout 数 / 每实例 rollout 数 / 全局 LLM 调用数（n_calls 累计）三档，超限停止新提交（在飞任务跑完，部分标注照常落盘）；
- **可观测**：Stats 计数器（submitted/done/failed/rate/avg reward/avg steps/cost）+ 周期日志；可选 Phoenix 追踪（每个 rollout 一次 trace，trace_id 写入结果；线程池共享进程，`setup_phoenix_tracing` 仅需一次）。

## 8. 可测试性（注入边界）

引擎与 Docker/LLM 完全解耦，通过两条注入点单测：

```
MCTSPipeline(executor_factory=..., env_factory=..., store=...)
  executor: RolloutExecutor(Protocol).run(task) -> RolloutResult
```

- 单测注入 **FakeExecutor**：脚本化 `(correct, steps)` 结果，可复现 ReARTeR 11.1–11.9 数值示例；并发语义测试用带 sleep + 活跃计数器的 FakeExecutor 断言"节点 N 次 rollout 同时执行、全局并发 ≤ cap"；
- SQLite 存储单独单测（`mcts/tests/test_store.py`）：schema / upsert-load 往返 / 并发写；
- `steps/reward/node/locate/tasks/store` 均不 import `agent`/`minisweagent`（惰性），纯逻辑可离线跑。

## 9. 目录与模块

| 文件 | 职责 |
| --- | --- |
| `mcts/steps.py` | 轨迹 → 步序列（assistant+tail）/ 出口识别 / 前缀消息裁剪 / node_key |
| `mcts/reward.py` | diff 解析（files/modules/entities）+ 三粒度 F1 + 终态回报 |
| `mcts/llm.py` | 模型配置构建（base_url/temperature/cost_tracking）+ CompletionClient（备用） |
| `mcts/replay.py` | D3 前缀回放 + 自由续跑（run_free / run_probe） |
| `mcts/node.py` | MCTSNode / MC / QU 选择（纯函数） |
| `mcts/locate.py` | locate_error（异步任务驱动）+ 标注条目构建 |
| `mcts/tasks.py` | TaskQueue / EnvFactory / Worker / TreeDriver / MCTSPipeline / Stats / 预算；持久化统一走 SQLite |
| `mcts/store.py` | StateStore：rollouts / nodes / annotations / instances 四表，WAL + 单写锁；聚合查询（报告用） |
| `mcts/report.py` | 数据报告：从 state.db 聚合吞吐/失败率/回报/标注 → rollout_report.{json,md}（run_mcts 自动生成，--report-only 可只读报告） |
| `mcts/executor.py` | 生产 rollout 执行器（AgentRolloutExecutor：create_env → 回放/自由跑 → 回报 → destroy_env） |
| `mcts/run_mcts.py` | CLI：`--sample/--instances --resume --concurrency --create-concurrency --budget --report-only --dry-run ...` |
| `agent/init_env.py` | EnvManager v2：**创建即用、用完即销毁**（无缓存、无复用）；保留 zip 缓存 / patch / checkout 环境准备 |
| `scripts/run_rollout.sh` | **PRM 训练数据 Rollout 完整脚本**：数据缺失自动 preprocess → `--seed`+`--trees` 抽样（= MCTS 树数）→ 真实 Rollout → SQLite 落库 + 报告 |

## 10. 验收对照（M1–M3）

- M1：单实例 root rollout × 5 全链路（env → agent → 回报）跑通，轨迹格式正确、patch 解析可解释；
- M2：合成测试通过（ReARTeR 11.1–11.9 数值示例），真实实例标注语义正确（best/leaf/add，SQLite 落库）；
- M3：500–2000 实例批量，吞吐/失败率记录在案（失败 <5%），中断后 `--resume` 可续（SQLite 节点复用），容器无泄漏（每个 rollout 用完即销毁 + 冒烟校验）。
