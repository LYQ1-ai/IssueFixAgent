# 施工文件 00 —— 总览与改造范围（MCTS 引擎 v5 重构）

> 配套阅读：`docs/mcts_fault_tolerance.md`（容错现状）、`docs/mcts_engine_design.md`（v2 引擎设计）、`PLAN.md §2.3–2.4`。
> 本组文档（00–05）是 **v5 状态机 + 会话原子提交 + 精简表设计** 的实施规格，基于 2026-09 系列设计评审结论。

---

## 1. 本期目标（一句话）

把 MCTS 引擎从"**内存权威 + DB 定期快照镜像**"改为"**DB 事实权威 + 会话原子提交**"：

- 节点/rollout 全部**实时落库**（创建即落、成功即落）；
- **一次 (select+locate) 会话**是崩溃恢复的原子单元——会话结束单事务提交判定（消费标记/visits/轮数/入池）；
- 崩溃任意时刻：**要么整会话没提交（= 该 rollout 未消费 → resume 重做、已完成 probe 全复用），要么整会话已提交**——无中间状态；
- 砍掉冗余（annotations 表、best 标注、rollouts_json、整树快照）。

## 2. 设计定稿决策清单（评审结论，实施以此为准）

| # | 决策 | 内容 |
| --- | --- | --- |
| D1 | 表结构 | 3 张表：`tree_instances / nodes / rollouts`（**无 annotations 表**，见 D8） |
| D2 | 树状态 | `not_started → running → done / failed`，`budget_exhausted` 可续跑（预算恢复后回 running） |
| D3 | 节点状态 | 显式 `status ∈ {rollout, ready}`；创建即 rollout（k=0），一次结算完成后置 ready（mc 已算）；**无 ready/closed 之外的跃迁** |
| D4 | 消费账本 | 由原 `visited_json` 下沉为 **rollouts.is_consumed**（0→1 仅一次），**locate 会话成功结束时提交**（非 select 命中时） |
| D5 | 会话原子性 | 一次 (select+locate) 的判定（is_consumed / visits+1 / 非 root 轮数+1 / expanded 入池）在**单个 DB 事务**内提交；事实（节点行、rollout 行）随时落 |
| D6 | 池成员 | `nodes.in_pool` **会话结束时**只为该会话的 expanded（0<mc<1）探针置 true；root 恒在池；leaf/全对探针永不入池。**禁止用 mc∈(0,1) 推导入池**（防中断会话孤儿污染候选池） |
| D7 | 失败语义 | rollout 硬失败不落库（缺槽 = 补跑信号）；恢复规则 = 一律按 rollouts 表缺槽补跑（**不看 status**，覆盖预算部分结算节点） |
| D8 | 标注 | **砍 annotations 表与 best**。leaf = `非 root ∧ mc==0`（查询期派生）；add 语义 = `in_pool`。PRM 数据 = 遍历 nodes 直接构造 |
| D9 | 链式标签 | 采用**简化版**：leaf 用自身前缀展开（前 len-1 步=1、末步=0），不依赖父 rollout 链接 |
| D10 | head | `messages_head_json` 在**注册（not_started 创建）时**用 config 提示词 + problem_statement 渲染写入，此后不变 |
| D11 | 恢复 | resume 从 DB **整树载入**（tree + nodes + rollouts）重建内存；`done/failed` 跳过；未提交会话（is_consumed=0）会被 select 自然重选重做 |
| D12 | run_id | 本期 **1 instance = 1 树**（instance_id 即树 PK），run_id 作为预留扩展点不实现 |
| D13 | 并发 | 保留现状：多 TreeDriver 并发 + 节点内 N 次 rollout 并发（TaskQueue/Worker/信号量/EnvFactory 节流）；**树级调度窗口本期不做**（活跃树上限列为后续优化） |

## 3. 目标模块布局（低耦合分组）

```
mcts/                      目标职责
├── node.py                [纯逻辑] MCTSNode / QU / select_best_node          —— 基本不动
├── locate.py              [纯逻辑] locate_error 二分 + 类型判定              —— 基本不动
├── steps.py               [纯逻辑] Step / 前缀解析 / node_key 内容寻址       —— 不动
├── executor.py            [执行] AgentRolloutExecutor（容器内跑一次 rollout）—— 不动
├── replay.py              [执行] ReplayRunner（probe 回放 + 自由续跑）        —— 不动
├── env.py                 [执行] 实例环境参数（SWE-Smith 分支）              —— 不动
├── llm.py                 [执行] 模型配置 / 备用 CompletionClient            —— 不动
├── reward.py              [执行] 终态回报（layered F1）                      —— 不动
├── store.py               [存储] StateStore（v5 schema + 新 API）            —— ★重写
├── tasks.py               [调度] TaskQueue/Worker/Budget/EnvFactory/
│                                  TreeDriver(会话流)/MCTSPipeline            —— ★TreeDriver 重写，其余不动
├── run_mcts.py            [入口] CLI（参数微调 + 初始化路径）                 —— 小改
├── report.py              [报告] 读 store 聚合（适配新表）                    —— 小改
├── instances.py/splits.py [数据] M0 层，不动
├── config.py/config.yaml  [配置] 增补 v5 键（见 00 §5）
└── tests/                 [测试] 见 05_test_plan.md

agent/                    完全不动（init_env / base_agent / submit_* / shell_tool）
```

**低耦合边界（写代码时遵守）**：
- 纯逻辑层（node/locate/steps）**不 import** store/executor/replay —— 通过注入 `get_node/perform_rollouts` 解耦（locate 现状已如此）；
- 执行层（executor/replay/agent）不感知树逻辑与存储 —— 只消费 `RolloutTask`/`RolloutResult`；
- 存储层（store）只提供"事实/判定"读写，不包含决策逻辑；
- 调度层（tasks.TreeDriver）是唯一把三者接起来的胶水。

## 4. 改造范围（不重复写多余代码）

### 4.1 完全不动（零改动，只做回归测试）
`agent/*`、`mcts/{instances,splits,env,llm,reward,steps,node,locate,replay,executor,config}.py`

### 4.2 小改
| 文件 | 改动 |
| --- | --- |
| `mcts/store.py` | **重写**（见 4.3） |
| `mcts/tasks.py` | 只改 `TreeDriver`（run / process_annotations / 落盘）与 messages_head 辅助的调用点；`RolloutTask/RolloutResult/TaskQueue/Worker/Budget/EnvFactory/Stats/MCTSPipeline` 不动 |
| `mcts/report.py` | 聚合改为读 `tree_instances/nodes/rollouts`（原 `instance_summaries/annotation_summaries` 已删） |
| `mcts/run_mcts.py` | 初始化路径接入（见 03 文档）；`--fresh`/`--force` 等小参数 |
| `mcts/config.yaml` | 增 `store: {db_path}`、`n_rounds_cap` 等键（默认值与原参数一致） |

### 4.3 重写：`mcts/store.py`
- schema v5（3 表 + 索引，见 01 文档）；
- 删：`save_nodes`（整树快照）、`write_annotations`、`rollouts_json` 语义、`annotation_summaries`；
- 增：会话提交等新 API（完整清单见 02 文档）；
- `upsert_rollout / load_node_rollouts / count_rollouts / counts / close` 保留并适配。

### 4.4 删除（现状代码里不再需要）
- `TreeDriver._write_annotations / _write_state`（整树快照 + 收束 DELETE+INSERT 标注）→ 换成 `commit_session`；
- `nodes.rollouts_json` 列、annotations 表；
- `select_best_node` 返回的 `visited_flags` 内存位图 → 以 DB `is_consumed` 为准（内存仅作运行期缓存）。

### 4.5 新增（最小集）
- `mcts/tests/test_store_v5.py`、`test_tasks_v5.py`（会话/恢复用例，见 05）；
- （后续 M4，本期不实现）`mcts/export_prm.py`：nodes → 步级样本导出。

## 5. 配置键变更
`config.yaml` 新增（默认值与现状一致，避免行为漂移）：
```yaml
mcts:
  n_rollouts: 5            # 不变
  n_rounds_cap: 20         # = 原 max_iterations
  store:
    db_path: outputs/mcts/state.db
```

## 6. 旧 state.db 处置（重要）

v5 表语义与旧库（4 表快照式、含 1 万+ error 行与 annotations）**不兼容**，且旧库由 8/31 前后实验产生（失败固化、标注删减等问题已在 fault 文档记录）：

- **默认策略：备份后重建**。`run_mcts --fresh` 时对旧 `state.db` 先 `cp state.db state.db.bak.<ts>`，再以 v5 schema 新建；
- 旧数据（rollouts/nodes）**不迁移**——语义（消费账本、入池、会话）无法从旧结构可靠还原，强行迁移会引入隐蔽错误；
- resume 从 v5 新库起算；实例从 parquet 重新注册。

## 7. 文档导航
| 文档 | 内容 |
| --- | --- |
| 01_database_design.md | v5 表设计：DDL、字段语义、事件写点、崩溃窗口、恢复语义 |
| 02_store_api.md | StateStore 操作层：方法全集（签名/语义/SQL 概要/调用方）、事务约定 |
| 03_init_and_resume.md | 启动路径：无 DB 从零初始化 / 有 DB resume / 树状态机 |
| 04_tree_execution.md | 单树执行：select / locate / 会话原子提交 / 并发任务提交 |
| 05_test_plan.md | 分模块单元测试清单与验收 |
