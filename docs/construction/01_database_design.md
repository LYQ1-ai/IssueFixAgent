# 施工文件 01 —— 数据库表设计（v5）

> 配套：00 文档 D1–D13；02 文档（操作 API）；04 文档（事件写点上下文）。

## 1. 设计原则

1. **表里只存事实，状态是派生谓词**：节点"要不要补 rollout"= 查 rollouts 缺槽；"能不能被选"= select 时用 `mc + is_consumed + in_pool` 现判；
2. **内容寻址**：`node_key = sha1(前缀步骤序列化)`，同一前缀全局唯一（跨会话、跨 resume 复用）；
3. **失败不落库**：rollout 硬失败 = 无行 = 补跑信号；软异常（LimitsExceeded / replay_drift）落库 + 标记；
4. **会话即事务**：一次 (select+locate) 的**判定**在会话成功结束时单事务提交；**事实**（节点行、rollout 行）随时落；
5. **in_pool 只随会话提交**：中断会话的半成品探针永不入池（防 resume 候选池污染）。

## 2. DDL

```sql
-- PRAGMA: journal_mode=WAL; synchronous=NORMAL; 单写锁(threading.Lock);
--        FK 是否强制(PK foreign_keys=ON)由运行配置决定（默认不强制，见 02 文档）

CREATE TABLE IF NOT EXISTS tree_instances (
    instance_id        TEXT PRIMARY KEY,        -- 原始数据集 instance_id（1:1 一棵树）
    status             TEXT NOT NULL,           -- not_started / running / budget_exhausted / done / failed
    n_rounds           INTEGER NOT NULL DEFAULT 0,  -- 非 root 消费轮数（cap = mcts.n_rounds_cap）
    messages_head_json TEXT,                    -- system+user 头部：注册时渲染写入，此后不变
    root_mc            REAL,                    -- 缓存：root 结算 MC（报告/审计）
    updated_at         REAL
);

CREATE TABLE IF NOT EXISTS nodes (
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,                  -- sha1(前缀序列化)；root 节点为 'root'
    status      TEXT NOT NULL DEFAULT 'rollout',-- rollout | ready（见 01 §3.2）
    prefix_json TEXT NOT NULL,                  -- 前缀步骤原文（node_key 不可反推，重建对话必需）
    mc_score    REAL,                           -- 结算缓存（可由 rollouts 重算）
    visits      INTEGER NOT NULL DEFAULT 0,     -- 真账本：每消费一次 +1（QU 探索项）
    in_pool     INTEGER NOT NULL DEFAULT 0,     -- 真账本：会话成功提交时为 expanded 置 1；root 恒 1
    created_at  REAL,
    PRIMARY KEY (instance_id, node_key)
);

CREATE TABLE IF NOT EXISTS rollouts (
    instance_id        TEXT NOT NULL,
    node_key           TEXT NOT NULL,
    rollout_idx        INTEGER NOT NULL,        -- 节点内槽位 0..N-1（resume 补槽/复用键）
    is_consumed        INTEGER NOT NULL DEFAULT 0,  -- 消费账本：会话成功结束时置 1
    consumed_iteration INTEGER,                 -- 可选：第几轮被消费（审计/复现）
    result_json        TEXT NOT NULL,           -- steps+reward+exit_status+submission+reward_details；不含大块 messages
    n_steps            INTEGER,                 -- 缓存（QU 的 β^len 用，免 parse）
    correct            INTEGER,                 -- 聚合缓存（报告）
    reward             REAL,                    -- 聚合缓存（报告）
    submitted          INTEGER,                 -- 缓存：exit_status=='Submitted'
    replay_drift       INTEGER NOT NULL DEFAULT 0,
    exit_status        TEXT,
    created_at         REAL,
    PRIMARY KEY (instance_id, node_key, rollout_idx)
);

CREATE INDEX IF NOT EXISTS idx_nodes_instance  ON nodes (instance_id);
CREATE INDEX IF NOT EXISTS idx_rollouts_consumed ON rollouts (instance_id, is_consumed);
```

## 3. 字段语义与派生谓词

### 3.1 tree_instances
| 列 | 可否推导 | 写点 |
| --- | --- | --- |
| status | — | 注册 not_started；激活 running；会话流收束 done/failed；预算熔断 budget_exhausted |
| n_rounds | 可由 rollouts.consumed_iteration 计数推导（不直观，存） | 每次非 root 会话提交 +1（同一事务） |
| messages_head_json | 可由 config+problem 重建 | 注册时一次写入，之后不变 |
| root_mc | 可由 root 节点行推 | 收束时缓存 |

### 3.2 nodes
| 列 | 性质 | 语义 |
| --- | --- | --- |
| status | 缓存（可观测/过滤） | `rollout` = 尚未完成一次补跑结算（含创建即 k=0）；`ready` = 完成一次结算、mc 已算（k 可为 N 或 <N，如失败/预算部分）。**恢复逻辑不信任它**，一律按缺槽补跑 |
| prefix_json | 事实 | 必须有（哈希不可反推） |
| mc_score | 缓存 | 可由 rollouts 重算（load 时可不信任、重算兜底） |
| visits | **真账本** | 每次消费会话提交 +1 |
| in_pool | **真账本** | 会话提交时对 expanded 置 1；root 恒 1；leaf/全对探针恒 0 |

派生谓词（不落库）：
- `needs_rollout(node) := count(rollouts 行) < N`（N = mcts.n_rollouts）
- `is_leaf(node) := node_key != 'root' AND mc_score == 0`（= 原 leaf 标注，查询期派生）
- `in_pool_effective(node) := (node_key == 'root') OR (in_pool == 1)`（root 恒在池）
- `selectable(node, idx) := in_pool_effective ∧ 0<mc<1 ∧ rollouts[idx] 存在 ∧ is_consumed=0`

### 3.3 rollouts
| 列 | 性质 | 语义 |
| --- | --- | --- |
| is_consumed | **唯一不可推导的消费账本** | 0→1 仅一次，**locate 会话成功结束时**（崩溃中途 = 0 = 可重选重做）。root 的消费同样记（root 不产 best 标注，故无法从别处反推，必须落） |
| consumed_iteration | 可选审计 | 该 (node,rollout) 在第几轮被消费 |
| 其余结果列 | 事实/缓存 | 成功即落一次，之后不变 |

## 4. 事件写点总表（DB 何时写什么）

| 事件 | 表操作 | 事务边界 |
| --- | --- | --- |
| ① 注册实例（无 DB / 新选实例） | INSERT tree_instances(status=not_started, head) | 单条 |
| ② 调度激活 | UPDATE tree status=running | 单条 |
| ③ 节点诞生（root / locate 探针） | INSERT nodes(status=rollout, in_pool=root?1:0) | 单条（FK 保证先于其 rollout） |
| ④ 每条 rollout 成功 | UPSERT rollouts（is_consumed=0 + 结果字段） | 单条 |
| ⑤ 节点结算（一次补跑完成） | UPDATE nodes status=ready, mc_score | 单条 |
| ⑥ 会话成功结束（**原子批**） | UPDATE rollouts is_consumed=1[, consumed_iteration]；UPDATE nodes visits=visits+1；UPDATE tree n_rounds=n_rounds+1(非 root)；UPDATE nodes in_pool=1（本会话 expanded） | **同一事务** |
| ⑦ 预算熔断 | UPDATE tree status=budget_exhausted | 单条 |
| ⑧ 树收束 | UPDATE tree status=done/failed, root_mc | 单条 |

> 崩溃窗口分析：④/⑤ 随时可中断（事实已落、无判定依赖）；⑥ 单事务原子 → 要么全提交（该 rollout 已消费、expanded 已入池），要么全没有（该 rollout 未消费 → resume 重做，probe 的 rollout 经内容寻址复用，只补在飞缺槽）。**不存在中间状态。**

## 5. 恢复语义（resume 视角）

1. 每实例：查 tree_instances；无行 → 走从零注册；`done/failed` → 跳过；否则激活；
2. 激活后**整树载入**：nodes 全部行 + 每节点 rollouts 行 → 内存重建节点/池；
3. root 若缺槽 → 补跑至结算（root 是后续门控前提）；
4. 进入 select 循环：候选 = `selectable` 谓词；被中断会话的 (node,rollout) 因 `is_consumed=0` 自然成为候选 → **重做会话**（probe 内容寻址命中 → 复用已落 rollout；缺槽补跑）；
5. 收敛：无候选 或 n_rounds 达 cap → done。

## 6. 与旧 schema 的差异清单（迁移用）
| 旧 | 新 | 说明 |
| --- | --- | --- |
| instances 表 | tree_instances | 语义扩展（not_started、n_rounds） |
| nodes：status/rollouts_json/无 in_pool | status 精简、删 rollouts_json、加 in_pool | rollouts_json 由 rollouts 表推导 |
| rollouts：无 is_consumed | 加 is_consumed/consumed_iteration | 消费账本从节点 JSON 下沉 |
| annotations 表 | **删除** | leaf 派生、best/add 砍（见 00 D8） |
| 失败行（error 非空） | **不落库** | 旧库清理：`DELETE FROM rollouts WHERE error IS NOT NULL`（或整体重建） |

## 7. 索引与 PRAGMA
- WAL + `synchronous=NORMAL` + 单写 `threading.Lock`（现状保留）；
- 索引：nodes(instance_id)、rollouts(instance_id,is_consumed)；PK 前缀已覆盖 (instance,node_key) 查询；
- FK：schema 声明外键作文档化约束；运行期默认不强制（`PRAGMA foreign_keys` 关闭），如需强约束在连接建立时开启——但**开启后节点行必须先于 rollout 存在**（本设计已保证写入顺序）。
