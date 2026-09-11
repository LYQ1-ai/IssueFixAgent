# MCTS Rollout 容错机制说明（树状态管理 / 落库 / 恢复）

> 面向后续修改与开发：完整阐述当前 MCTS 数据生成引擎的**失败判定、重试、落库、断点续跑
> 与崩溃恢复**机制，包括涉及文件、数据模型、已知问题与修复记录、后续开发建议。
> 配合阅读：`Overall_PLAN.md`、`docs/mcts_engine_design.md`、`docs/reward_design.md`。
> 最后更新：2026-09-01（含当日 head 缺失 / 失败固化 / 标注删减三处修复与遗留问题）。

---

## 1. 概述：容错目标与设计原则

**目标**：10 万+ 次 rollout 的采集管道在**单点失败、进程中断、环境抖动**下：
不丢已算数据（LLM 调用不浪费）、可续跑（`--resume`）、失败不污染正确性、预算可熔断。

**三层防线**：

| 层 | 机制 | 位置 |
| --- | --- | --- |
| 1 | **单次 rollout 失败**：worker 重试 3 次 → 树级重跑 1 次 → 失败不落库 | `tasks.py::Worker` / `TreeDriver._retry_rollout` |
| 2 | **进程中断**：`--resume` 断点续跑（实例级跳过 + 节点级复用 rollout） | `tasks.py::TreeDriver.run` / `store.py::StateStore` |
| 3 | **预算熔断**：全局/每实例 rollout 与 LLM 调用软预算，超限保留部分数据 | `tasks.py::Budget` |

**核心原则**：
- **DB 是崩溃安全镜像，内存是权威**：rollout 完成即落（最小粒度）、树节点定期快照、实例状态与标注落库；
- **失败 = 没跑过**（2026-08-31 修复）：失败结果不落库，resume 时槽位为 None 自动补跑；
- **内容寻址**：节点 key = 前缀内容哈希 → 同一前缀跨运行共享节点与 rollout（复用基础）；
- **标注幂等**：`write_annotations` 先 DELETE 再 INSERT，resume 重跑不重复累积。

---

## 2. 涉及文件与职责

| 文件 | 职责（容错相关） |
| --- | --- |
| `mcts/store.py` | **SQLite 持久化唯一事实源**：`StateStore`（WAL、单写锁）；rollouts/nodes/annotations/instances 四表；`upsert_rollout`（成功即落）、`load_node_rollouts`（按节点读 N 槽位）、`save_nodes`（节点快照）、`write_annotations`（DELETE+INSERT 覆盖）、`set_instance_status`/`get_instance_status`（状态+root_mc+head）、`_migrate`（列补迁移） |
| `mcts/tasks.py` | **任务驱动引擎**：`Worker`（重试/信号量/统计）、`TaskQueue`（去重/优先级）、`Budget`（熔断）、`Stats`（进度）、`TreeDriver`（树状态机 + 恢复逻辑）、`MCTSPipeline`（编排）；head 三函数 `messages_head_from_rollouts` / `resolve_messages_head` / `build_messages_head`；`_retry_rollout`（树级重跑） |
| `mcts/node.py` | `MCTSNode`：`valid_rollouts`（剔除失败）、`compute_mc`（有效样本 MC）、`gated`（门控）、`visited_flags`（每条 rollout 只定位一次） |
| `mcts/executor.py` | 生产 rollout 执行器：**失败捕获**（异常 → `RolloutResult(error=...)`）；结构化判定（layered reward + reward_details 落库） |
| `mcts/replay.py` | `ReplayRunner`：probe 前缀回放（drift 容忍/硬失败）；`run_probe` 入口兜底（prefix 无 user 直接拒绝） |
| `mcts/locate.py` | 二分定位首个错误步；leaf/add/best 标注生成（容错相关：probe 全失败按 MC=0 记 leaf 的局限在此） |
| `mcts/steps.py` | 轨迹→步序列、`messages_for_prefix`（head 提取）、node_key 内容寻址 |
| `mcts/run_mcts.py` | CLI：`--resume / --max-rollouts / --max-rollouts-per-instance / --max-llm-calls / --worker-retries`（经 config） |
| `mcts/config.yaml` | `mcts.worker_retries`（默认 2）、`stats_interval`、`step_limit`、`reward.*` 等 |
| `mcts/tests/` | `test_tasks.py`（resume/预算/失败重跑）、`test_tasks_head.py`（head 解析/重建）、`test_store.py`（落库往返/线程安全）、`test_replay.py`（兜底）、`test_executor.py`（失败结果构造） |

---

## 3. 数据持久化模型（SQLite 四表）

`schema`（`store.py::_SCHEMA`，WAL + 单写锁，`_migrate` 补列）：

```sql
instances  (instance_id PK, status, root_mc, messages_head_json, updated_at)
nodes      (instance_id, node_key, prefix_json, mc_score, visits,
            n_rollouts, rollouts_json, updated_at, PK(instance_id, node_key))
rollouts   (instance_id, node_key, rollout_idx, result_json,
            correct, reward, submitted, error, created_at,
            PK(instance_id, node_key, rollout_idx))
annotations(id AUTOINC, instance_id, node_key, type, mc_score, n_steps, written_at)
```

**落库语义（关键）**：

| 表 | 写入时机 | 内容 | 恢复用途 |
| --- | --- | --- | --- |
| `rollouts` | **成功即落**（`if res.error is None: upsert_rollout`） | result_json = steps + reward + correct + exit_status + submission + **reward_details**；不含大块轨迹（控制体积） | ✅ 节点级复用/补跑依据 |
| `nodes` | 定期快照（`process_annotations` 每轮 + `_write_state`） | prefix_json / mc_score / visits / n_rollouts / rollouts_json | ⚠️ **仅存档**，resume 不读取重建状态机 |
| `annotations` | done / budget_exhausted 时一次性写（**先 DELETE 再 INSERT**） | best / leaf / add + node_key + mc_score | ✅ 幂等覆盖（不重复累积） |
| `instances` | 状态迁移（running→done/failed/budget_exhausted） | status / root_mc / messages_head_json | ✅ 跳过判定 + head 恢复 |

---

## 4. 失败判定与来源

**判定标准**：`RolloutResult.error` 非空（`result.failed = bool(error)`）。

| 来源 | 触发 | 典型错误 |
| --- | --- | --- |
| LLM 调用异常 | litellm 调 sglang 抛错 | BadRequestError(400)、ContextWindowExceededError、InternalServerError(连接)、Timeout |
| 容器层 | create_env / docker exec 失败 | EnvFactory 抛错、exec 超时 |
| 回放硬失败 | `replay_require_match=True` 且回放不一致 | RuntimeError: replay observation mismatch |
| probe 兜底拒绝 | prefix_messages 无 user 消息（现被 build_messages_head 兜底，基本不触发） | ValueError: missing a user message |
| 其它 | executor.run 内未捕获异常 | 统一 catch → error 结果 |

**回放漂移（drift）**：重放输出与记录不一致 → **标记 drift 不失败**（默认容忍，MC 视为噪声；`--replay-require-match` 才硬失败）——属设计预期，非错误。

---

## 5. 重试机制（三层）

```
第 1 层 Worker（tasks.py::Worker._execute_with_retry）
  attempts = 1 + worker_retries（默认 2 → 最多 3 次）
  每次 asyncio.to_thread(executor.run, task)；executor.run 抛异常 → 指数退避
  sleep(backoff^attempt) → 重试；耗尽 → error="retries exhausted"
  ⚠️ executor.run 内部已把异常转 error 结果（不抛），此层兜"executor.run 自身抛异常"

第 2 层 树级重跑（tasks.py::TreeDriver._retry_rollout）
  _perform_rollouts 收到 error 结果 → 同 payload 重新 submit 一次 → 成功替换 / 仍失败保留
  （worker 3 次 + 树级 1 次 = 单 rollout 最多 4 次尝试）

第 3 层 resume 补跑（跨进程）
  失败不落库 → DB 槽位 None → 下次 --resume 的 missing 列表包含该槽位 → 重新提交
```

---

## 6. 断点续跑（`--resume`）全流程

```
重启 → MCTSPipeline → TreeDriver.run(instance)
  ├─ 实例状态 done/failed → 跳过（不重跑）
  ├─ 实例状态 running/budget_exhausted → 继续：
  │    root 节点：load_node_rollouts 按 idx 槽位复用【成功】rollout，
  │               缺失槽位（未跑完/失败未落库）→ 重新提交
  │    → MC 重算 → 门控 → process_annotations 扩展
  └─ probe 节点：按 node_key 内容寻址复用已算 rollout，缺失补跑
```

**head（system+user 对话头部）恢复三级**（`resolve_messages_head` → `build_messages_head`）：

```
1. root rollout 轨迹计算（messages_for_prefix(traj, 0)）   —— 首跑路径
2. 落库的 instances.messages_head_json                     —— resume 路径
3. config 提示词 + 实例 problem_statement 直接渲染重建      —— 任何实例可还原（2026-08-31）
```

**标注幂等**：resume 后树重新 process_annotations → `write_annotations` 先 DELETE 旧标注再 INSERT 新标注，不重复累积。

---

## 7. 崩溃恢复现状与边界

### 7.1 能恢复什么

- **rollout 结果**（最小粒度，成功即落）：步序列 + 终态 reward + correct + 分层细节 → 100% 复用，**LLM 调用零浪费**；
- 实例状态 / root_mc / head；
- 预算额度（Budget 从 stats 重建）。

### 7.2 不恢复什么（树"状态机"）

- `visited_flags`（每条 rollout 是否已定位过）、`visits`（QU 探索项）、
- QU 选择进度、二分进行位置、`best_entries / expanded / leaves` 内存列表、
- `max_iterations` 轮数计数。

**后果**：resume 后从 root **重新走一遍 select/locate 决策**。但由于：
1. **内容寻址**：节点 key = 前缀哈希 → 已算探针节点复用 → MC 一致；
2. **二分确定性**：`locate_error` 对同一组 rollout 结果确定；
→ **已算部分的标注判定不变**；只有**新补算的 probe**（temperature 随机）引入新样本，与"从未中断的一次运行"在统计上等价。

**对 PRM 数据质量的影响**：低——标签正确性由 rollout（复用一致）+ 确定性二分保证，与选择顺序无关；标注最终一次性写入，不会拿到半成品。

### 7.3 已知局限

| 局限 | 影响 | 状态 |
| --- | --- | --- |
| 状态机不重建 | resume 重新决策（CPU 重跑 + 少量新 probe） | 后置项（v5 已实现，见 Overall_PLAN.md §4.4） |
| probe 节点全失败 → `compute_mc()=0` → 记 leaf | 全失败被当"全错"，可能误标负样本 | 待改进 |
| `_write_annotations` 无条件执行 | **head 缺失跳过扩展时会把旧标注 DELETE 成空**（9/1 期间发生过 ~3100 条标注损失） | **待加保护**（见 §9.1） |
| 历史失败记录固化 | 8/31 前落库的 error 记录被 resume 当作"已有"不重跑（当前 DB 10,696 条） | 待清理（见 §9.2） |

---

## 8. 已知问题与修复记录（时间线）

| 日期 | 问题 | 修复 |
| --- | --- | --- |
| 2026-08-27 | qwen 从不提交（0/50） | submit_locations 工具化提交 + 20 轮 + 催收 |
| 2026-08-28 | 三层独立 F1 冗余 | layered Soft-F1 判定（docs/reward_design.md） |
| 2026-08-31 | sglang 重启期大量 Connection error → 失败结果落库被 resume 固化，节点有效 N 减少 | **失败不落库** + **树级重跑 1 次** |
| 2026-08-31 | root 第 0 个 rollout 失败 → head 算不出 → probe 无 user → sglang 400 "No user query" | `messages_head_from_rollouts`（跳过失败取首个有效） |
| 2026-08-31 | resume 后 trajectory 丢失 → head 依赖落库，历史实例 DB 为 NULL | `resolve_messages_head`（回退 DB stored_head） |
| 2026-08-31 | head 仍可能缺失（历史 NULL）→ 成批 probe 失败 | `build_messages_head`（config 提示词 + 原始输入重建，**任何实例可还原**）+ `_perform_rollouts` probe 兜底跳过 |
| 2026-09-01 | head 缺失期间 resume 完成的树 `_write_annotations` 写空标注 → 标注删减（9797→6665） | 待加保护（§9.1）；running 实例 resume 后修复代码可重新产出 |

---

## 9. 后续开发建议（按 ROI）

### 9.1 必做：`_write_annotations` 保护（防标注删减）
`run()` 中 head 缺失跳过扩展时，**不应调用 `_write_annotations`**（当前无条件调用，会把旧标注 DELETE 成空）：
```python
if root.gated:
    if self._messages_head:
        await self.process_annotations(root)
    else:
        logger.warning(...)   # 跳过扩展 → 不写标注（保留旧标注或留空均可，但别删）
self.status = "done"
if <实际执行了扩展>: self._write_annotations()
```

### 9.2 必做：清理历史失败记录
当前 DB 有 10,696 条 error 记录（759 实例）会被 resume 固化。清理后槽位变 None 自动补跑：
```sql
DELETE FROM rollouts WHERE error IS NOT NULL;
```
（执行前备份 state.db；8 个 done 实例的失败槽位删除后不再补跑——树已完成，可接受。）

### 9.3 建议：`load_node_rollouts` 把 error 视为 None
通用兜底：`load_node_rollouts` 读到的槽位若 `error` 非空 → 置 None（视为缺失）→ 无论历史还是未来，失败槽位都能被 resume 补跑，不依赖"不落库"约定。

### 9.4 可选：状态机快照恢复（严格接续）
`save_nodes` 已存 `visits / rollouts_json`；可扩展存 `visited_flags / best_entries / expanded / leaves / iteration`，resume 时恢复 → 严格接续而非重新决策。收益主要是省"重新决策"成本（对 PRM 质量提升有限，正确性已由 rollout 复用保证）。

### 9.5 可选：probe 全失败处理
`locate_error` 中 probe 节点 `n_rollouts == 0`（全失败）与"全错"（MC=0）应区分——全失败不应记 leaf，应重试或跳过该分支。

### 9.6 测试补充
- kill -9 模拟中断 → `--resume` 验证 rollout 复用、缺失补跑、标注幂等；
- 失败重跑（worker 耗尽 + 树级重跑 + resume 补跑）链路的端到端用例；
- `_write_annotations` 保护用例（head 缺失时不删旧标注）。

---

## 10. 一句话总结

**容错三支柱**：单点失败三层重试 + 失败不落库（resume 补跑）、进程中断 `--resume` 复用已算 rollout（LLM 零浪费）、预算熔断保留部分数据；**当前边界**：树选择-定位状态机不重建（resume 重新决策，对 PRM 质量影响低）、历史失败记录需清理、`_write_annotations` 需加防删保护。
