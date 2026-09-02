# 施工文件 04 —— 单树执行流程（select / locate / 会话原子提交 / 并发提交）

> 配套：00 D4/D5/D13；01 §4；02 §3.4。本文给出 `TreeDriver` 会话式主循环的完整逻辑规格。
> 与现状代码的差异集中在 `TreeDriver`；`locate.py/node.py` 算法本体不变（仅"判定落库"时机变化）。

## 1. 单树执行状态机（节点级）

```
节点诞生(ensure_node)   status=rollout(k=0)
        │ 第一次补跑完成（attempts resolved）
        ▼
      status=ready + mc 已算        （k 可 = N，或 <N：失败/预算部分）
        │ 后续：被 select 消费（内存标记 → locate 会话 → 会话提交 is_consumed/visits）
        ▼
      status 仍为 ready（不变）——节点状态只表达 rollout 完备性，消费是 rollout 级账本
```

不变量：
- 每树任意时刻 **≤1 个未提交会话**（驱动串行）；
- 池成员 = `root ∪ {in_pool=1}`；孤儿（in_pool=0）永不进池；
- 已提交消费不可逆（is_consumed=1 永不再选）。

## 2. 主循环：`TreeDriver.process_annotations(pool_nodes)`

```python
async def process_annotations(self, pool: list[MCTSNode]) -> None:
    while True:
        # —— select（纯内存判定，见 §3）——
        node, idx = self._select(pool)          # 跳过已消费(idx: is_consumed)与 mc∉(0,1)
        if node is None:
            break                               # 无候选 → done

        # —— 轮数上限（非 root 才计）：超限则本轮不再 locate ——
        if node.prefix_steps and self.rounds >= self.rounds_cap:
            break

        node.increment_visits()                 # 内存 +1（随会话提交落库）
        rollout = node.rollouts[idx]

        # —— locate 会话（产生/复用探针并逐个补跑；崩溃安全边界）——
        expanded, _leaves = await self._locate_session(node, rollout)

        # —— ★会话原子提交（唯一判定落库点）——
        inc_round = bool(node.prefix_steps)
        self.store.commit_session(
            self.iid, node.node_key, idx,
            visits=node.visits,
            increment_rounds=inc_round,
            n_rounds=self.rounds + (1 if inc_round else 0),
            expanded=[n.node_key for n in expanded],
        )
        if inc_round:
            self.rounds += 1

        # —— 内存池增长（expanded 入池；leaves 只留在节点行，查询期派生）——
        for n in expanded:
            if n not in pool:
                pool.append(n)
```

要点：
- **select 的内存标记**：选中即在本进程内把该 (node,idx) 记为"已消费"（防同一会话流内重选）；DB 的 `is_consumed` 在会话结束时提交——两者解耦（崩溃窗口由 DB 侧 is_consumed=0 兜底：resume 重选重做）；
- **cap 语义与现状一致**：超限那一轮不执行 locate（不产生新会话）；
- `expanded` 即 locate 返回的 0<mc<1 探针；`leaves`（mc==0 探针）不入池、不落标注——PRM 侧由 `非root ∧ mc==0` 派生。

## 3. select（选择函数）

保持 `node.py::select_best_node` 的 QU 公式与候选资格，**唯一变化：visited_flags 由内存位图改为 is_consumed 账本**：

```
候选(node, idx) 需要同时满足：
  node 在池中（pool 列表内）          —— 内存池 = root + in_pool=1（载入时已重建）
  0 < node.mc < 1                     —— 门控
  idx < len(node.rollouts) 且该 rollout 无 error（成功行才在内存）
  is_consumed[node_key][idx] == False —— 账本（载入时从 DB 读入，运行时内存标记）
QU = α^(1−mc)·β^(len(idx.rollout)/max_len) + c·√(Σpool.visits)/(1+node.visits)
取严格最大；无候选 → None
```

- 命中后**内存**置 consumed，返回 (node, idx)；会话完成才提交 DB（§2）；
- `visits` 计入 QU 的是**已提交值**（= 上次会话提交后的值 + 本会话内存 +1 前的值），保证与 DB 一致（会话崩溃 → visits 未提交 → resume 重算一致）。

## 4. locate 会话（二分 + 探针结算）

算法本体 = `locate.py::locate_error`（不变），驱动层（`TreeDriver._locate_session`）负责探针的**创建与补跑结算**，并在会话层保证原子性：

```python
async def _locate_session(self, node, rollout) -> tuple[list[MCTSNode], list[MCTSNode]]:
    return await locate_error(
        node, rollout,
        get_node=self._get_or_create_node,          # 内容寻址：ensure_node(DB) + 内存注册
        perform_rollouts=self._perform_rollouts,    # 探针 N 次 rollout（见 §5）
    )
```

`locate_error` 内部逐轮：
```
cur = rollout.steps; conf = []
while len(cur) >= 2:
    left, right = split_middle(cur)
    probe = get_node(node.prefix + conf + left)     # ① 内容寻址
    await perform_rollouts(probe)                    # ② 结算（复用/补跑）
    if probe.mc >= 1:   break                        # 左半全对 → 停（无角色）
    elif probe.mc > 0:  conf += left; cur = right; expanded.append(probe)
    else:               cur = left; leaves.append(probe)   # mc==0 → leaf 语义（不入池）
```

探针生命周期与 DB 的对应：
| 步骤 | DB 操作 | 性质 |
| --- | --- | --- |
| probe 诞生（get_node 未命中） | `ensure_node`（status=rollout, in_pool=0） | 事实，随时可中断 |
| probe 补跑 | 每条成功 `upsert_rollout`（is_consumed=0） | 事实 |
| probe 结算完成 | `set_node_ready`（mc 缓存） | 事实结算 |
| 会话成功结束 | `commit_session`：expanded 的 `in_pool=1` | **判定**（原子） |

> 崩溃在任一 probe 结算中 → 该会话未提交 → expanded 不落 in_pool → resume 重做会话时内容寻址复用这些 probe 的 rollout（缺槽补跑），完成后正常入池。**无孤儿污染、无重复标注（无标注表）。**

## 5. 并发任务提交（`_perform_rollouts` 与执行层契约）

`_perform_rollouts(node, kind)` 现状语义保留，补两点 DB 钩子：

```python
async def _perform_rollouts(self, node, kind="root"|"probe") -> None:
    # 0) 已在内存且已结算（rollouts 齐 + mc 已算）→ 直接返回（内容寻址复用）
    # 1) 缺槽 = 该节点 N 个槽位中未落库的 idx：
    #      载入 load_node_rollouts(iid, node_key, N) → None 槽 = missing（现状逻辑）
    # 2) budget.allowed_count 截断 → 提交 RolloutTask（root 优先级 0 < probe 1）
    #      N 条 asyncio.gather 并发 → Worker 池执行（全局信号量）
    #      Worker 内 executor.run 阻塞（容器+agent，经 asyncio.to_thread）
    # 3) 每条结果: 失败 → _retry_rollout 树级重跑 1 次 → 仍失败不落库
    #      成功 → store.upsert_rollout(res)          ← 事实即落
    # 4) 全部 attempts resolved → set_node_ready(mc)  ← 结算
    #     若节点为新探针且其 mc 由调用方(locate)消费 → 不在此提交 in_pool（等会话）
```

执行/环境层契约（**不动**，只列边界）：
- `RolloutTask`（instance_id, node_key, rollout_idx, kind, priority, payload）→ `TaskQueue`（按 task_id 去重幂等）→ `Worker._execute_with_retry`（重试 + 指数退避）→ `AgentRolloutExecutor.run`（`EnvFactory.create_env` → `ReplayRunner.run_free/run_probe` → `reward` 结算 → `destroy_env`）；
- 失败重试三层（现状）：Worker 重试 → 树级 `_retry_rollout` → resume 缺槽补跑；
- `EnvFactory` 并发创建节流（creation_concurrency）与容器"创建即用、用完即销毁"不变；
- Phoenix 追踪（可选）不变：每次 rollout 一个 trace。

## 6. 根节点特例
- root 由 `ensure_node(node_key='root', in_pool=1)` 建立（激活时兜底）；
- root 的首次 N 次 rollout 走同一 `_perform_rollouts`（kind=root，优先级 0）；结算后 `mc` 决定门控；
- root 被 select 消费时：不产 best、不计 n_rounds（`increment_rounds=False`），但仍提交 `is_consumed/visits`（root 消费必须落库——否则 resume 不知道 root 的哪些 rollout 已定位）。

## 7. 终止与收束
- 循环终止：无候选 或 rounds ≥ cap（§2）；根不门控（03 §3）→ 直接 done；
- 收束：`set_tree_status(done/failed/budget_exhausted, root_mc)`；不再写任何标注（派生）；
- 报告/PRM 导出另走 store 聚合（02 §3.6；00 §4.5）。

## 8. 正确性不变量清单（测试断言用）
1. 每树任意时刻 ≤1 个未提交会话；
2. 已提交会话的 (node,idx) 永不被再次 select；
3. 未提交会话的 expanded 探针 in_pool 恒 0（孤儿不进池）；
4. 任何节点缺槽（count < N）在下次被探到时补跑；失败不产生行；
5. 同一前缀 node_key 恒定（跨运行 sha1 稳定）；
6. 崩溃后 resume：done/failed 跳过；running/budget_exhausted 续跑且前序会话零重放（probe 复用），仅崩溃点缺槽产生新样本。
