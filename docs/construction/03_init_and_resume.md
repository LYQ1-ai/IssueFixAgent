# 施工文件 03 —— 初始化与恢复（无 DB 从零 / 有 DB resume）

> 配套：00 D2/D11；01 §5；02 §3.5。描述 `TreeDriver.run()` 的完整启动路径。

## 1. 启动路径总览

```
run_mcts main
  ├─ 载入 config / 选实例（select_instances，与现状一致）
  ├─ 新建 StateStore（--fresh 时先备份旧库）
  ├─ 注册缺失实例：对每个选中实例 create_tree(not_started, head)   ← 无 DB 从零 / 增量补注册
  ├─ MCTSPipeline.run(instances):
  │    每实例一个 TreeDriver.run() 并发（Worker 池 + 信号量限流，现状不变）
  │    TreeDriver.run = 见 §2/§3
  └─ 收尾：store.close() + 报告（report 适配新表）
```

## 2. 无数据库的从零初始化

前提：`state.db` 不存在或为旧库（--fresh 已备份重建）。逐实例：

```
TreeDriver.run(instance):
  row = store.get_tree(iid)
  if row is None:
      head = build_messages_head(instance, config_file)   # D10：注册即渲染（不依赖 root rollout）
      store.create_tree(iid, head)                        # status=not_started
      row = store.get_tree(iid)
  → 进入 §3 的激活流程（not_started 等同可激活）
```

- head 一旦注册不再变（probe 消息 = head + 前缀步骤）；
- 若 `build_messages_head` 失败（异常实例）→ 记 tree failed 并跳过（不阻塞整体）；
- `not_started` 树批量注册可在 Pipeline 前一次性完成（减少逐树建表开销），实现时放在 `MCTSPipeline.run` 前置一步。

## 3. 有数据库的 resume（激活流程）

```
TreeDriver.run(instance):
  row = store.get_tree(iid)
  if row is None:                        # 新实例（增量跑）
      head = build_messages_head(...); store.create_tree(iid, head); row = get_tree(iid)

  # —— 跳过判定 ——
  if row.status in ("done", "failed"):
      log skip; return {status: row.status, skipped: True}

  # —— 激活 ——
  store.set_tree_status(iid, "running")
  state = store.load_tree_state(iid)     # 整树载入（02 §3.5）

  # —— 重建内存树（内容寻址）——
  pool_nodes = _rebuild_pool(state)      # root(恒在池) + in_pool==1 的节点，按 created_at 排序
  # 每个节点: MCTSNode(prefix 由 prefix_json 还原)；
  #  rollouts 由行还原为 RolloutResult；is_consumed 记入内存账本（select 跳过用）
  #  mc: 不信任缓存 → 由 rollouts 行重算（或校验一致后采用）

  # —— 根节点前置补跑（门控前提）——
  root = ensure 根节点存在
  if count_node_rollouts(root) < N:
      await _perform_rollouts(root)      # lazy 补缺槽 → 结算 → set_node_ready
  root.mc = compute_mc(root)
  if root.n_rollouts == 0:               # 无任何有效数据
      set_tree_status(failed); return
  store.set_tree_status(running, root_mc=root.mc)   # root_mc 缓存

  # —— 门控 ——
  if not root.gated:                     # MC∈{0,1}：无判别价值
      set_tree_status(done, root_mc=root.mc); return

  # —— 决策循环（会话式）——
  try:
      await self.process_annotations(pool_nodes)     # 04 文档 §3
      set_tree_status(done, root_mc=root.mc)
  except BudgetExhausted:
      set_tree_status(budget_exhausted, root_mc=root.mc)   # 保留已提交数据，可续跑
  except Exception:
      log; set_tree_status(failed)                    # 单树失败不拖垮整体
```

## 4. 中断会话的恢复原理（为什么不需要二分状态）

- 崩溃发生在某次 (select+locate) **会话中途**：会话判定未提交 → 该 (node, rollout) 的 `is_consumed=0`；
- 该会话产生的探针节点行/rollout 行已落（事实）——但它们：
  - `in_pool=0`（会话没提交 → 不会误入候选池）；
  - `status` 可能是 rollout（未跑完）或 ready（跑完但无 add/入池标记）；
- resume 激活后重建池 = root + `in_pool=1` 节点 → **候选集与崩溃前该轮 select 时刻完全一致**（前序会话已提交、本会话未提交）→ select 确定性复现同一 (node, rollout) → **重做会话**；
- 重做时 locate 走原二分路径：每个 probe 内容寻址命中（同 node_key）→ `load_node_rollouts` 复用已落 rollout，只补在飞缺槽 → 新增随机样本仅限崩溃点（与"从未中断"统计等价，既定语义）；
- 因此**二分进行位置（confirmed/current_span）永不落库、永不恢复**——重做即恢复。

## 5. 恢复/补跑的通用规则（所有场景）

1. **缺槽判定只看 rollouts 行数**：`count_node_rollouts(iid, node_key) < N` → 需要补跑；**不看节点 status**（覆盖：创建未跑 k=0、失败缺槽、预算部分结算的 ready）；
2. 补跑为 **lazy**：只在 `_perform_rollouts` 被调用时发生（root 前置、locate 产探针、会话重做命中）——**不做 resume 全树预扫描**（避免给永远不会再被探到的孤儿节点白花 LLM）；
3. 每次补跑完成（attempts 全部 resolved）→ `set_node_ready`（status=ready + mc）；
4. 预算将尽：`Budget.allowed_count` 截断本批提交数；allowed=0 → 抛 `BudgetExhausted` → 树标记 budget_exhausted（已提交会话保留，缺失槽位待下次预算恢复后补）。

## 6. 状态机（树级）

```
        注册(无DB) ──▶ not_started
                        │ 激活
                        ▼
   budget_exhausted ◀── running ──▶ done（无候选 / n_rounds 达 cap / 根不门控）
        │ 预算恢复          └─▶ failed（根无有效 rollout / 异常）
        └──────────▶ running（resume 续跑，流程同 §3）
```

- `done/failed`：resume 跳过；如需重跑 → 显式清库重建（--fresh 或删实例行，本期不做版本化重跑）；
- `budget_exhausted`：resume 视为可续跑（§3 激活流程）。

## 7. 边界与校验
- **root 行存在性**：激活时若 nodes 无 root 行 → `ensure_node(root)`（幂等）；
- **重复激活**：同一实例不会并发激活（MCTSPipeline 每实例一个 driver）；
- **孤儿节点无害**：中断会话的 ready/rollout 孤儿不是池成员，只占行；会话重做时会复用或覆盖其语义；
- 激活完成后建议做一次一致性日志：`tree status / n_rounds / 池节点数 / 未消费 rollout 数`（对拍统计用）。
