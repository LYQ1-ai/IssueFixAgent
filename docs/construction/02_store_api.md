# 施工文件 02 —— StateStore 操作层 API（v5）

> 配套：01 文档（schema）；04 文档（调用方上下文）。本文件给出 `mcts/store.py` 的**方法全集**：
> 签名、语义、SQL 概要、调用方、线程/事务约定。实施时以本文件为规格。

## 1. 连接与线程模型（保留现状）
- `sqlite3.connect(path, check_same_thread=False)`；`PRAGMA journal_mode=WAL`；`PRAGMA synchronous=NORMAL`；
- 单写锁 `threading.Lock`（`_write_lock`）：所有写操作持锁；读操作也持锁（简单起见，量小可接受）；
- `StateStore.__init__` 内 `executescript(_SCHEMA)` + `_migrate_v5()` + commit；
- `close()` 幂等。

## 2. 迁移：`_migrate_v5()`
- 全新库：直接建 v5 三表（01 §2 DDL）；
- 旧库探测：若存在 `annotations` 表或 `instances` 表（旧名）且无 `tree_instances` → **判定为旧库**：
  1. 抛错提示使用 `run_mcts --fresh`（备份+重建），**不自动迁移**（语义不可靠还原，见 00 §6）；
- `run_mcts --fresh` 流程：`shutil.copy(state.db → state.db.bak.<ts>)` 后删除旧文件重建。

## 3. 方法全集

### 3.1 树（tree_instances）
| 方法 | 签名 | 语义 / SQL 概要 | 调用方 |
| --- | --- | --- | --- |
| `create_tree` | `(instance_id: str, messages_head: list[dict]) -> None` | INSERT OR IGNORE，status=not_started、n_rounds=0、head=json | run_mcts 注册 / TreeDriver 激活兜底 |
| `get_tree` | `(instance_id: str) -> Optional[dict]` | 返回 `{status, n_rounds, messages_head(list), root_mc}` 或 None | TreeDriver.run |
| `set_tree_status` | `(instance_id, status: str, *, root_mc: float|None = None) -> None` | UPSERT status(+root_mc)，updated_at=now | 激活/熔断/收束 |
| `increment_rounds` | `(instance_id: str) -> None` | `UPDATE tree_instances SET n_rounds=n_rounds+1` | 仅供审计/独立调用；正常走 `commit_session` |

### 3.2 节点（nodes）
| 方法 | 签名 | 语义 / SQL | 调用方 |
| --- | --- | --- | --- |
| `ensure_node` | `(instance_id, node_key, prefix_steps: list[Step]) -> bool` | INSERT OR IGNORE(status=rollout, prefix_json, in_pool=(node_key=='root'))；返回是否新建 | `_get_or_create_node` |
| `get_node_row` | `(instance_id, node_key) -> Optional[dict]` | 查 status/mc/visits/in_pool/prefix_json | 调试/审计 |
| `set_node_ready` | `(instance_id, node_key, *, mc_score: float, n_rollouts_cached: int) -> None` | `UPDATE nodes SET status='ready', mc_score=?, updated_at=now` | `_perform_rollouts` 结算点（⑤） |
| `set_node_in_pool` | `(instance_id, node_key) -> None` | `UPDATE nodes SET in_pool=1`（仅会话提交内调用） | `commit_session` |

> 说明：`mc_score` 缓存按"不信任、可重算"处理；load 时由 rollouts 行重算并回写（见 3.5 `load_tree_state`）。
> `n_rollouts_cached` 参数仅为报告便利，不参与判定（判定一律数 rollouts 行）。

### 3.3 rollout（rollouts）
| 方法 | 签名 | 语义 / SQL | 调用方 |
| --- | --- | --- | --- |
| `upsert_rollout` | `(result: RolloutResult) -> None` | 保留现状：幂等 UPSERT 单条；`is_consumed` 不覆盖（ON CONFLICT 时保留旧值） | Worker 完成回调 |
| `load_node_rollouts` | `(instance_id, node_key, n: int) -> list[Optional[RolloutResult]]` | 保留现状：按 idx 槽位返回（缺失=None） | `_perform_rollouts` / 载入 |
| `load_node_rollout_rows` | `(instance_id, node_key) -> list[dict]` | 原始行（含 is_consumed/consumed_iteration/聚合列），载入用 | `load_tree_state` |
| `count_node_rollouts` | `(instance_id, node_key) -> int` | `SELECT COUNT(*)`（判定缺槽的真值来源） | 恢复/补跑判定 |
| `is_rollout_consumed` | `(instance_id, node_key, rollout_idx) -> bool` | 单条查询 | 调试/审计 |

> `upsert_rollout` 幂等语义必须**保留已提交的 is_consumed/consumed_iteration**（会话提交后同槽位不会重写结果，但防御性保留旧值）。

### 3.4 会话提交（★核心，单事务）
```python
def commit_session(
    self,
    instance_id: str,
    node_key: str,            # 被消费 rollout 所属节点
    rollout_idx: int,
    *,
    visits: int,              # 该节点提交后的 visits（含本次 +1）
    increment_rounds: bool,   # 非 root 会话为 True
    n_rounds: int,            # 提交后的 n_rounds（increment_rounds=True 时有效）
    expanded: list[str],      # 本会话 expanded 探针的 node_key 列表
) -> None:
    """会话成功结束的原子提交（对应事件⑥，01 §4）。

    单个事务内执行：
      UPDATE rollouts SET is_consumed=1, consumed_iteration=? WHERE (iid,node_key,idx)
      UPDATE nodes SET visits=? WHERE (iid,node_key)
      UPDATE nodes SET in_pool=1 WHERE (iid, node_key IN expanded)
      (increment_rounds) UPDATE tree_instances SET n_rounds=? WHERE instance_id=?
    任一步失败整体回滚 —— 崩溃在事务内 = 全部未提交 = 会话可重做。
    """
```
调用方：`TreeDriver.process_annotations`（04 文档 §4）；单写锁 + `BEGIN IMMEDIATE … COMMIT`（或依赖单个 `executemany` + commit 的原子性，实施时用显式事务）。

### 3.5 整树载入（resume / 激活用）
```python
def load_tree_state(self, instance_id: str) -> dict:
    """返回 {tree: dict, nodes: {node_key: node_row}, rollouts: {node_key: [row...]}}。

    - nodes：全部行（status/mc/visits/in_pool/prefix_json，按 created_at 排序 = 池顺序）；
    - rollouts：每节点按 rollout_idx 升序；
    - 不做任何判定（mc 重算/补跑由 TreeDriver 负责）。
    """
```
调用方：`TreeDriver.run`（03 文档 §3）。行数级：单树节点通常 ≤ 几十，整树加载可接受。

### 3.6 聚合 / 报告（适配）
| 方法 | 语义 |
| --- | --- |
| `tree_summaries() -> list[dict]` | 每树：status / n_rounds / root_mc / n_nodes / n_rollouts / n_correct / n_consumed / avg_reward（nodes×rollouts 聚合，替代旧 `instance_summaries`） |
| `leaf_summaries() -> list[dict]` | 每实例 leaf 派生统计：`非 root ∧ mc==0` 节点数（替代旧 `annotation_summaries`） |
| `counts() -> dict` | tree/nodes/rollouts 行数（保留） |
| `close()` | 保留 |

### 3.7 删除的旧方法（实施时移除，避免死代码）
`save_nodes`、`write_annotations`、`instance_summaries`、`annotation_summaries`、`get_instance_status`（语义并入 `get_tree`）、`set_instance_status`（并入 `set_tree_status`）。

## 4. 事务与并发约定（实现守则）
1. 所有写方法持 `_write_lock`；`commit_session` 必须在锁内完成全部 UPDATE + 一次 commit；
2. `upsert_rollout` 的 ON CONFLICT **不得覆盖** is_consumed/consumed_iteration（`excluded` 不含这两列）；
3. 读（load_*）也持锁——保证与并发写不交错；批量载入走单次 SELECT（不用 N+1）；
4. WAL 下 reader/writer 不互斥，锁只串行化本进程写者（多进程打开同一 DB 由 WAL 兜底，本期不承诺多进程写）。

## 5. 对现有调用点的兼容清单
| 现状调用 | 改为 |
| --- | --- |
| `set_instance_status(iid,"running"/...)` | `set_tree_status` |
| `get_instance_status(iid)` | `get_tree`（status 字段） |
| `save_nodes(...)`（每轮 + 收束） | 删除；节点状态实时写（ensure_node/set_node_ready/commit_session） |
| `write_annotations(...)`（收束 DELETE+INSERT） | 删除（标注派生，见 00 D8） |
| `load_node_rollouts` | 保留（缺槽补跑） |
