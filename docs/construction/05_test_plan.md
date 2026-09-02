# 施工文件 05 —— 单元测试计划

> 配套：00 §4 改造范围；02/03/04 文档。原则：**先测存储层与驱动层（新语义），再回归纯逻辑/执行层（不动代码也补回归）**；
> 全部可离线运行（FakeExecutor 风格，不依赖 Docker/LLM，与现状 `mcts/tests` 一致）。

## 1. 测试资产现状与分层

| 层 | 现有文件 | 处理 |
| --- | --- | --- |
| 纯逻辑 | `tests/test_node.py` `test_locate.py` `test_steps.py` | 保留；locate/node 用例**补充"判定落库语义"断言**（见 §3） |
| 执行 | `tests/test_replay.py` `test_executor.py` `test_reward*.py` | 不动，回归即可 |
| 存储 | `tests/test_store.py` | **重写为 v5**（原用例针对 4 表快照语义，全部失效） |
| 调度 | `tests/test_tasks.py` `test_tasks_head.py` | 保留队列/预算/head 用例；**驱动会话流用例重写/新增**（test_tasks_v5.py） |
| 数据/CLI | `test/`（instances/splits/env/init_env/…）、`tests/test_report.py` | 不动；report 用例适配新聚合 |

新增测试文件：`mcts/tests/test_store_v5.py`、`mcts/tests/test_tasks_v5.py`、`mcts/tests/test_resume_v5.py`。
共用一个 `helpers.py`（FakeEnv + FakeExecutor + 内存/临时 DB 工厂，现状已有，扩展之）。

## 2. 存储层（test_store_v5.py）—— 每个方法一条主用例 + 边界

| 用例 | 断言要点 |
| --- | --- |
| schema 建表 | 三表存在；PRAGMA journal_mode=WAL；`_migrate_v5` 对旧库抛错提示 --fresh |
| create_tree/get_tree | 往返一致（head JSON 还原）；重复 create 幂等 |
| set_tree_status | 各状态写入；root_mc 更新；updated_at 变化 |
| ensure_node | 新建返回 True、重复 False；status=rollout；root 时 in_pool=1、其余 0 |
| set_node_ready | status→ready、mc 写入；不覆盖 prefix_json/visits/in_pool |
| upsert_rollout | 幂等覆盖结果列；**is_consumed/consumed_iteration 不被覆盖**（先提交会话再同槽重写，断言仍=1） |
| load_node_rollouts | 槽位往返；缺失=None；坏 JSON 当缺失 |
| count_node_rollouts | 成功行计数（失败无行） |
| commit_session 原子性 | ① 正常：is_consumed=1、visits、n_rounds、expanded.in_pool=1 同批生效；② **注入中途异常 → 整体回滚**（全字段未变，模拟"崩溃在事务内"）；③ increment_rounds=False（root）不增 n_rounds |
| load_tree_state | 返回 tree+nodes+rollouts；顺序=created_at；空树（仅 root）可用 |
| 并发写 | 多线程并发 upsert/commit 不丢行（单写锁语义） |
| counts/tree_summaries/leaf_summaries | 聚合正确；leaf 派生 = 非 root ∧ mc==0 |

## 3. 纯逻辑补充断言（test_locate.py / test_node.py）

| 用例 | 断言 |
| --- | --- |
| locate 三分支返回 | 沿用 ReARTeR 数值示例（11.2–11.5）：expanded/leaves 集合与顺序不变（回归） |
| probe 前缀构造 | probe 前缀 = 父前缀 + conf + left（内容寻址键正确性） |
| select 候选资格 | mc∉(0,1) 排除；is_consumed=1 排除；error rollout 排除；QU 数值回归 |
| select 确定性 | 相同池/visits/账本 → 相同 (node,idx)（resume 重选一致性的根基） |

## 4. 驱动/会话流（test_tasks_v5.py）—— 核心新语义

Fake 环境：`FakeRolloutExecutor`（现成）按 payload 前缀返回**确定性脚本化结果**（每前缀固定 correct 序列），从而 MC 可预期。

| 用例 | 场景 | 断言 |
| --- | --- | --- |
| 从零初始化 | 空 DB + 1 实例 | tree not_started→running→done；root 行存在；rollouts 落库数=N；n_rounds 正确 |
| root 不门控 | MC=0/1 | 无任何消费会话；tree done |
| 门控 + 单会话 | 0<MC<1 | locate 产 expanded → commit_session：is_consumed=1、visits+1、in_pool=1、n_rounds=1 |
| 多会话 / 池增长 | 连续 select | expanded 入池后可再被选；每 (node,idx) 只消费一次 |
| cap 上限 | rounds 超 cap | 超限轮不执行 locate；tree done |
| 会话中途"崩溃"模拟 | 注入异常于第 k 个 probe 结算前（单元内模拟进程死亡：不调用 commit_session） | DB：probe 行/部分 rollout 在；is_consumed=0；expanded 无 in_pool |
| 同一进程内继续 | 崩溃后重建 TreeDriver（同 DB） | select 复现同一 (node,idx)；重做会话；已完成 probe rollout 全复用（Fake 计数不重复执行）；缺槽补跑；最终标注集与"未崩溃"对照一致（确定性 Fake 下） |

## 5. 恢复（test_resume_v5.py）—— 端到端 resume 语义

| 用例 | 断言 |
| --- | --- |
| done 树跳过 | resume 后 skip=True，零写入 |
| failed 树跳过 | 同上 |
| running 树续跑 | 重建后从断点继续：前序已消费 (node,idx) 不被重选（账本生效）；n_rounds 从 DB 恢复 |
| budget_exhausted 续跑 | 预算恢复后缺槽补齐；已提交会话保留 |
| 根缺槽补跑 | root k<N → resume 先补 root 再门控 |
| 孤儿不污染池 | 中断会话的 0<mc<1 探针（in_pool=0）在重做前不被 select（用"该探针 QU 最高"的场景断言仍先选被中断会话） |
| 预算部分结算节点 | ready 但 k<N → 再次被探时补跑（不看 status） |
| 内容寻址跨运行 | 两棵不同实例/两轮运行产生相同前缀 → node_key 相同 → rollout 复用（sha1 稳定性） |

## 6. 执行/环境回归（不动代码）
- `test_replay.py`：drift 容忍/硬失败、prefix 无 user 兜底（现状用例）；
- `test_executor.py`：失败 → RolloutResult(error) 构造；reward 分层（现状用例）；
- `test_tasks.py`：TaskQueue 去重、Worker 重试、Budget 熔断、EnvFactory 节流（保留）；
- `test_tasks_head.py`：head 三函数（注册路径改为 build_messages_head 优先，补充"注册即写、此后不变"用例）。

## 7. 报告适配（test_report.py 更新）
- `tree_summaries / leaf_summaries` 喂入 build_report；
- 报告字段：statuses 分布、n_rounds、rollout 总数/correct/consumed、leaf 派生数、吞吐（现状字段保留）。

## 8. 集成/手动验收（A800，非单测）
1. `--dry-run`（FakeExecutor）：3 实例全链路冒烟；
2. 真实 rollout 小批量（1 实例 × N=5）+ 中途 `kill -9`：
   - 重启 `--resume` → 断言：done/failed 跳过、running 续跑、已提交会话零重放（rollouts 表行数只增补跑部分）、容器无泄漏（docker ps）；
3. `--fresh` 旧库备份 + 重建路径；
4. 验收数字对齐 fault 文档口径：resume 复用率（重跑 0 新样本 for 已提交部分）、失败不落库（error 行 =0）。

## 9. 优先级（实施顺序建议）
1. store_v5（含迁移/原子提交）→ 2. TreeDriver 会话流（test_tasks_v5 通过）→ 3. resume（test_resume_v5）→ 4. report 适配 → 5. CLI --fresh/注册路径 → 6. 集成 kill -9 手动验收。
