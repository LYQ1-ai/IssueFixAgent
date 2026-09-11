# PRM 训练计划书（M4）v3 —— 编码实施蓝本

> **定位：本文是后续 Agent 编码的直接参考**。按"数据流顺序 = 阅读顺序 = 实现顺序"组织，
> 每个环节写清：做什么、逻辑是什么、参考哪里。讨论性内容已删除（历史版本见 git）。
>
> 已定决策：打分 = **verdict token**（非思考模式）；标签 = **混合 soft**（0.8·binary+0.2·soft）；
> prompt = **PRM 专用 system/user 重设计**；训练模型 = **仅 Qwen3.5-4B**；
> 训练环境 = 独立 conda `CodeAgentRL-PRM`。
>
> 硬约束：`outputs/batch500/state.db`（~5GB）**严格只读**，mcts 引擎代码不改。

---

## 0. 已核实事实（编码依据，全部实测）

| # | 事实 | 用途 |
| --- | --- | --- |
| F1 | Qwen3.5-4B：词表 248,077，`tie_word_embeddings=True`（embedding 与 lm_head 共享矩阵） | 打分实现、冻结策略 |
| F2 | `Correct`=31995、`Incorrect`=39130，均为单 token；备选 `Yes/No`、`对/错` 亦单 token | verdict token 选型 |
| F3 | chat template（jinja L149-150）：`enable_thinking=False` 时生成提示渲染 **`<|im_start|>assistant\n<think>\n\n</think>\n\n`——开闭两个 think 标签都在**（完整空 think 块）；think 模式则只有 `<think>\n`（闭标签由模型生成） | 打分位置定义，**无需补闭合标签** |
| F4 | nothink 渲染时**历史消息中的 reasoning_content 完整保留**（渲染为 `<think>...</think>`） | 轨迹上下文无损 |
| F5 | 轨迹 assistant 消息 100% 带 `tool_calls`（命令在 `function.arguments`）、100% 带 `reasoning_content`、99% content 近空；tool 消息 content 已含 `{"returncode":..,"output":..}`；另有 `function_call`/`provider_specific_fields`/`extra` 冗余字段 | 规范化白名单 |
| F6 | 非 root 节点重算 MC 分布（n=21,294）：0.0→11.3%、0.2→10.6%、0.4→12.7%、0.6→16.6%、0.8→22.7%、1.0→26.0%；**MC∈{0.4,0.6} 边界占 29.2%**；train/dev/test 分布形状一致 | 标签设计、评估重点 |
| F7 | 本地模型：`/media/shared_e/models/Qwen3.5-4B` | 训练基座路径 |

参考文件索引：

| 参考什么 | 去哪里看 |
| --- | --- |
| DB 三表 schema、字段语义、mc_score=缓存可重算 | `docs/construction/01_database_design.md` |
| Step 解析 / 前缀消息 / node_key 内容寻址 | `mcts/steps.py`（`Step.from_json`、`prefix_node_key`） |
| MC = 正确 rollout 占比（Eq.3）、leaf 语义 | `mcts/node.py::MCTSNode.compute_mc`、`mcts/locate.py` |
| ReARTeR 原始 label 做法（纯二值 MC>0.5，本项目增强为混合 soft） | `ref_papers/ReARTeR/PRM_Data/process_to_prm_data.py` |
| 总体进度与 M4 上下文 | `Overall_PLAN.md` §5/§7 |

---

## 1. 总体数据流与代码结构

```text
outputs/batch500/state.db ─┐
outputs/mcts/splits.parquet ─┴─→ prm/raw.py（原始加载，纯函数）
                                         ▼
                    prm/build_dataset.py::PRMDatasetBuilder（遍历节点→样本→标签→去重）
                                         │  调用 prm/preprocess.py::TrajectoryPreprocessor
                                         │  （规范化 / 组 prompt / 渲染计长）
                                         ▼
              outputs/prm/{train,dev,test}.parquet + length_report + manifest
                                         │  依据 length_report 确定 max_length
                                         ▼
        prm/data.py（collator + 截断） + prm/modeling.py（VerdictScorer）
                                         │
                        prm/train_prm.py（Trainer 微调）──→ outputs/prm/runs/<run>/
                                         │
                        prm/eval_prm.py（前向评估 / best-of-5 / 报告）
```

新增文件清单（全部为新增，不改 mcts/）：

| 文件 | 职责 |
| --- | --- |
| `prm/raw.py` | 原始数据加载纯函数（DB 只读、parquet 读取） |
| `prm/preprocess.py` | `TrajectoryPreprocessor` 类：步解析、消息规范化、prompt 组装、渲染计长 |
| `prm/prompts.py` | PRM prompt 模板常量（版本化） |
| `prm/labeling.py` | 标签纯函数（binary/soft/混合/leaf 链/类别权重） |
| `prm/build_dataset.py` | `PRMDatasetBuilder` 类 + CLI：编排构建、长度报告、manifest |
| `prm/data.py` | `VerdictCollator`：训练期截断 + 张量化 |
| `prm/modeling.py` | `VerdictScorer`、verdict id 解析 |
| `prm/metrics.py` | AUC/F1/Brier/ECE/分桶 |
| `prm/train_prm.py` | 训练 CLI（Trainer 子类） |
| `prm/eval_prm.py` | 评估 CLI（含 best-of-5） |
| `prm/probe.py` | M4.0 零训练侦察 CLI |
| `config/prm.yaml` | 全部配置 |
| `test/test_prm_{raw,preprocess,build,data,model,eval}.py` | 单测 |

### 1.1 仓库与目录

仓库根：`/home/lyq/PycharmProjects/CodeAgentRL`（git 远端 `github.com/LYQ1-ai/IssueFixAgent`，
主分支 `main`；PRM 相关改动按模块分 commit 提交）。全部新增文件：

```text
prm/                             # 全部新代码（Python 包）
  __init__.py
  raw.py                         # §2 原始数据加载纯函数
  preprocess.py                  # §3 TrajectoryPreprocessor
  prompts.py                     # §4 prompt 模板常量
  labeling.py                    # §5.2 标签纯函数
  build_dataset.py               # §5.3 PRMDatasetBuilder + CLI
  data.py                        # §7.2 VerdictCollator（截断+张量化）
  modeling.py                    # §7.1 VerdictScorer
  metrics.py                     # §9 指标与分桶
  train_prm.py                   # §8 训练 CLI
  eval_prm.py                    # §9.2 评估 CLI
  probe.py                       # §9.1 零训练侦察 CLI
config/
  prm.yaml                       # 全部配置（新增）
scripts/
  run_prm.sh                     # 可选：一键串联 build → probe → train → eval
test/
  test_prm_{raw,preprocess,build,data,model,eval}.py   # §10 单测
outputs/prm/                     # 运行产物（不入 git）
  {train,dev,test}.parquet  manifest.json  build_report.md
  length_report.md  probe_report.{json,md}
  runs/<run>/  eval_report.{md,json}  predictions.parquet
```

元数据改动：`pyproject.toml` 的 `[tool.setuptools.packages.find].include` 加 `"prm*"`
（`pip install -e . --no-deps` 后可用 `python -m prm...`）。

**禁止改动**：`mcts/**`、`agent/**`、`scripts/run_rollout.sh`、
`outputs/batch500/state.db`、`outputs/mcts/*.parquet`、`docs/construction/*`。

### 1.2 Python 环境（两个 conda 环境分工）

| 环境 | 状态 | 跑什么 | 依赖操作 |
| --- | --- | --- | --- |
| `CodeAgentRL` | 已存在（Python 3.12） | ① `build_dataset`（不传 `--tokenizer` 时走 chars/token 近似）；② 全量回归 `python -m pytest mcts/tests/ test/ -q`（保持 266+128） | **严禁安装 torch/transformers/peft**（`openai==2.54.0`/`litellm==1.97.0` 已手工锁定，勿动）；pandas/pyarrow 已具备 |
| `CodeAgentRL-PRM` | **新建**（Python 3.12） | 精确 token 统计、`probe/train/eval`、PRM 单测 | `conda create -n CodeAgentRL-PRM python=3.12 -y` → `pip install torch transformers peft accelerate pandas pyarrow scikit-learn pyyaml`；装完 `pip freeze > outputs/prm/requirements-freeze.txt` 留档 |

- 模型权重：`/media/shared_e/models/Qwen3.5-4B`（本地已有，无需下载）；
- GPU：训练/评估用 GPU1（A800 80G）；**运行前 `nvidia-smi` 确认显存空闲，若被占用
  （如 sglang 服务在跑）先询问用户，不自动停任何服务**；
- `sglang` conda 环境（GPU 推理服务用）与 M4 无关，不安装、不改动。

---

## 2. 原始数据加载（`prm/raw.py`）

纯函数，供构建器与评估复用。DB 打开方式（硬约束）：

```python
def open_db_readonly(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn
```

| 函数 | 返回 | 说明 / 参考 |
| --- | --- | --- |
| `load_tree_status(conn)` | `{instance_id: status}` | 只取 `done` 树；status 枚举见 `docs/construction/01` §3.1 |
| `load_instance_heads(conn)` | `{instance_id: messages_head_json 原文}` | 2,000 行；head user 内容含 issue+工具约定（§4.2 引用） |
| `load_mc_table(conn)` | DataFrame`[instance_id, node_key, mc, n_rollouts]` | **见 §2.1**（为什么重算）；SQL：`SELECT instance_id, node_key, AVG(CAST(correct AS REAL)) AS mc, COUNT(*) AS n FROM rollouts GROUP BY 1,2` |
| `iter_nodes(conn)` | `Iterator[dict]` | `SELECT instance_id, node_key, prefix_json, visits, in_pool FROM nodes`，cursor 逐行（注意只选需要的列） |
| `load_splits(path)` | DataFrame`[instance_id, repo, prm_split]` | `outputs/mcts/splits.parquet` |

### 2.1 为什么要重算 MC（原 §3.3 的用途说明）

`nodes.mc_score` 是**结算缓存**（`docs/construction/01` §3.2 明确"可由 rollouts 重算"），
`rollouts.correct` 才是事实。构建标签时**不信任缓存，从 rollouts 重算**，原因：

1. 标签的权威来源必须是"5 次续跑的成败计数"这一原始事实，缓存可能因旧口径或
   异常中断而陈旧；
2. 重算顺带得到 `n_rollouts`，用于 `--min-rollouts 3` 过滤（有效证据不足的节点
   不进训练集，保留在审计报告）；
3. leaf 判定（`leaf := node_key != "root" AND mc == 0.0`，语义见
   `mcts/locate.py`——"从这里续跑 5 次全失败"的状态）必须基于重算值；
4. 重算值与缓存差 >1e-6 的节点**跳过**并列入构建报告（异常信号）。

---

## 3. 数据预处理（`prm/preprocess.py::TrajectoryPreprocessor`）

所有"原始数据 → 可渲染 messages"的逻辑集中在这一个类（可多进程实例化）：

```python
class TrajectoryPreprocessor:
    def __init__(self, tokenizer_path: str | None): ...

    def parse_steps(self, prefix_json: str) -> list[Step]:
        """prefix_json -> [Step]；直接用 mcts/steps.py::Step.from_json，不要重写解析。"""

    def canonical_messages(self, steps: list[Step]) -> list[dict]:
        """步序列 -> 规范化消息序列（assistant+tail 展开），规则见 §3.1。"""

    def build_messages(self, head_user: str, steps: list[Step]) -> list[dict]:
        """组装 PRM 全 prompt（§4），返回 [system, user, *trajectory, user] 。"""

    def rendered_tokens(self, messages: list[dict]) -> int:
        """apply_chat_template(add_generation_prompt=True, enable_thinking=False)
           后 tokenize 计数（含 nothink 生成提示）。无 tokenizer 时用 3.5 chars/token 近似并在报告标注。"""
```

### 3.1 消息规范化规则（依据 F5 实测）

| 消息 | 保留 | 丢弃 |
| --- | --- | --- |
| assistant | `role, content, reasoning_content, tool_calls[{id,type,function{name,arguments}}]` | `function_call`、`provider_specific_fields`、`extra`（actions/response/cost/timestamp）、tool_calls 内的 `index` |
| tool | `role, content, tool_call_id`（content 已含 returncode+output） | `extra`（raw_output 等冗余） |
| user 反馈（tail 中的 FormatError user 消息） | `role, content` | `extra` |

理由：assistant 的**动作本体在 tool_calls 里**（F5：100% 覆盖，content 近空），
tool 的 content 已是 observation 全文；reasoning_content 是 agent 思维链（100% 覆盖），
既是上下文也是后续升级 rationale 方案的原料，保留。

### 3.2 步与被判定步

一步 = `mcts/steps.py` 定义的 `Step`（assistant 动作 + tool observation tail）。
**每个训练样本的被判定步 = 前缀最后一步**（标签语义 = "执行完这一步后继续能否成功"，
与 MC 的测量方式一致，见 §5.1）。

---

## 4. PRM Prompt 设计（`prm/prompts.py` + `preprocess.build_messages`）

### 4.1 渲染结构

```text
[system]              PRM 角色定义（新设计）
[user]                任务上下文 + 轨迹说明（引用 head user 原文）
[assistant/tool ...]  规范化后的轨迹前缀（含被判定步）
[user]                末轮判定指令（新设计）
── apply_chat_template(add_generation_prompt=True, enable_thinking=False) ──
<|im_start|>assistant\n<think>\n\n</think>\n\n   ← 打分位置 = 这之后的最后 token（F3）
```

### 4.2 模板常量（英文，与轨迹/issue 一致；版本化，改动须重建 parquet）

```python
TEMPLATE_VERSION = "v1"
SYSTEM_PRM_V1 = (
    "You are a process reward model (PRM) for a software engineering agent. "
    "You will read a GitHub issue and the agent's execution trajectory "
    "(assistant messages contain the agent's reasoning and a bash tool call; "
    "tool messages contain the command output). "
    "Judge whether the agent is on track to fix the issue from its current state. "
    "Answer with exactly one word: Correct or Incorrect."
)
USER_CONTEXT_V1 = (
    "You are reviewing a coding agent's attempt to fix the GitHub issue below.\n\n"
    "<task_context>\n{issue_and_conventions}\n</task_context>\n\n"
    "The following conversation is the agent's trajectory so far."
)
USER_INSTRUCTION_V1 = (
    "The trajectory above ends with the agent's latest action and its observation. "
    "Based on the current repository state, judge whether continuing from here is "
    "likely to fix the issue. Respond with exactly one word: Correct or Incorrect."
)
```

规则：

1. `{issue_and_conventions}` = head user 内容**原文**（含 agent 的工具约定，无损）；
2. **严禁 gold 信息**（patch / gold locations）进入任何 prompt；
3. 构建期把完整 messages 烘焙进 parquet（单一代码路径）；manifest 记录
   `template_hash = sha1(prompts.py 全部常量拼接)` 与 tokenizer 版本。

---

## 5. 样本生成与标签（`prm/build_dataset.py::PRMDatasetBuilder` + `prm/labeling.py`）

### 5.1 核心概念：一个节点如何变成样本（先读这段再写代码）

- MCTS 的一个**节点** = "agent 执行到某步时的完整前缀状态"；节点的 `mc` =
  从该前缀续跑 5 次的成功率（`mcts/node.py::compute_mc`），是**实测**的状态质量；
- 一条**训练样本** = 把某个前缀渲染成 PRM 输入（§4），让模型给**前缀最后一步**打分；
  标签就是该节点的 mc（因为"最后一步执行完的状态"就是这个节点）；
- **非 leaf 节点**（0<mc<1 或 mc=1）：真实测量的状态 → 产 **1 条**样本，
  `binary = (mc>0.5)`，`soft = mc`；
- **leaf 节点**（非 root 且 mc=0，即续跑 5 次全失败）：二分定位（`mcts/locate.py`）
  认为该前缀**最后一步**是首个错误步 → 最后一步标 0；其之前的步骤还没有坏，
  链式回填标 1 → 该前缀展开成 **L 条**样本（第 i 条 = 前 i 步的前缀）；
- **root**：空前缀，没有步可打分 → 不产样本。

**具体例子**（一棵树的两次探针）：

```text
probe① [s1..s6]  MC=0.8（续跑 5 次 4 成功）      → 1 条样本：目标步 s6，label=0.96
probe② [s1..s9]  MC=0.0（续跑 5 次 0 成功，leaf）→ 9 条样本：目标步 s1..s9
                                                    标签 1,1,1,1,1,1,1,1,0
  └ 其中 i=6 的样本键 (instance, prefix[1..6], 6) 与 probe① 的真实样本撞键
    → 去重优先级：真实节点样本 > leaf 回填 → 保留 0.96 那条
```

### 5.2 标签函数（`prm/labeling.py`，纯函数）

```python
def node_label(mc: float) -> tuple[int, float]:
    return int(mc > 0.5), mc                    # (binary, soft)

def mixed_label(binary: int, soft: float, soft_w: float = 0.2) -> float:
    return (1 - soft_w) * binary + soft_w * soft    # 主训练标签

def leaf_chain_labels(n_steps: int) -> list[float]:
    return [1.0] * (n_steps - 1) + [0.0]

def class_weights(binary_labels: list[int], cap: float = 4.0) -> tuple[float, float]:
    n_pos = sum(binary_labels); n_neg = len(binary_labels) - n_pos
    return 1.0, min(cap, n_pos / max(1, n_neg))     # (w_pos, w_neg)
```

类别权重在 **train split 内**按 `label_binary` 统计一次（预期 ≈5.56:1 →
`w_neg` 触顶 4），写入 manifest 与 config，训练/评估共用。

### 5.3 构建器类（`PRMDatasetBuilder`）

```python
class PRMDatasetBuilder:
    def __init__(self, cfg: dict): ...

    def prepare(self):
        """调 prm/raw.py 加载 heads/mc/splits，过滤 done 树。"""

    def iter_samples(self) -> Iterator[dict]:
        """遍历节点（§5.1 规则）产出样本 dict：
           sample_id, instance_id, repo, split, node_key, step_index, step_count,
           messages(§4 组装), label, label_binary, label_soft, label_source,
           mc_score, n_rollouts, visits, in_pool, rendered_tokens"""

    def dedupe(self):  # 键 (instance_id, prefix_node_key, step_index)；
                       # 优先级 node_mc > leaf_chain，记录去重前后数量
    def write(self):   # 按 split 分写 parquet（zstd）；每 1024 条 flush writer
    def build(self):   # prepare → iter_samples → dedupe → write → 长度报告 → manifest
```

去重键中的前缀键直接用 `mcts/steps.py::prefix_node_key(prefix[:i])`（内容寻址，
与引擎一致，保证跨节点撞键能被识别）。

### 5.4 输出 schema（parquet）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sample_id` | string | 全局唯一 |
| `instance_id` / `repo` / `split` | string | |
| `node_key` | string | 来源节点 |
| `step_index` / `step_count` | int32 | 被判定步（1-based）/ 前缀长度 |
| `messages` | list<struct> | §4 全 prompt（struct: role/content/reasoning_content/tool_calls/tool_call_id） |
| `label` | float32 | 混合 soft，训练目标 |
| `label_binary` / `label_soft` | int8 / float32 | 加权统计与审计 |
| `label_source` | string | node_mc / leaf_chain |
| `mc_score` / `n_rollouts` | float32 / int16 | 评估相关性 / 审计 |
| `visits` / `in_pool` | int16 / int8 | 审计 |
| `rendered_tokens` | int32 | 渲染后总 token（nothink 计入） |

### 5.5 CLI

```bash
python -m prm.build_dataset \
  --db outputs/batch500/state.db \
  --splits outputs/mcts/splits.parquet \
  --output outputs/prm \
  --min-rollouts 3 \
  --leaf-chain --soft-label-weight 0.2 \
  --tokenizer /media/shared_e/models/Qwen3.5-4B \
  --workers 8 --overwrite
```

### 5.6 Split 防泄漏

只用 `splits.parquet.prm_split`；instance 不跨 split、repo 不跨 split；
构建后断言 train/dev/test 两两 instance 与 repo 交集为空（不通过直接报错）。

---

## 6. 长度分布 → 确定 max_length（独立步骤，先于训练）

**构建完成后、训练开始前**，对**全部样本**执行（builder 的 `length_report`）：

1. 全量渲染计长（`rendered_tokens` 已在构建时算好，此处聚合）；
2. 产出 `outputs/prm/length_report.md`：p50/p75/p90/p95/p99/max，
   8K/16K/32K 覆盖率，及各候选 max_length 下的**假想截断率**（按 §7.2 截断规则
   模拟，不真正截断）；
3. 依据报告**确定训练 max_length**（v1 抽样预期 16K 量级、覆盖率≈88.6%，最终以
   全量实测为准），写入 `config/prm.yaml` 后再开训；
4. 截断**不在构建期执行**：parquet 存全量 messages，训练/评估时 collator 按选定
   max_length 动态截断；后续调整 max_length 无需重建数据。

---

## 7. 模型与打分（`prm/modeling.py`）+ 训练期数据处理（`prm/data.py`）

### 7.1 verdict 打分（主形式）

```python
def resolve_verdict_ids(tokenizer, pair=("Correct", "Incorrect")) -> tuple[int, int]:
    # 逐个 encode，断言 len(ids)==1（F2），返回 (id_c, id_i)；结果写 manifest

class VerdictScorer(nn.Module):
    """backbone(Qwen3.5-4B) + PEFT LoRA。forward(input_ids, attention_mask) -> z (B,)
       z = logits[b, last_non_pad, id_c] - logits[b, last_non_pad, id_i]
       p = sigmoid(z)   # = 2 类 softmax(exp(l_c)/(exp(l_c)+exp(l_i)))
       verdict 两行 embedding/lm_head 权重 requires_grad=False（F1 tie 共享，
       冻结以免输出训练污染这两个 token 的输入语义）；梯度仅经 LoRA 入 backbone。"""
```

要点：

1. **verdict 形式 ≡ 方向冻结为预训练语义的线性头**：
   `z = (w_c − w_i)·h + (b_c − b_i)`，无需新增任何层/resize；
2. 渲染以 `add_generation_prompt=True, enable_thinking=False` 结尾——模板渲染的是
   **完整空 think 块 `<think>\n\n</think>`（开闭两个标签都有，F3 模板源码 L149-150）**，
   **不需要在训练时补 think 闭合标签**；打分位置 = 该序列最后一个非 padding token；
3. padding 全局**右侧** + attention-mask 取 last non-pad；单测断言
   "逐条 vs 批量（含 padding）分数一致"；
4. 禁用 `AutoModelForSequenceClassification`（last-token/pad 语义不可控），自写包装类。

### 7.2 训练期截断（`VerdictCollator`，max_length 由 §6 报告确定）

顺序：保 system+user 上下文（预算不足报错）→ 保末轮指令 → 从早到晚整步删除
轨迹中较早的步 → 插入 marker `[earlier steps omitted to fit the PRM context window]`
→ 仍超限按单条 tool content 尾部截断（保 returncode+前 512 token）。禁止截断 issue、
禁止丢末轮指令。统计写入 trainer_state。

### 7.3 损失（Trainer 子类 `compute_loss`）

```text
z  = model(input_ids, attention_mask)          # (B,)
y  = label（混合 soft，collator 传入）
L  = -mean( w_pos·y·logσ(z) + w_neg·(1−y)·log(1−σ(z)) )
w_pos=1.0, w_neg=manifest 中的统计值（cap 4）
```

- 全序列**无 LM loss**（不传 labels 给 LM head），`use_cache=False`；
- 无校准辅助项（soft 目标本身携带校准信号）。

---

## 8. 训练（`prm/train_prm.py`）

| 项 | 值 | 说明 |
| --- | --- | --- |
| 基座 | `/media/shared_e/models/Qwen3.5-4B`，bf16 | 仅此一个模型；后续有必要再扩展其他模型 |
| LoRA | r=16, alpha=32, dropout=0.05, bias=none, target=[q,k,v,o,gate,up,down]_proj | 唯一可训练参数（verdict 行冻结） |
| batch | per_device=1 × grad_accum=16 = 16 | |
| lr | 1e-4，cosine，warmup 0.05 | 单参数组（无 head） |
| epochs / max_grad_norm / weight_decay | 2 / 1.0 / 0.0 | |
| bf16 / grad ckpt / use_cache | true / true / false | |
| attn | flash_attention_2（不可用则 sdpa） | |
| seed | 42 | |
| **eval/save 间隔** | **1,000 steps**（≈ N_train/16≈2,300 步/epoch → 每 epoch 约 2 次评估、全程 4-5 次），`save_total_limit=2` | 避免频繁保存/评估拖慢训练；间隔随训练数据量在 config 调整 |
| 选优 / early stop | dev_auc / patience=3（按 eval 次数计） | |
| 产出 | `outputs/prm/runs/<run>/`：config.yaml、adapter_model.safetensors、tokenizer/、trainer_state.json、checkpoints/、manifest（含 verdict_token_ids、template_hash、w_neg、max_length） | best 由 dev_auc 决定 |

训练前检查：`nvidia-smi` 确认 GPU1 显存足够（4B+16K 峰值预估 <50GB，A800 80G）；
**若 GPU 被占用（如 sglang 服务在跑），报告并询问用户，不自动停任何服务**。

---

## 9. 评估（`prm/eval_prm.py`、`prm/probe.py`、`prm/metrics.py`）

### 9.1 M4.0 零训练侦察（训练前，`prm/probe.py`）

基座（未训练）+ 构建好的 dev 样本抽 500 条 → 批量前向取 P₂(correct) →
对 `label_binary` 算 AUC/Brier → `outputs/prm/probe_report.{json,md}`。
作用：免费验证 verdict 方向先验（预期 AUC 明显 >0.5 才说明
Correct/Incorrect 方向可 Hijack；≈0.5 时先换 `Yes/No`、`对/错` token 对复测）。

### 9.2 评估内容（dev/test，进程内批量前向，无生成）

| 块 | 内容 |
| --- | --- |
| 总指标 | Accuracy(p>0.5)、P/R/F1（正类=label_binary=1）、ROC-AUC（选型主指标）、PR-AUC、**Brier/ECE（主看）**、Log loss |
| **label_source 分桶** | node_mc vs leaf_chain 两桶的 AUC 差——**主验收项**（回填推定正样本占 66%，必须单独盯） |
| 位置/长度分桶 | 相对位置 10% 分桶、绝对步数 1/2/3/4/5/6-10/11-15/16-20/21+、上下文 ≤4K/4-8K/8-12K/12-16K/>16K、truncated 与否 |
| 与 MC 相关性 | test 真实非 leaf 节点：Pearson/Spearman(PRM p, mc) |
| 轨迹级 | root rollouts（`--db` 读 `result_json.steps`，按 §4 模板渲染逐步打分）；聚合 mean/min/last/discounted(γ=0.95) 与 correct/reward/submitted 相关 |
| best-of-5 | 每实例 ≤5 条 root rollouts；选择器 random/shortest/PRM-mean/PRM-min/PRM-last/oracle；指标 selected_reward/selected_correct/regret/win_rate；**差值报 bootstrap 95% CI**（test 仅 175 树 ≈875 条 root rollouts） |

产出：`outputs/prm/eval_report.{md,json}` + `predictions.parquet`。

---

## 10. 单元测试（全部离线，无 Docker/LLM/GPU/网络；`test/test_prm_*.py`）

| 文件 | 关键场景 → 断言 |
| --- | --- |
| `test_prm_raw.py` | 只读打开（query_only）；mc 重算 SQL 结果与手算一致 |
| `test_prm_preprocess.py` | Step 解析往返一致（对齐 `mcts/steps.py`）；规范化白名单（F5：留 tool_calls/reasoning，删 extra/function_call/provider_specific_fields/index）；build_messages 四段结构齐全、无 gold 泄漏；nothink 渲染结尾=完整空 think 块（F3 固化）、历史 think 保留 |
| `test_prm_build.py` | 非 leaf mc=0.8→1 条样本 label=0.96；mc=0.4→0.08；leaf len=3→1,1,0；len=1→仅负样本；root 不产样本；去重优先级 node_mc>leaf_chain；min_rollouts=3 跳过；split/repo 不跨 split（交集断言）；DB mtime/行数不变；manifest 字段齐全 |
| `test_prm_data.py` | 截断顺序（保上下文/指令、recent-first、marker）；右 padding 逐条 vs 批量分数一致；collator 张量形状 |
| `test_prm_model.py` | verdict id 单 token 断言（异常 token 对报错）；verdict 行 requires_grad=False；z=两 token logit 差；加权 soft-BCE 与手算一致且仅 verdict 位有 loss；padded 前后分数一致 |
| `test_prm_eval.py` | 指标与手算一致；分桶正确；best-of-N 选择与 regret；bootstrap CI；report JSON/MD 字段一致 |

回归：`conda activate CodeAgentRL && python -m pytest mcts/tests/ test/ -q`
（保持 266+128 不回归）；PRM 单测在 `CodeAgentRL-PRM` 环境跑。

---

## 11. 里程碑与验收

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| **M4.0 零训练侦察** | `prm/probe.py` | probe_report 产出；AUC≈0.5 时先换 token 对 |
| **M4.1 数据构建** | parquet×3 + manifest + build_report + **length_report** | DB mtime/行数不变；无泄漏；单测过；**依据 length_report 确定并记录 max_length** |
| **M4.2 训练冒烟** | 4B + 8K 跑 20 steps | loss 降且无 NaN；dev AUC 可算；显存峰值记录 |
| **M4.3 正式训练** | 4B + 定稿 max_length，≥1 epoch | dev AUC>0.70、F1>0.65；early/late 分桶差距记录；best 可复现加载 |
| **M4.4 评估报告** | eval_report + predictions | test-dev 差 ≤0.05 AUC；Spearman>0.35；best-of-5 含 CI；label_source 两桶差距报告；失败样本 ≥20 条抽查 |

未达标排查顺序：标签分布 → 截断率 → label_source 差距 → 加权/soft 权重 →
是否过拟合 dev → verdict token 对更换。

---

## 12. 风险与对策

| 风险 | 对策 |
| --- | --- |
| verdict 方向先验弱（Correct/Incorrect 预训练语义与任务不匹配） | M4.0 先侦察；换 token 对（Yes/No、对/错）复测 |
| 29.2% 边界标签噪声（F6） | 混合 soft 以 80% binary 权重缓冲；label_source/Brier 主验收 |
| 回填推定正样本占 66% | label_source 分桶盯防 |
| nothink 渲染依赖 template 行为 | F3 已从模板源码核实并单测固化；template_hash + tokenizer 版本入 manifest |
| tie_word_embeddings 下误训 verdict 行 | 冻结断言单测（test_prm_model） |
| GPU 显存冲突 | 训练前 nvidia-smi 检查；被占用则询问用户，不自动停服务 |
| 16K 覆盖不全（≈11% 截断，v1 抽样值） | §6 全量长度分布实测后定 max_length；截断率入报告 |
| M5 时策略模型迭代使 PRM 过时 | 记录版本对应；必要时增量重训（阶段 2 议题） |

---

## 13. 实现顺序（= 阅读顺序）

### 13.1 前置阅读（写代码前按序完成，三层递进）

**第一层：项目背景（回答"数据从哪来、每行数据是什么"）**

| 序 | 材料 | 重点 |
| --- | --- | --- |
| 1 | `Overall_PLAN.md` §1/§2 | 研究路线（MCTS 产过程标签 → PRM → KTO/DPO）与 M4 的位置 |
| 2 | `Overall_PLAN.md` §3.3/§4 | MCTS 引擎模块分工；四大设计模式：步定义 / 内容寻址 / 树逻辑 / 会话原子提交——理解 `nodes`/`rollouts` 每一行是什么 |
| 3 | `Overall_PLAN.md` §5/§7 | 2000 树批量的产物数字与已知遗留（10 个节点槽位<5 等）；M4 输入约定 |
| 4 | `docs/construction/01_database_design.md` | 三表 DDL 与字段语义：`prefix_json`=事实、`mc_score`=缓存可重算、`rollouts.correct`=事实、`is_consumed`=消费账本——§2 重算逻辑与 §9.2 best-of-5 的依据 |

**第二层：算法实现逻辑（直接读代码，标签与解析的出处）**

| 序 | 材料 | 重点 |
| --- | --- | --- |
| 5 | `mcts/steps.py` | `Step`（一步=assistant+tail）、`Step.from_json`（prefix_json 解析，`preprocess.py` 直接复用）、`prefix_node_key` 内容寻址（去重键直接复用）、`steps_to_messages` |
| 6 | `mcts/node.py::MCTSNode.compute_mc` | MC = 正确 rollout 数 / 有效 rollout 数；门控 0<MC<1（`gated`） |
| 7 | `mcts/locate.py::locate_error` | leaf 的来历：二分定位首个错误步，`leaf := 非 root 且 mc==0` ≈ 前缀末步是首个错误步——§5.1 链式回填标签的算法依据 |
| 8 | `mcts/reward.py` + `docs/reward_design.md` 第一部分 | `correct` 的来源：submit_locations 分层 Soft-F1，τ=0.6 判对错——理解标签里"对/错"的语义 |
| 9 | `mcts/tasks.py::TreeDriver`（浏览）+ `mcts/store.py` | rollout 落库时机、`is_consumed` 语义（best-of-5 为什么不排除 consumed 的 root rollouts） |
| 10 | `ref_papers/ReARTeR/PRM_Data/process_to_prm_data.py`（对照 `module.py::process_annotations`） | ReARTeR 原始做法：节点级样本 + 纯二值 MC>0.5；本文档的混合 soft 与 leaf 链回填是对它的增强——对照阅读防止口径走偏 |

**第三层：本文档**——§0 事实表（F1-F7，打分/渲染设计的实测依据）→ §3-§9（各模块规格）。

### 13.2 实现步骤

1. `prm/prompts.py` → `prm/raw.py` → `prm/preprocess.py`（含 §3 规范化/§4 组装）；
2. `prm/labeling.py` → `prm/build_dataset.py`（§5）+ `test_prm_{raw,preprocess,build}`；
3. 跑构建 → **length_report → 定 max_length**（§6）→ 人工抽检 20 条样本；
4. `prm/data.py` + `prm/modeling.py`（§7）+ `test_prm_{data,model}`；
5. `prm/probe.py` → M4.0 侦察；
6. `prm/train_prm.py` → 冒烟（8K, 20 steps）→ 正式训练；
7. `prm/metrics.py` + `prm/eval_prm.py` → M4.4 报告；
8. M5：导出候选轨迹（PRM 打分筛 rollout，导出格式阶段 2 再定）。
