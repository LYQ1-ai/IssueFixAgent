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
- GPU：**由仓库根 `.env` 的 `CUDA_VISIBLE_DEVICES` 决定**（2026-09-14 起；此前写死 GPU0/GPU1），
  见 §14.5；CUDA toolkit 由宿主 shell 提供，不归本项目管理；**运行前 `nvidia-smi` 确认显存空闲，
  若被占用（如 sglang 服务在跑）先询问用户，不自动停任何服务**；
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
   （**实现已收紧为强制 batch_size=1 / 无 padding，理由见 §14.2，务必先读**）
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

训练前检查：`nvidia-smi` 确认 **`.env` 选中卡**显存足够（4B+16K 峰值预估 <50GB，A800 80G）；
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

---

## 14. 实现状态登记（偏离项 / 未完成项）

> **本节由实现方维护**，记录与原计划的偏离、未完成项，以及实现期实测得到的新硬约束。
> 阅读顺序建议：**§14.2（padding 硬约束，最重要）→ §14.1（偏离）→ §14.3（未完成）→ §14.4（不可回归清单）**。
> 最近更新：2026-09-14。

### 14.1 与计划的偏离（含原因与依据）

| # | 计划原文 | 实际实现 | 原因 / 依据 |
| --- | --- | --- | --- |
| **D1** | §7.1 要点 3：padding 全局**右侧** + attention-mask 取 last non-pad | **强制 `batch_size=1`（不产生任何 padding）**；右 padding 仅为 API 兼容保留；`VerdictScorer.forward` 用 `logits_to_keep=1` 只算末位 logits | 见 **§14.2**：Qwen3.5 混合架构对 **pad 前缀**敏感，多样本批会改变打分。bs=1 时 `pad = max_len - len(s) = 0`，位置切片天然精确 |
| D2 | §7.1 要点 1：verdict 两行 `requires_grad=False` | 梯度 hook 逐行置零（`VerdictScorer._freeze_verdict_rows`） | PyTorch 的 `requires_grad` 是 **per-tensor**，无法只关矩阵中的两行；hook 置零后 Adam 动量恒 0，语义等价。LoRA 下 embedding 本就冻结，hook 不注册（早期误注册曾抛 `RuntimeError`） |
| D3 | §7.2 末句：截断统计**写入 trainer_state** | 独立文件 `outputs/prm/runs/<run>/truncation_stats.json` | 内容等价（`n_truncated / steps_dropped / tools_truncated / errors`），仅存放位置不同，不影响 M4.3 验收 |
| D4 | §5.2 `class_weights` 公式 `min(cap, n_pos / max(1, n_neg))` | 全正 / 空输入返回 `(1.0, 1.0)` | 全正时负类样本数为 0 → 加权 BCE 中 `(1−y)` 项恒为 0，`w_neg` 取值**不影响 loss**；空数据集会被构建器"构建结果为空"检查拦下。纯文档一致性问题 |
| D5 | §1.2 环境 = conda `CodeAgentRL-PRM` | 仓库内 **`.venv-prm`**（Python 3.12；torch 2.14.0+cu130；transformers 5.17.0；peft 0.20.0；swanlab 0.10.0；flash-linear-attention 0.5.2） | 隔离目标一致（`CodeAgentRL` 环境零污染，基线 348 测试保持）；但**可复现性受损**——照本节搭环境会得到不同版本组合，且缺 `requirements-freeze.txt`（见 U1） |
| D6 | §1.1 `scripts/run_prm.sh`（可选，一键 build→probe→train→eval） | `scripts/run_prm_m4_gpu.sh`（check/probe/smoke/train/eval/all 分步）+ `scripts/train_prm_m4.sh`（正式训练：预检 + 冒烟 + 训练 + 评估） | 功能覆盖更全：GPU/显存门槛、`template_hash` 一致性、run 目录占用、**断点续训 `--resume`**、`--background`、SwanLab |
| D7 | 未提及训练曲线记录 | `config/prm.yaml` 增加 `train.report_to: [swanlab]` / `train.swanlab.{project,workspace,mode,log_dir}`；`prm/train_prm.py::_setup_reporting` | 需求外新增。官方流程 `pip install swanlab` → `swanlab login` → 训练自动上报（实验名 = run 名）。**两个上游坑已规避**：① transformers 5.17 回调以 `get_run() is None` 判断初始化，而 swanlab 0.10 的 `get_run()` 无 run 时抛 `RuntimeError` → 必须由我们先 `init`；② `SWANLAB_PROJECT` 只接受 JSON 对象，普通字符串会 `SettingsError` → 改为只经 `init(project=...)` 传 |
| D8 | 未提及显存优化 | `forward` 传 `logits_to_keep=1`（Qwen3.5 支持，切片发生在 lm_head **之前**） | 全量 `(B,L,V)`：`8×16384×248044×2B ≈ 65GB`，真机 probe 直接 OOM（日志 `64353206272 bytes`）。改后 logits 仅 `B×1×V ≈ 4MB` |
| D9 | 未提及混合层内核 | `.venv-prm` 额外装 `flash-linear-attention` | 消除 `chunk_gated_delta_rule is falling back to its reference PyTorch implementation` 警告（参考实现慢且可能materialize 大中间张量）。**副作用见 §14.2-4** |
| **D10** | §7.2 截断第 1 步：预算不足**报错** | **跳过该样本并计数**（`VerdictCollator.fits` / `oversize_indices` / `collate_or_skip`） | 报错发生在 **DataLoader worker** 内 → 直接杀掉整个训练。2026-09-15 真机：19,862 条训练样本里**仅 1 条**（`rendered_tokens=43,769`，末步本身 31,230 > 16,384）在 step 1146 崩掉，白跑 10 h。该样本被判定步本身超出窗口 → 本来就不该用它训练。扫描结果按 parquet 指纹缓存（`outputs/prm/oversize_skip_<split>.json`），统计进 `skipped_oversize` / manifest。详见 §14.7 |

### 14.2 新硬约束：Qwen3.5 混合架构对 padding 敏感（真机实测）

**背景**：为省显存（D8）曾把 collator 改成**左** padding + 位置切片。真机实测证明该方案**错误**。

**实验一：三种摆放方式的对照**（真实 Qwen3.5-4B，同一 71-token 样本，CPU 参考内核）

| 配置 | z | σ(z) | 与单条的偏差 |
| --- | --- | --- | --- |
| A 单条（无 padding） | -1.3750 | 0.2018 | 基准 |
| B **左** padding + mask 取位 | -0.1562 | 0.4610 | **Δp = +0.259** ❌ |
| C **右** padding + mask 取位 | -1.3750 | 0.2018 | **Δz = 0.0000** ✅ |

**实验二：pad 数量扫描**（同一被判位置）

| 左补 pad 数 | 1 | 2 | 4 | 8 | 16 | 46 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Δp | -0.185 | **+0.652** | -0.134 | **+0.703** | +0.368 | -0.183 | +0.259 |

| 右补 pad 数 | 1 | 8 | 46 |
| --- | --- | --- | --- |
| Δp | 0.0000 | 0.0000 | +0.0209 |

左 padding 的偏差**剧烈且随 pad 数跳变**（同配置复测两次结果一致，是确定性偏差，不是随机噪声）。

**实验三：position_ids 不是成因**（同一 pad 数下三种写法几乎相同）

| npad=2 | 默认 `arange` | `cumsum−1` 截断（早期实现） | 严格递增（含负数） |
| --- | --- | --- | --- |
| Δp | +0.6540 | +0.6521 | +0.6521 |

**结论与机制**

1. 真正的变量是**被判位置的因果前缀里有没有 pad 步**。Qwen3.5 含 `causal_conv1d` 与 gated delta rule（线性注意力）等**顺序递推**层；源码 `apply_mask_to_padding_states`（`modeling_qwen3_5.py:237-244`）只把 pad 的 hidden 清零，**不足以让递推/分块内核真正跳过 pad 前缀**（零输入仍要走步数，分块累积因子在 bf16 下会放大误差）。
2. **右 padding 在因果上是安全的**：被判位置（最后一个真实 token）之前没有 pad，其前缀与单条完全一致 → 实测精确。
3. 早期把偏差归因于"位置编码整体偏移"的注释**已被实测否定**；`position_ids` 显式给出保留为防御性写法（对绝对位置类模型才必需）。
4. **`flash-linear-attention` 的副作用**：其内核是 triton 实现，**无 CUDA 时前向直接抛 `RuntimeError: 0 active drivers`**，不再回退参考实现（装饰器 `use_kernel_func_from_hub_with_fallback` 在**导入期**绑定 fla）。⇒ 装上 fla 后，**任何真实基座的前向（含 CPU 调试）都必须有可用 GPU**。

**影响面与强制措施**

| 环节 | batch | 有 padding？ | 状态 |
| --- | --- | --- | --- |
| 训练 | `per_device_train_batch_size=1` | ❌ 无 | ✅ 不受影响 |
| 训练中 dev 评估 | `per_device_eval_batch_size=1`（**必须显式设置**，默认 8 会引入 padding） | ❌ 无 | ✅ |
| M4.0 probe / eval_prm | `eval.batch_size=1` | ❌ 无 | ✅ |
| 轨迹级评估（best-of-5） | 同上 | ❌ 无 | ✅ |

实现层强制（写死在代码里，违反即报错）：
- `prm/data.py::VerdictCollator.__call__`：`len(batch) != 1` → `ValueError`；
- `prm/train_prm.py::train`：`per_device_train_batch_size != 1` → `SystemExit`；
- `prm/probe.py` / `prm/eval_prm.py`：`--batch-size != 1` → `SystemExit`；
- 回归护栏：`test/test_prm_model.py::TestRealModelGpu::test_real_model_right_padding_gather_is_exact`（真实架构上比较"右 padding 批 vs 单条"，若有人改回左 padding 或放开批大小会失败；tiny Llama 测不出这类问题——它没有卷积/线性注意力层）。

### 14.3 未完成项 / 修复状态（2026-09-14 更新）

状态图例：✅ 已修复（含测试）｜⏸️ 按用户指示暂缓｜⬜ 未开始。

| # | 项 | 计划出处 | 状态 | 落实方式 / 影响 |
| --- | --- | --- | --- | --- |
| **U1** | `requirements-freeze.txt` | §1.2 | ✅ | 已生成（94 项，含 torch/transformers/peft/swanlab/fla 版本）；`scripts/train_prm_m4.sh` 预检时自动刷新 |
| **U2** | `runs/<run>/config.yaml` | §8 产出表 | ✅ | `train_prm.train()` 写有效配置快照（含 `max_length` 覆盖与 `run.{name,smoke,resumed_from}`）；manifest 仍保留原字段 |
| **U3** | `truncated` 与否分桶 + `predictions.parquet` 列 | §9.2 | ✅ | collator 逐样本输出 `truncated` 标记 → `evaluate_samples` 落列 → `bucket_block_truncated` 产出 `not_truncated/truncated` 两桶 + **AUC 差**（`auc_gap_truncated_minus_clean`），报告新增该区块 |
| **U4** | best-of-5 每实例 ≤5 条 + `eval.best_of.k` 接线 | §9.2 | ✅ | 新增 `metrics.cap_candidates`（按 `rollout_idx` 升序取前 k，确定性）；`trajectory_eval` 读 config 的 `k`，报告输出候选数分布（`n_candidates.hist`）。当前数据分布：1999 实例×5 + 1 实例×4 |
| **U5** | length_report 的 §7.2 截断模拟 | §6 步骤 2 | ⏸️ | **按用户指示暂缓**；现仅上界（16K→4.80%）。U3 的分桶已能间接回答"截断是否伤指标" |
| **U6** | 人工抽检留档 | §13.2 步骤 3 | ✅ | 新增 `build_dataset.write_spot_check()`：构建后自动产出 `outputs/prm/spot_check.md`（split 分层、固定 seed、渲染 prompt 全文 + 标签/来源），并支持 `python -m prm.build_dataset --spot-check-only` 对已有 parquet 单独生成（已生成 20 条） |
| **U7** | 显存峰值记录口径 | §11 | ✅ | `run_manifest.json` 新增 `peak_gpu_memory.{allocated_gb,reserved_gb}`（`torch.cuda.max_memory_allocated/reserved`；无 CUDA 时为 `null`） |
| **U8** | GPU 里程碑 M4.0/M4.2/M4.3/M4.4 | §11 | ⬜ | 代码就绪、数据已产出；`pytest -k TestRealModelGpu` 已通过；**下一步跑 M4.0 probe** |
| **U9** | dev 评估成本 | §8 | ✅ | 新增 `train.eval_max_samples`（默认 `null`=全量）：设 N 则只评前 N 条（确定性），manifest 记录 `n_dev`/`n_dev_eval`/`eval_max_samples`。建议正式训练先设 500 对齐 probe 口径 |
| **U10** | fla 无 CUDA 硬崩的运维提示 | §1.2 | ⏸️ | **按用户指示暂缓**（用户将安装 CUDA 13.0）；机理记录见 §14.2-4 |
| **U11** | 训练吞吐：单步 25 s，全量训练不可行 | §8 | ✅ | **已结案：共享卡被抢占**（bench 复测 matmul 9.6 → 250 TFLOPS、倍数 3.47× 正常、单步 2.46 s ⇒ 1 epoch ≈ 10-13 h）。顺带落地 `train.use_kernels`（fail-open）+ 评估封顶 + manifest 留档；详见 §14.6 |
| **U12** | `eval_prm`/`probe` 未接线内核化 | §9 | ⬜ | `load_scorer_for_run` 仍是 `use_kernels=False`。纯速度项（不影响打分口径，bs=1 下 test 2.3k 条 ≈ 17 min、best-of-5 ≈ 1.5 h）；训练吞吐定型后再顺手加 `--use-kernels` |

### 14.4 不可回归清单（这些"看起来可以优化"的地方其实是坑）

1. **不要把 collator 改成多样本批**（或把 `per_device_eval_batch_size` 调回默认 8）——会引入 padding，打分被污染（§14.2 实验一/二）。
2. **不要改回左 padding**——哪怕配合 per-row gather，pad 前缀本身就会污染混合层（§14.2 实验一 B）。
3. **不要删掉 `logits_to_keep=1`**——会回到 65GB 全量 logits 并 OOM（§14.2 背景 / D8）。
4. **不要删掉显式 `position_ids`**——虽然实测对 RoPE 等价，但它是防御性写法，且对多模态 3D 位置路径（`compute_3d_position_ids` 会读 attention_mask）有意义。
5. **不要在没有 GPU 的环境跑真实基座前向**（装了 fla 之后）——见 §14.2-4。
6. **不要把 `gradient_checkpointing` 改成 `false`**——bench 的 12.9 GB 峰值不可外推：
   真实训练（16K/bs=1/32 层）第 1 步就 OOM（实测 78.72 GiB，§14.6.3）。它是硬约束。
7. **不要拿 `scripts/bench_prm_step.py` 的峰值显存判断训练能不能跑**——它只有 1 个样本、
   1 步、无优化器状态；显存结论只对"相对比较"有效（哪个配置更省），不对"绝对值"背书。

### 14.5 环境变量约定：GPU 选择（2026-09-14 起）

**用哪张卡 = 仓库根 `.env` 的 `CUDA_VISIBLE_DEVICES`**（唯一由本项目管理的环境变量）：

| 变量 | 作用 | 约定 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 用哪张（哪几张）卡 | 单卡写卡号（如 `1`），多卡逗号分隔。装载后**进程内 `cuda:0` 即选中卡**，故 CLI 一律用默认 `cuda`，**不要再写 `cuda:1` 这类物理卡号** |

- **装载时机**：各 CLI 入口（`python -m prm.{build_dataset,probe,train_prm,eval_prm}` 的 `main()`）
  开头调用 `prm.env.load_project_env()`，早于任何 torch/CUDA 初始化（`CUDA_VISIBLE_DEVICES`
  只在首次 CUDA 初始化时被读取，torch 的 CUDA 是懒初始化）。
  **包导入不装载**——否则会把整个 `.env`（含 agent 侧配置）灌进宿主进程，污染库使用者与测试
  （曾因此打挂 `test/test_init_env.py` 14 个用例）。
- **优先级**：脚本 `--gpu N` > shell 已 export 的同名变量 > `.env`；`.env` 不覆盖已存在的变量。
- **脚本预检会打印映射**：`CUDA_VISIBLE_DEVICES=1 → 物理卡 1`，跑之前先确认。
- **CUDA toolkit（`CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH`）不由本项目管理**（2026-09-14 决定）：
  由宿主 shell（`~/.bashrc` 等）提供；需要编译扩展时在 shell 里 `export CUDA_HOME=...` 即可，
  或临时 `CUDA_HOME=... pip install <pkg> --no-build-isolation`。
- 相关实现：`prm/env.py`（`load_project_env` / `visible_devices` / `describe`）、四个 CLI 的 `main()`、
  `test/test_prm_env.py`（离线单测）。库用法需显式调用 `prm.env.load_project_env()`。

### 14.6 吞吐实测与内核化：为什么"能跑通"≠"跑得起"（2026-09-15）

冒烟（M4.2 路径）**功能上通过**，但把时间账算出来后发现**按当前配置正式训练不可行**——
这条比任何精度指标都优先。

#### 14.6.1 实测数据（`outputs/prm/runs/smoke`，run 2026-09-15 14:49–15:34）

| 项 | 值 | 说明 |
| --- | --- | --- |
| 训练步 | 20 步 @ `max_length=8192`，bs=1，accum=16 | 首步 125 s（预热），稳态 **≈25 s/步** |
| 训练段耗时 | 9 min 58 s | loss 1.090 → 1.066；grad_norm 5.60 / 3.60 |
| dev 评估 | **4,864 条全量，`eval_runtime=2098 s`（2.32 条/s）** | 35 min —— 冒烟的 78% 时间花在这里 |
| 冒烟总耗时 | 44 min 57 s | = 训练 10 min + 评估 35 min |
| 显存峰值 | allocated 13.09 GB / reserved 14.17 GB | `run_manifest.peak_gpu_memory`（U7） |
| dev 指标 | AUC 0.6262 / Brier 0.2604 | 对比 M4.0 零训练 probe 0.5373/0.3279——**别过度解读**（20 步 + 全量 dev，与 probe 的 500 抽样口径不同） |

时间外推（`n_train=19,862`，bs=1×accum=16 → **1,242 优化步/epoch**）：

| 假设 | 单步 | 1 epoch | 2 epochs |
| --- | --- | --- | --- |
| 8K / **25 s（冒烟，卡被占）** | 25 s | 8.6 h | **17 h** |
| 8K / **2.46 s（bench 复测，卡空闲）** | 2.46 s | **13.6 h** ← 见 §14.6.2 | **27 h** |
| 16K（定稿配置；长度≈2×） | ~5 s | ~27 h | ~2.3 天 |

> **结论已被 §14.6.2 更正**：冒烟的 25 s/步是**卡被别的进程占用**所致，不是代码缺陷。
> 卡空闲时单步 2.46 s、前向+反向 = 3.47×（正常）。用训练集真实长度均值（7,426 token）
> 预计 1 epoch ≈ 10-12 h、2 epochs ≈ 1 天，**属于可接受量级，不需要"先优化再开训"**。

#### 14.6.2 归因（已实测结案）：**共享卡资源竞争**，不是代码问题

先算评估这头的账：dev 截断到 8K 后共 3.019e7 token、4,864 条、2,098 s，
按 dense 近似 `2·N·T`（N=4.227e9）折合 **≈122 TFLOPS**。这是混合架构 + bs=1 下
**正常**的吞吐（A800 bf16 峰值 ≈312 TFLOPS）。

`scripts/bench_prm_step.py` 在**同一张 A800** 上两次运行，得到决定性对照：

| 运行 | bf16 matmul 基线（4096³） | 前向 @8K | 前向+反向 | 倍数 |
| --- | --- | --- | --- | --- |
| 首次（16:0x） | 中位 **9.6 TFLOPS** | 崩（脚本 bug，见下） | — | — |
| 复测（卡空闲） | 中位 **250.5** / 最快 251.3 TFLOPS | 0.709 s（11.6k tok/s ≈ 98 TFLOPS 等效） | **2.460 s** | **3.47×** |

三点结论：

1. **卡没问题**：matmul 250 TFLOPS = 峰值的 80%，标准水平。
2. **代码没问题**：前向+反向 / 前向 = **3.47×**（理论 ≈3×，多出的来自
   gradient checkpointing 的重算）——**完全正常**，不存在"反向慢 15 倍"。
3. **那 15 倍是卡被抢占**：两次 matmul 相差 **26×**（9.6 vs 250），
   而冒烟训练的 14:49–14:59 与评估的 14:59–15:34 恰好跨越了抢占窗口。
   ⇒ `AISC-Ubuntu-Server` 是共享机，**独占性**是这类测量的前提。

**因此早期的归因假设全部作废**（当时的日志无法区分"代码慢"与"卡被占"）：
既不是 delta rule 回退（`fla 0.5.2` 确实是实现体），也不是缺 `causal_conv1d`
（其 fallback 是向量化 `F.conv1d`，代价极小，代码里仍会打印一条警告）。

**副作用提醒**：这类"看起来是代码慢"的现象，在共享卡上必须先用 matmul 基线
做体检（本脚本已内置并给出"正常/偏低"判语），否则会白查一整天。

#### 14.6.3 由此得到的真实可调项（都不改模型口径）

bench 给的实测盘子（8,211 token 单样本，bs=1，LoRA + grad ckpt）：

| 项 | 值 | 判读 |
| --- | --- | --- |
| 前向 | 0.709 s | 正常 |
| 前向+反向 | 2.460 s（3.47×） | 正常（多出的是 grad ckpt 重算）；**关掉 ckpt 试过，见下** |
| 显存峰值 | 12.88 GB / 85 GB | ⚠️ **这是"单样本单步"的峰值，不可外推到训练** |
| 外推 | 2.46 s/步 × 19,862 样本 → 1 epoch ≈13.6 h、2 epochs ≈27 h | 用训练集真实均值（7,183 token）≈ 单优化步 34 s、2 epochs ≈ 23.5 h |

> **⚠️ `gradient_checkpointing` 不能关（2026-09-15 实测，教训）**：bench 峰值只有 12.9 GB
> 极具误导性——bench 只跑 1 个样本、1 步、没有优化器状态。真实训练（16K/bs=1、32 层、
> bs 内不含 padding）在**第 1 步**就
> `torch.OutOfMemoryError: ... this process has 78.72 GiB memory in use (75.06 GiB allocated by PyTorch)`
> 直接崩（16:33 那次 run）。原因：关闭重算后每层 MLP 的 swiglu 中间激活
> （`16384 × 9728 × 2B ≈ 0.6 GB/层`）全部驻留，30+ 层就是 30-50 GB，再加权重 8.5 GB。
> ⇒ **`gradient_checkpointing: true` 是硬约束**，不是"顺手优化"；想省时间只能降
> `max_length`（8K 覆盖率仅 63.96%，见 §6）或换更大显存的卡。

⇒ 正式训练前只剩两个可选微调（各跑一条 bench 对照即可定夺）：
`--no-grad-ckpt` 与 `--use-kernels`；**都不是开训的前置条件**。

#### 14.6.4 已落地的三个开关

**① `train.use_kernels`（默认 `false`，fail-open）** —— `prm/train_prm.py::_load_scorer`

transformers 5.17 自带内核映射：`chunk_gated_delta_rule` → `kernels-community/fla`、
`causal_conv1d_fn/update` → `kernels-community/mamba-ssm`，且 flash-attn 也有 hub 兜底。
启用条件：`pip install "kernels>=0.16,<0.17"`（transformers 的
`KERNELS_MIN/MAX_VERSION`）+ `from_pretrained(..., use_kernels=True)`。

> **本机实测：该路线不可用（2026-09-15）**。`kernels 0.16.1` 装得上（PyPI 通），但
> 内核化时会调 `list_repo_refs` 去 HF Hub 解析 kernel repo 的 revision →
> `httpx.ConnectTimeout: [Errno 110] Connection timed out`（本机到 huggingface.co
> 不通；沙箱里经代理是通的，属环境差异）。故 config 默认 `false`，免得每次启动
> 白等一次连接超时；想用得先给机器配 HF 访问（`HF_ENDPOINT=https://hf-mirror.com`
> 或 `HTTPS_PROXY`，注意 kernel 仓库类型在镜像站的支持情况需自行验证）。
> **不必为它阻塞训练**——见 §14.6.3，基准吞吐本就正常。

**两条实现要点**（缺一个就会得到"开了内核却没变快"的假结论）：

1. `from_pretrained` **不会**把模型搬上 GPU，而 Hub 内核按**设备类型**匹配 →
   顺序必须是「先 `.to(cuda)`，再在 cuda 上重新内核化」，由
   `VerdictScorer.to_device_and_kernelize` 统一负责；
2. 内核化发生在搬设备那一刻，失败点也在那里 → `train_prm._load_scorer` 的
   `try` 必须把它包进去（bench 脚本最初漏了，真机上就是这样崩的；已补回归
   `test_prm_train.py::test_use_kernels_kernelize_failure_after_move_falls_back`）。

**② 评估成本封顶** —— `train.eval_max_samples: 500`（原 `null`=全量 4,864 ≈ **35 min/次**；
500 条 ≈ 3.6 min，且与 M4.0 probe 抽样口径对齐）、
`train.smoke_eval_max_samples: 64`（冒烟专用上限：**否则一次冒烟的钱几乎全花在评全量 dev 上**
——实测训练 10 min + 评估 35 min）、CLI `--eval-max-samples N` 优先级最高（`<=0`=全量）。

**③ 留档补齐**（之前 manifest 缺这些字段，出事无法归因）：`use_kernels`（实际值）、
`attn_implementation`（请求）与 `attn_implementation_effective`（`config._attn_implementation`）、
`gpu`（`torch.cuda.get_device_name(0)`）、`steps_per_epoch`；启动日志新增
"计划：N 优化步/epoch"。

#### 14.6.5 定位工具：`scripts/bench_prm_step.py`（离线自检见 `test/test_prm_bench.py`）

```bash
set -a; source .env; set +a
# 基线：默认 LoRA + grad-ckpt，8K/16K 各 3 次（约 3-5 min）
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 16384

# 对照实验（每条命令单独跑，看 step_s / bwd_ratio 变化）
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 --no-grad-ckpt
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 --no-grad-ckpt --no-lora
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 --use-kernels
```

判读（`step_s` = 前向+反向中位耗时，`bwd_ratio` = step/fwd）：

| 现象 | 结论 / 动作 |
| --- | --- |
| matmul 基线 中位/最快 都 ≫120 TFLOPS | 卡干净，下面的数字可信（**先看这一行**） |
| matmul 基线远低于该卡标称 TFLOPS | 卡被抢占/降频 → 换卡或错峰（先问用户，不自动停服务）；此时 step_s 全部不可信 |
| 倍数（前向+反向 / 前向）≈3-3.5× | **正常**（§14.6.2 实测 3.47×） |
| 倍数 ≫4 | 反向异常 → 看 `--no-grad-ckpt` / `--no-lora` 哪个把它压回 ~3 |
| `--no-grad-ckpt` 让 `step_s` 下降、峰值 ≪ 卡容量 | 关掉 `gradient_checkpointing`（改 config 一行）；实测峰值仅 12.9/85 GB，预计收益 ~12% |
| `--use-kernels` 让 `step_s` 明显下降 | 保持 `use_kernels: true`（看"内核化模块类型"是否 >0） |
| 中位 ≪ 最快（同一行内） | 时有时无的抢占 → 重跑到两者接近为止 |

**判读顺序**：先看 matmul 基线（卡干不干净）→ 再看 倍数（代码有没有问题）→ 最后才谈优化。
§14.6.2 的实测结论是「卡干净时 倍数=3.47× 属正常」，所以**不必等优化就能开训**。

#### 14.6.6 验证清单（定性 + 定量）

```bash
# 0) 装内核（宿主机；CUDA 13.0 toolkit 已在 PATH。首次调用会在 HF 缓存里下载/编译，
#    失败也不影响训练——_load_scorer 会回退并在日志里告警）
.venv-prm/bin/pip install "kernels>=0.16,<0.17"
# 1) 定量：bench 脚本（上面 §14.6.5）
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 --json outputs/prm/bench.json
# 2) 定性：重跑冒烟，看日志是否出现 "已启用 HF Hub 内核" 与稳态 s/it
set -a; source .env; set +a
./scripts/train_prm_m4.sh --force            # 或 ./scripts/run_prm_m4_gpu.sh smoke
# 3) 核对留档字段（之前缺这几个，出事无法归因）
.venv-prm/bin/python -c "import json;m=json.load(open('outputs/prm/runs/smoke/run_manifest.json'));print({k:m[k] for k in ('use_kernels','attn_implementation_effective','gpu','eval_max_samples','n_dev_eval','steps_per_epoch')})"
```

冒烟在 64 条 dev 上限下的期望耗时：训练 20 步（≈10 min，未优化前）+ 评估 ≈30 s
≈ **11 min**（原 45 min，其中 35 min 是评全量 dev）。

#### 14.6.7 `causal_conv1d`：用户决定暂不安装（2026-09-15）

- 缺失时 transformers 只打印一条 fallback 警告，走**向量化** `F.conv1d` 参考实现——
  正确性相同、代价极小（conv 的 FLOPs 占比低；主项是注意力 + delta rule）；
- 与 `kernels` 路线的关系：装 `kernels` 后 `use_kernels=True` 会尝试从
  `kernels-community/mamba-ssm` 取 `causal_conv1d_fn` 的**预编译内核**，
  从而**不需要本地编译** `causal_conv1d`；
- 若将来仍要本地装（ABI 必须是 cxx11=TRUE，与 flash-attn 轮一致）：
  源码编译 `CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 pip install
  --no-build-isolation causal-conv1d`，或用 HF 上 `torch2.9+cu130` 的
  `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl` 预编译轮。

### 14.7 运行事故与处置登记

#### 事故 1：§7.2 边界样本在 DataLoader worker 里杀掉训练（2026-09-16 02:42）

| 项 | 内容 |
| --- | --- |
| 现象 | `runs/m4-v1` 训练到 **step 1146 / 2484**（约 10 h，已过 step-1000 评估，dev AUC **0.616**）时报错退出 |
| 报错 | `RuntimeError: Caught RuntimeError in DataLoader worker process 0` → `截断后仍超限（31230 > 16384）：issue/末步过长`（`prm/data.py::truncate_messages`） |
| 根因 | 数据集里存在**被判定步本身**就超过 `max_length` 的样本：`facebookresearch__fvcore.a491d5b9.lm_rewrite__yrfwq4ch::n_dc848d679fb4f4e1::14`，`rendered_tokens=43,769`，整步删除 + tool 截断后仍 31,230 token。§7.2 规定"预算不足**报错**"，而 collator 在 DataLoader worker 内执行 → 异常直接终止训练进程 |
| 影响面 | 扫描全量：train **19,862 条里 1 条**（0.005%）、dev 378 条超长样本中 0 条失败、test 40 条中 0 条失败。即**一条样本换 10 小时** |
| 修复（D10） | ① `VerdictCollator.fits()` 预检；② `oversize_indices()` 在**开训前**按 `rendered_tokens > max_length` 只扫候选（train 880、dev 378、test 40 条），把不可用样本从数据集里摘掉（`_drop_oversize` + `Subset`）；③ 评估路径 `collate_or_skip()` / `filter_oversize_items()` 逐样本跳过，`probe` 因为要按下标对齐标签，改为**先摘后打分**；④ 扫描结果按 **parquet 指纹**（size/mtime + template_hash + max_length + tool_keep_tokens）缓存在 `outputs/prm/oversize_skip_<split>.json`，重复开训不再重扫（全扫要几分钟）；⑤ 计数进 `collator.stats["skipped_oversize"]` → `truncation_stats.json`，样本 id 进 `run_manifest.json` 的 `oversize_skipped` |
| 恢复方式 | `./scripts/train_prm_m4.sh --skip-smoke --resume --background`（从 `checkpoint-1000` 续；过滤后可用 19,861 条 > 恢复时跳过的 16,000 条，索引空间安全） |
| 回归护栏 | `test_prm_data.py::TestOversizeSkip`（fits 不污染统计 / 只扫候选 + 缓存命中 / collate_or_skip 计数 / filter 保序）；`test_prm_train.py` 的 tiny 数据集**固定含 1 条不可用样本**，断言训练照常完成且 manifest 记录 `oversize_skipped.n_train == 1` |

**教训**：凡是"训练期抛异常"的校验，都要先问一句"这个异常会不会发生在 DataLoader worker 里"。会的话就必须改成**跳过 + 审计**，否则一条脏样本 = 一次全量重跑。

#### 事故 2：关闭 gradient checkpointing 导致首步 OOM（2026-09-15 16:33）

被 bench 的 12.88 GB 峰值误导（那是"单样本单步"），关掉重算后真实训练第 1 步即
`75.06 GiB allocated by PyTorch`（80 GB 卡）→ OOM。已列入 §14.4 不可回归清单，
bench 文档也加了"峰值不可外推"警告。
