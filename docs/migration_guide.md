# 迁移指南：把 CodeAgentRL（含阶段二 PRM）搬到新机器

> 适用版本：`main` = `ff1c6f8`（2026-09-24）。
> 配套自检脚本：`bash scripts/check_migration.sh`（迁移前当清单、迁移后当验收，退出码 = 必选项失败数）。

## 0. 一句话结论

| 层次 | 内容 | 怎么拿到 |
| --- | --- | --- |
| **A. 已同步 GitHub** | 全部代码 / 配置 / 测试 / 脚本 / 文档（107 个文件） | `git clone` 即可 |
| **B. 必须手工迁移** | 数据与产物（`outputs/` 下，gitignore）、基座模型、`.env`、flash-attn 轮、Docker 镜像 | `rsync` / `tar`（见 §4） |
| **C. 新机重建** | Python venv、CUDA toolkit、SSH 凭据、SwanLab 凭据 | 按 §3 重建（**不要直接拷 venv**） |

关键前提：`.gitignore` 忽略了 `outputs/`、`*.parquet`、`*.db`、`*.whl`、`.env`、`.venv*`——
所以**数据、模型产物、环境都不在 GitHub 上**，必须手工搬。

## 1. A 类：GitHub 上已同步的内容

```bash
git clone git@lyq1-ai:LYQ1-ai/IssueFixAgent.git CodeAgentRL   # 需 SSH 别名+私钥（见 §3.4）
# 或 HTTPS：git clone https://github.com/LYQ1-ai/IssueFixAgent.git CodeAgentRL
cd CodeAgentRL && git log --oneline -1        # 期望 ff1c6f8
```

含：

- `prm/`（阶段二全部实现：`build_dataset` / `data` / `modeling` / `train_prm` / `eval_prm` / `probe` / `metrics` / `env`）、`mcts/`、`agent/`
- `config/prm.yaml`（含 16K、bs=1、eval 上限 500、`use_kernels=false`、`gradient_checkpointing=true`）
- `scripts/`（`train_prm_m4.sh`、`run_prm_m4_gpu.sh`、`bench_prm_step.py`、`check_migration.sh`、`Dockerfile` 等）
- `test/`（PRM 离线套件 161 项；conda 基线 362 项）
- `docs/`（`prm_training_plan.md` 含 §14 实测结论文档、`prm_m4_runbook.md`、本文件）
- `.env_template`（照着填 `.env`）

新机 clone 后即可 `pytest test/test_prm_*.py`（需 torch 环境）与阅读全部设计/实测文档。

## 2. B 类：必须手工迁移（gitignored）

### 2.1 最小可跑（outputs 部分 ≈0.86 GB；另需 flash-attn 轮 233 MB + 基座模型 8.8 GB）——只做评估 / 续训 / 下游筛选

| 路径 | 体积 | 用途 | 不迁的后果 |
| --- | --- | --- | --- |
| `outputs/prm/{train,dev,test}.parquet` | 22 MB | 训练 / dev 评估 / test 评估数据 | 无法训练与评估 |
| `outputs/prm/manifest.json` | 2.8 KB | 类别权重 `w_pos/w_neg`、`template_hash`、计数 | w_neg 退化为即时统计（口径漂移） |
| `outputs/prm/runs/m4-v1/` | 830 MB | 已训练打分器（adapter 85 MB + tokenizer + `checkpoints/`） | 无法评估；`checkpoints/` 缺了就不能续训 |
| `outputs/prm/{build_report,length_report}.md`、`spot_check.md`、`probe_report.{md,json}` | 0.4 MB | 数据/长度/抽检/侦察留档 | 只是丢证据，不影响运行 |
| `outputs/prm/oversize_skip_*.json` | 1 KB | §7.2 边界样本跳过缓存 | 首次训练自动重扫（约 4 min） |
| `.env` | 1 KB | 运行时配置（Docker 镜像、超时、`CUDA_VISIBLE_DEVICES`） | 脚本无法定位卡/镜像 |
| `flash_attn-*.whl` | 233 MB | 与 torch 2.9.1+cu130、cxx11 ABI **精确匹配**的预编译轮 | 重建 venv 时 flash-attn 装不上（会降级 sdpa，更慢） |
| 基座模型 `/media/shared_e/models/Qwen3.5-4B`（仓库外！） | 8.8 GB | 训练/评估的基座与 tokenizer | 训练与评估都跑不了 |

> `runs/m4-v1/` 若只想搬"能评估"的最小集：`model.safetensors` + `adapter_config.json` +
> `tokenizer*.json` + `chat_template.jinja`（约 110 MB，`checkpoints/` 可省）。

### 2.2 完整可重建（再 +5.8 GB）——还需要重跑 `build_dataset`

| 路径 | 体积 | 用途 |
| --- | --- | --- |
| `outputs/batch500/state.db`（`+ state.db-wal`、`state.db-shm`） | 5.8 GB | MCTS 树库，**数据重建的唯一来源**（只读使用，迁移时三个文件一起拷） |
| `outputs/mcts/splits.parquet` | 840 KB | instance 级三层划分（防泄漏），`config.splits.path` 指向它 |

### 2.3 可选（按需）

| 路径 | 体积 | 说明 |
| --- | --- | --- |
| `outputs/prm/logs/` | 1.5 MB | 训练/评估日志（复盘与答辩用，建议留） |
| `outputs/prm/swanlog/` | 48 MB | SwanLab 本地副本（云端另有） |
| `outputs/prm/runs/smoke/` | 506 MB | 冒烟产物，可重跑 |
| `outputs/mcts_batch/`、`outputs/mcts_qwen*/`、`outputs/mcts_gemma/`、`outputs/traces/` | ≈4 GB | 历史实验产物，与阶段二无关 |
| `/media/shared_e/lyq/repos` | 821 MB | agent 侧仓库缓存（只跑 rollout 才需要） |
| Docker 镜像 `codeagentrl-agent:ubuntu24` | 506 MB | agent/rollout 侧沙盒镜像，可按 `scripts/Dockerfile` 重建 |
| Docker 镜像 `codeagentrl-agent:lean4.33.1{,-mathlib4.33.1}` | 4.1 GB / 12.2 GB | 数学类任务的沙盒镜像，按需 |

### 2.4 不要迁移的

- `.venv-prm/`（6.2 GB）：`pyvenv.cfg` 记录绝对路径（`home = /home/lyq/anaconda3/envs/CodeAgentRL/bin`）+ 绝对路径 shebang，跨机拷贝易坏，**在新机重建**（§3.1）
- `.git/`、`__pycache__/`、`.pytest_cache/`：clone 即得 / 自动生成

## 3. C 类：新机环境重建

### 3.1 Python 环境（务必重建，不要拷 venv）

```bash
cd <repo>
python3.12 -m venv .venv-prm                  # 任意 3.12 即可（原环境用 conda CodeAgentRL 的 3.12）
cat > .venv-prm/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
# ① 先装 torch：清华镜像**没有** +cu130 构建，必须走 PyTorch 官方源
.venv-prm/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1
# ② flash-attn：用随仓库搬过来的本地轮（ABI 必须 cxx11=TRUE）
.venv-prm/bin/pip install ./flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
# ③ 其余依赖按留档复原，并核对
.venv-prm/bin/pip install -r outputs/prm/requirements-freeze.txt    # 注意：该文件里的 flash_attn 是绝对路径
.venv-prm/bin/pip check
```

注意事项：

- `outputs/prm/requirements-freeze.txt` 里 `flash_attn @ file:///home/lyq/PycharmProjects/CodeAgentRL/...whl`
  是**绝对路径**：新机路径不同时要先 `sed -i` 改成新路径，或跳过该行（已由步骤 ② 装好）。
- 装完自检：`.venv-prm/bin/python -c "import torch,flash_attn,transformers,peft,fla;print(torch.__version__, torch.cuda.is_available())"`
- 若脚本用的解释器不叫 `.venv-prm`，在 `.env` 里加 `PRM_PY=/abs/path/to/python`（脚本会优先用它）。

### 3.2 CUDA toolkit（可选）

跑通 PRM 训练**不需要** nvcc（torch cu130 轮自带运行时）。只有以下情况需要装 CUDA 13.0 toolkit 并 `export CUDA_HOME`：
编译 `causal_conv1d`、或启用 HF Hub 内核（`use_kernels`，见 §6）。

### 3.3 其它外部依赖

- **基座模型**：把 8.8 GB 的 `Qwen3.5-4B` 放到**相同绝对路径** `/media/shared_e/models/Qwen3.5-4B`（最省事）；
  若路径不同，至少改 `config/prm.yaml` 两处：`tokenizer:` 与 `train.base_model:`。
  ⚠️ 已训练 run 的 `run_manifest.json` 里也记着训练时的 `base_model` 绝对路径，`eval_prm` 会按它加载——
  路径变了要么保持同名软链接，要么重新生成该字段。
- **Docker**：agent/rollout 侧需要 `codeagentrl-agent:ubuntu24`；`docker load` 迁移或按 `scripts/Dockerfile` 重建。
- **repo 缓存**：`CODEAGENTRL_REPO_CACHE_DIR`（本机 `/media/shared_e/lyq/repos`），首次 rollout 会自己拉取。

### 3.4 凭据

| 凭据 | 用途 | 迁移方式 |
| --- | --- | --- |
| `~/.ssh/config` 的 `Host lyq1-ai` + 私钥 | git 远端用的是 SSH 别名 | 复制配置与私钥（或把 remote 改成 HTTPS + token） |
| `~/.netrc`（swanlab）或 `SWANLAB_API_KEY` | 训练曲线云上报 | 复制；没有则自动降级为不上报（不影响训练） |
| HuggingFace 访问 | Hub 内核（可选） | 本机实测**不可达**，故 `use_kernels: false`；新机若能连 HF 可改为 `true` |

## 4. 迁移操作步骤

### 4.1 源机打包

```bash
cd /home/lyq/PycharmProjects/CodeAgentRL

# 最小集（≈0.9G，含环境/配置/产物；不含 state.db）
tar czf /tmp/prm-min.tgz .env flash_attn-*.whl \
  outputs/prm/train.parquet outputs/prm/dev.parquet outputs/prm/test.parquet \
  outputs/prm/manifest.json outputs/prm/build_report.md outputs/prm/length_report.md \
  outputs/prm/spot_check.md outputs/prm/probe_report.md outputs/prm/probe_report.json \
  outputs/prm/oversize_skip_*.json outputs/prm/runs/m4-v1

# 完整集（再 +state.db / splits，用于重跑数据构建）
tar czf /tmp/prm-full.tgz outputs/batch500/state.db outputs/batch500/state.db-wal \
  outputs/batch500/state.db-shm outputs/mcts/splits.parquet

# 基座模型（8.8G，走网络建议 rsync 断点续传）
rsync -avhP /media/shared_e/models/Qwen3.5-4B/ NEWHOST:/media/shared_e/models/Qwen3.5-4B/
```

> 大数据优先用 `rsync -avhP --partial`（可断点续传），例如：
> ```bash
> rsync -avhP outputs/batch500/ NEWHOST:<repo>/outputs/batch500/
> rsync -avhP outputs/mcts/splits.parquet NEWHOST:<repo>/outputs/mcts/
> rsync -avhP outputs/prm/ NEWHOST:<repo>/outputs/prm/
> ```
> `state.db-wal` 非空时**必须**与 `state.db` 一起搬（SQLite WAL 语义）；本机 wal 为 0 字节，属干净状态。

### 4.2 新机还原

```bash
git clone git@lyq1-ai:LYQ1-ai/IssueFixAgent.git CodeAgentRL && cd CodeAgentRL
tar xzf /tmp/prm-min.tgz                 # 或 prm-full.tgz
python3.12 -m venv .venv-prm             # 见 §3.1 完整命令
```

### 4.3 验收（三步，必做）

```bash
bash scripts/check_migration.sh          # ① 逐项自检：❌ 必选项、⚠️ 可选项
set -a; source .env; set +a
./scripts/train_prm_m4.sh --dry-run      # ② 只打印将要执行的命令（不占卡）
./scripts/run_prm_m4_gpu.sh check        # ③ 3 个 GPU 门控单测
./scripts/run_prm_m4_gpu.sh smoke        # ④ 冒烟 20 steps @8K（约 11 min，验通链路）
```

冒烟通过即代表：数据、模型、环境、显存、SwanLab（可选）整条链路在新机可用。

## 5. 迁移核对清单

- [ ] `git clone` 后 `git log -1` = `ff1c6f8`（或更新）
- [ ] `outputs/prm/{train,dev,test}.parquet` + `manifest.json` 到位
- [ ] `outputs/prm/runs/m4-v1/model.safetensors` 到位（+ `checkpoints/` 若要续训）
- [ ] 基座模型 `/media/shared_e/models/Qwen3.5-4B` 存在（或有软链接/改过 config）
- [ ] `.env` 已就位，`CUDA_VISIBLE_DEVICES` 指向新机空闲卡
- [ ] `.venv-prm` 重建完成，`pip check` 无冲突，`torch.cuda.is_available()=True`
- [ ] `flash_attn` 可导入（`attn_implementation_effective=flash_attention_2`）
- [ ]（要重跑数据构建）`outputs/batch500/state.db` + `outputs/mcts/splits.parquet` 到位
- [ ]（要 rollout）Docker 镜像 + repo 缓存
- [ ] `bash scripts/check_migration.sh` 退出码 0
- [ ] `--dry-run` → `check` → `smoke` 全部通过

## 6. 已知坑（都踩过，别重复）

1. **不要直接拷 `.venv-prm`**：`pyvenv.cfg` 与脚本 shebang 内是绝对路径；换机换路径必坏，重建只要十几分钟。
2. **torch 必须走 PyTorch 官方 cu130 源**：清华镜像只有 CPU/旧 CUDA 构建，`pip install torch==2.9.1+cu130` 会找不到。
3. **flash-attn 轮的 ABI**：必须是 `cxx11abiTRUE` 且 cp312/cu13/torch2.9 四要素匹配，否则 import 报 `undefined symbol`。
4. **基座模型路径**：`config/prm.yaml`（2 处）+ 已训练 run 的 `run_manifest.json` 都写死了绝对路径。
5. **`state.db` 三件套**：`state.db` / `-wal` / `-shm` 一起搬；本库严格只读，迁移时**不要**对源库执行 checkpoint 写操作。
6. **`gradient_checkpointing` 必须 `true`**：关掉后 16K × bs=1 第 1 步就 OOM（实测 78.7 GiB / 80 GB）。
7. **batch size 必须 1**：Qwen3.5 混合架构对 pad 前缀敏感（左 padding 打分偏移最多 0.70），collator/CLI 会直接拒绝 >1。
8. **HF Hub 不通时 `use_kernels` 保持 false**：否则每次启动白等一次连接超时（代码会 fail-open 回退，不影响训练）。
