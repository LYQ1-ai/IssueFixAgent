# M4 PRM 训练 Runbook（实机 A800 交接）

> 阶段 2（M4）代码已全部落地（commit `0d666dc` → `eabf1a9`，见
> `docs/prm_training_plan.md` §13.2）。离线/单测路径已在沙箱完成；以下步骤
> 需要 GPU 与 CodeAgentRL-PRM 环境，在实机执行。执行前通读 §1.2 与本文件。

## GPU 可用性现状（2026-09-11 修订，跨会话备忘）

> ⚠️ 本节此前记录为「会话早于驱动修复、`/dev` 节点没建出来，重启会话或宿主机
> `sudo mknod` 即可修复」——**该诊断已被证伪，勿再按旧结论操作**。

- **宿主机侧完全正常**：内核模块 595.84，两块 A800 80GB 见
  `/proc/driver/nvidia/gpus/`；宿主 `/dev` 里 `nvidia0` / `nvidia1` / `nvidiactl` /
  `nvidia-uvm` 等节点齐备（创建于 8月18日，早于本次排查）。用只读容器挂载宿主
  `/dev` 可直接看到，说明节点从未缺失。
- **真正根因**：DSH 的 bash 工具把**每一条命令**都跑在 bubblewrap 里，profile 由
  `dsh-sandbox-local` 的 `bwrapProfileArgs()` 生成，其中**硬编码** `--dev /dev`
  —— bwrap 会新建一个只有 `core/fd/null/ptmx/pts/shm/tty/urandom` 等的迷你 `/dev`，
  NVIDIA 节点被结构性屏蔽。同时沙箱内 `CapEff=0` 且 `/dev` 是 `ro` 挂载，
  `mknod` 在内核能力层面就不可能成功。
- **因此**：重启会话无效、宿主机 `mknod` 无用（宿主本来就有节点）。DSH 没有可
  「重启」的沙箱对象（无持久沙箱进程），唯一开关是 file policy 模式：
  `danger-full-access` 时 `dsh-bash-sandbox` 压根不调用沙箱 provider，命令即宿主
  裸进程，`/dev/nvidia*` 随之可见。
- **解锁方式（三选一）**：
  1. 会话内执行 `/permission danger-full-access`（立即生效，但该预设同时把
     approval 设为 `never`）；
  2. `~/.dsh/settings.yaml` 增加 `permission: {defaultPreset: danger-full-access}`
     （对之后新建的会话生效）；
  3. 在宿主机 shell 直接执行本文件步骤 2–5 的命令（不依赖 DSH 沙箱）。
- **`.venv-prm` 已装 CUDA 版 torch 2.14.0+cu130**（完整 CUDA 13 运行时）；venv 的
  `pip.conf` 已固化清华源（`https://pypi.tuna.tsinghua.edu.cn/simple`）。
- **除设备访问外，其余均已就绪**（2026-09-11 实测）：基座 9.1GB 双分片可读、
  `tie_word_embeddings=True`（F1 成立）、tokenizer 248,044 词表、
  verdict ids `Correct=31995` / `Incorrect=39130` 解析正常；
  `.venv-prm` 全套 PRM 单测 **121 passed / 2 skipped**，跳过的正是下面这 2 个。
- **设备就绪后的自检顺序**：
  ```bash
  nvidia-smi                                   # 两块 A800 可见
  .venv-prm/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  .venv-prm/bin/python -m pytest test/test_prm_model.py -k TestRealModelGpu -v   # 2 个 GPU 门控测试转绿
  ```

## 已完成（沙箱/离线，截至本文）

| 项 | 状态 |
| --- | --- |
| M4.1 数据构建（node_mc 21,379 冲突去重后 19,862/4,864/2,317） | ✅ `outputs/prm/{train,dev,test}.parquet` + manifest |
| 真实 tokenizer length_report（approx=False） | ✅ p95=16169，**max_length=16384 已定稿**（16K 覆盖 95.2%） |
| w_neg（train 统计，cap 4.0 未触顶） | 2.8211（pos 14,664 / neg 5,198） |
| M4.0 probe 代码 / eval 代码 / 训练代码 + 单测 | ✅ PRM 单测 122 项（含 torch 门控），基线 266 保持绿 |
| 离线训练冒烟（tiny 模型过完整 Trainer 循环） | ✅ `test/test_prm_train.py`（含 LoRA + grad-ckpt 组合回归） |
| **GPU 门控端到端（真实 Qwen3.5-4B / A800 GPU0）** | ✅ **2026-09-11 实测两条用例 PASSED**：`pytest test/test_prm_model.py -k TestRealModelGpu`（bf16 加载 → 真 tokenizer 渲染 → 前向 z 有限 → 不同前缀 z 非退化） |
| GPU 路径已修的 bug（沙箱 skip 藏住，宿主首跑才暴露） | ① `make_tiny_messages` 未 import（NameError）；② LoRA 下 `embed.weight.register_hook` 因 `requires_grad=False` 抛 RuntimeError；③ `flash_attn` 缺失时未按 config 承诺降级 sdpa |

## 步骤 1：CodeAgentRL-PRM 环境（§1.2）

> **当前实际在用的已验证环境是仓库内 `.venv-prm`**（torch 2.14.0+cu130、transformers
> 5.17.0、peft 0.20.0、swanlab 0.10.0、flash-linear-attention 0.5.2）。下面的 conda
> 环境是备选/可复现路径；两者任选其一，但装的东西要一致。

```bash
conda create -n CodeAgentRL-PRM python=3.12 -y
conda activate CodeAgentRL-PRM
pip install torch==2.7.* --index-url https://download.pytorch.org/whl/cu128   # 按驱动
pip install "transformers==5.17.0" "peft==0.20.0" pandas pyarrow pyyaml \
            safetensors accelerate pytest
pip install swanlab flash-linear-attention     # 曲线记录 / Qwen3.5 线性注意力优化内核
# 可选（需与 torch 的 CUDA 版本匹配的完整 toolkit，见下）：
# pip install causal_conv1d flash-attn --no-build-isolation
```

- 不要在该环境安装 mcts/agent 侧依赖；PRM 训练不依赖 minisweagent。
- `python -m pytest test/test_prm_model.py -q` 快速自检（有 GPU 时跑真 tokenizer 版）。
- **Qwen3.5 是混合架构**（部分层为 gated delta rule 线性注意力），因此：
  - `flash-linear-attention`（导入名 `fla`，≥0.2.2，当前 0.5.2）→ 消除
    `chunk_gated_delta_rule is falling back to its reference PyTorch implementation`
    警告；参考实现既慢又可能materialize 大中间张量（真机首跑 OOM 的次要嫌疑）。
    它走 triton JIT，**首次前向会多花几十秒编译**，属正常。
  - `causal_conv1d` / `flash-attn` 需要**与 torch 同版本的完整 CUDA toolkit** 现场编译
    （PyPI/清华源均无预编译轮子）。本机 `/usr/local/cuda-12.1` 与 torch cu130 **不匹配**，
    故未安装：`causal_conv1d_fn` 继续走参考实现（仅慢），注意力已由
    `attn_implementation` 自动降级 `sdpa`（PyTorch 内置 memory-efficient 注意力），
    不影响正确性。
- 自检是否生效（**需在能看到 GPU 的宿主环境执行**）：
  ```bash
  python -c "import torch; from transformers.utils import is_flash_linear_attention_available as f; \
             print(torch.cuda.is_available(), f())"   # 期望 True True
  ```

## 环境变量：GPU 选择（`.env`）

**用哪张卡由仓库根 `.env` 决定**，不写死在代码/脚本/配置里（§14.5）：

```bash
# .env
CUDA_VISIBLE_DEVICES=0                 # 用哪张卡（进程内 cuda:0 = 这张）
```

- 各 CLI 入口（`python -m prm.*` 的 `main()`）开头 `load_dotenv`（早于任何 torch/CUDA 初始化；
  **包导入不装载**，避免污染其他进程/测试），所以 CLI 一律用默认 `cuda`，
  **不要再写 `--device cuda:1`**（那是物理卡号，会和 `.env` 冲突）；
- 优先级：`--gpu N`（脚本）> shell 已 export 的同名变量 > `.env`；`.env` 不覆盖已存在的变量；
- 脚本预检会打印 `CUDA_VISIBLE_DEVICES → 物理卡` 供确认；
- **CUDA toolkit 不归本项目管**：`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` 由宿主 shell（`~/.bashrc`）提供，
  编译扩展时确保 `nvcc` 在 PATH 里即可。

## 一键脚本（推荐）

**正式训练用 `scripts/train_prm_m4.sh`**（在**宿主机 shell** 执行；**用哪张卡由仓库根 `.env`
的 `CUDA_VISIBLE_DEVICES` 决定**，见 §14.5，脚本预检会打印映射，
用户 2026-09-11 指定）。它把 步骤 3–5 串成一次调用，并带预检与断点续训：

```bash
cd /home/lyq/PycharmProjects/CodeAgentRL
./scripts/train_prm_m4.sh                  # 预检 → 冒烟 → 正式训练 m4-v1 → 评估
./scripts/train_prm_m4.sh --resume         # 中断后从最新 checkpoint 续训
./scripts/train_prm_m4.sh --background     # nohup 后台跑（打印日志路径）
./scripts/train_prm_m4.sh --require-probe  # probe 门禁未过则拒绝开训
./scripts/train_prm_m4.sh --dry-run        # 只打印命令
```

预检项：GPU 存在且空闲显存 ≥ `--min-free-gb`（默认 30GB，不足即退出且**不自动停服务**）、
数据产物齐全、**build manifest 的 `template_hash` 与当前 `prm/prompts.py` 一致**、
run 目录未被占用（已训过需 `--force` 或 `--resume`）、磁盘 ≥10GB。

**分步则用 `scripts/run_prm_m4_gpu.sh`**：

```bash
./scripts/run_prm_m4_gpu.sh check    # 自检 + 2 个 GPU 门控单测
./scripts/run_prm_m4_gpu.sh probe    # = 步骤 2
./scripts/run_prm_m4_gpu.sh smoke    # = 步骤 3
./scripts/run_prm_m4_gpu.sh train    # = 步骤 4
./scripts/run_prm_m4_gpu.sh eval     # = 步骤 5
./scripts/run_prm_m4_gpu.sh all      # 一条龙，中途失败即停
```

环境变量：`PRM_GPU=<物理卡号>`（临时覆盖 .env）、`PRM_RUN=m4-v1`、`PRM_PY=.venv-prm/bin/python`、
`PRM_MIN_FREE_GB=20`。脚本会在训练前检查目标卡空闲显存，不足直接退出（不自动停服务）。

## 步骤 2：M4.0 零训练侦察（.env 选中卡，§9.1）

```bash
nvidia-smi   # 确认显存空闲；被占用先询问用户，不自动停任何服务
# 不传 --device：默认 cuda = .env 选中的那张卡（CUDA_VISIBLE_DEVICES）
python -m prm.probe --config config/prm.yaml
```

- 判据 `AUC > 0.55`：通过 → 继续；≈0.5 → 看报告里 `Yes/No`、`对/错` 备选对，
  依 §12 决策后修改 `config/prm.yaml` 的 `train.verdict_pair` 再复测。
- 产出 `outputs/prm/probe_report.{json,md}`。

## 步骤 3：M4.2 冒烟训练（20 steps @ 8K，§8）

```bash
# 注意：train_prm 没有 --device 参数（旧版本文档此处有误），选卡用 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python -m prm.train_prm --config config/prm.yaml --run smoke --smoke
```

- 观察：loss 下降、`dev_auc` 出现、无 OOM；数据管道端到端打通即通过。
- 产物 `outputs/prm/runs/smoke/`（可删除）。

## 步骤 4：正式训练（§8）

```bash
CUDA_VISIBLE_DEVICES=0 python -m prm.train_prm --config config/prm.yaml --run m4-v1
```

- 约 19,862 样本 ÷ 16（有效 batch）≈ 1,242 steps/epoch，2 epochs；
  每 1,000 steps 评估一次，dev_auc 停涨 3 次早停。
- 产物 `outputs/prm/runs/m4-v1/`：adapter 权重 + `run_manifest.json`
  （verdict ids / template_hash / w_neg / 数据版本）+ `truncation_stats.json`。
- `config/prm.yaml` 的 `attn_implementation` 若为 `flash_attention_2` 而未安装
  flash-attn，`train_prm` 会自动降级 `sdpa`（已实现，不必手工改配置）。

## 步骤 5：评估（§9.2）

```bash
python -m prm.eval_prm --config config/prm.yaml --run m4-v1
```

- 产出 `outputs/prm/eval_report.{md,json}` + `predictions.parquet`（新增 `truncated` 列）。
- 主验收（§13.1）：
  1. dev/test ROC-AUC ≫ 0.5；
  2. **node_mc 与 leaf_chain 两桶 AUC 差不大**（回填推定正样本单独盯）；
  3. Brier/ECE 校准可接受；**新增**：`truncated` vs `not_truncated` 两桶 AUC 差（判断截断是否伤指标）；
  4. PRM 聚合分与 correct/reward 相关为正，best-of-5（候选数固定 k=5，报告含 `n_candidates` 分布）中
     PRM 选择器 selected_reward ≥ random 且差值 95% CI 不含 0（弱验收）。

## 训练曲线记录（SwanLab，已接入）

官方三步：`pip install swanlab` → `swanlab login` → 训练自动上报。本项目已在
`config/prm.yaml` 打开：

```yaml
train:
  report_to: [swanlab]
  swanlab:
    project: CodeAgentRL
    workspace: "2410104018"   # 用户名/团队空间
    mode: cloud               # cloud | local | disabled
    log_dir: outputs/prm/swanlog   # 仅 local 模式用
```

- **实验名 = `--run` 名**（`m4-v1`）；超参由代码 `swanlab.init(config=...)` 上报
  （另含 lr/batch/LoRA/seed 等），HF 的 `SwanLabCallback` 负责 train_loss /
  eval 指标逐 step 上报。曲线地址与 run_id 会写进 `run_manifest.json` 的
  `swanlab` 字段，便于回溯。
- **凭据落点**（swanlab 0.10 实测）：`swanlab login`（`save="root"`）→
  `~/.swanlab/.netrc`；`swanlab login --local` → `./.swanlab/.netrc`；
  也兼容 `SWANLAB_API_KEY` 环境变量与标准 `~/.netrc`。
- **两个上游坑（已规避，勿回退）**：
  1. transformers 5.17 的 `SwanLabCallback` 以 `swanlab.get_run() is None` 判断是否
     已初始化，而 swanlab 0.10 的 `get_run()` 在无 run 时**抛 RuntimeError** →
     `on_train_begin` 直接崩。故代码**先由自己 `swanlab.init`**，让回调走「已初始化」
     分支。
  2. `SWANLAB_PROJECT` 环境变量的字段类型是嵌套模型，普通字符串会让 `swanlab.init`
     抛 `SettingsError`（只认 `'{"name": "x"}'`）→ 代码不写该环境变量，project 只经
     `init(project=...)` 传入，并清理环境里已有的非 JSON 写法。
- **安全网**：`mode=cloud` 但既无 API key 也无上述任何凭据文件时，代码会
  **禁用上报并打 ERROR 日志**（避免交互式登录卡住 nohup 训练），训练照常进行；
  初始化异常同样只禁用记录器、不中断训练。
- 想完全离线：把 `mode` 改 `local`（另需 `pip install 'swanlab[dashboard]'` 才能看
  `swanlab watch outputs/prm/swanlog`）。
- 若改用 runbook 步骤 1 的 `CodeAgentRL-PRM` conda 环境，记得该环境里也要
  `pip install swanlab`（当前 `.venv-prm` 已装 0.10.0）。

## 常见问题

- **CUDA OOM（真机首次 probe 即遇，已修）**：现象是
  `memory allocation failed ... trying to allocate 64353206272 bytes`（≈65GB）。
  根因：`VerdictScorer.forward` 原来调 `backbone(...)` 拿**整段** `(B, L, V)` logits，
  V=248,044、B=8、L=16384 → `8×16384×248044×2B ≈ 65GB`。
  修复：`forward` 用 Qwen3.5 支持的 **`logits_to_keep=1`**（transformers 5.x 名称；
  4.x 为 `num_logits_to_keep`，代码自动探测，PeftModel 先解包）——切片发生在 lm_head
  **之前**，logits 从 65GB 降到 ≈4MB。
- **⚠️ padding 事故（2026-09-11→14 定位并修正，勿回退）**：曾把 collator 改成**左**
  padding + 位置切片。真机实测左 padding 让 σ(z) 偏移最多 **+0.70** 且随 pad 数剧烈
  跳变；**右** padding + 逐行取位则 Δz=0.0000（精确）。原因是 Qwen3.5 混合架构
  （`causal_conv1d` + gated delta rule）的顺序递推对 pad 前缀敏感。
  ⇒ **现行方案：全链路强制 `batch_size=1`（不产生任何 padding）+ 右 padding 保留兼容**。
  代码里已写死拒绝：`VerdictCollator`（多样本批）、`train_prm`（`per_device_train_batch_size`）、
  `probe`/`eval_prm`（`--batch-size`）；`per_device_eval_batch_size` 也必须显式设为 1
  （TrainingArguments 默认 8，会在 dev 评估时引入 padding）。**不要再加 `--batch-size 2`。**
  完整实验数据与影响面见 `docs/prm_training_plan.md` §14.2。
- **Qwen3.5 混合架构的慢速回退（不影响正确性，只影响速度）**：
  - `chunk_gated_delta_rule is falling back ...` → **本机不会出现**：`flash-linear-attention 0.5.2`
    已装，transformers 的优先级是「Hub 内核 > 原始包(fla) > torch 参考实现」，实测
    `resolve_internal_import(fla, "ops.gated_delta_rule.chunk_gated_delta_rule")` 解析到真内核
    （`fla/ops/gated_delta_rule/chunk.py`）。**注意**：这条与 `fla` 是否 `import` 成功无关地
    决定了训练快慢，排查时先确认它没退化成参考实现。
  - `causal_conv1d_fn is falling back ...` → 用户决定**暂不装** `causal_conv1d`：其 fallback 是
    向量化 `F.conv1d`（`modeling_qwen3_5.py:280`），正确性相同、代价极小。若哪天想省这点时间，
    优先走 `kernels` 的 Hub 预编译内核（见下节），而不是本地编译。
  - 注意力侧 `attn_implementation=flash_attention_2` 不可用时会自动降级 `sdpa`（不报错）；
    manifest 里 `attn_implementation_effective` 记录**实际**用了哪个。
- **⚠️ 训练吞吐（2026-09-15 实测，正式训练前必须先解决）**：见下节「吞吐：先量后跑」。
- **`Some tensors share memory`**：已由 `VerdictTrainer._save` 覆写修复
  （tie_word_embeddings 的共享矩阵须走模型 save_pretrained 去重）。
- **transformers 5.x 参数**：`warmup_ratio` 已移除 → 代码内换算
  `warmup_steps`；`eval_strategy` 是新名。
- **template_hash 不一致**：`load_scorer_for_run` 会拒绝评估（prompt 与权重
  错配）；改过 `prm/prompts.py` 后需重建数据并重训。
- **state.db 只读**：任何脚本都以 `prm/raw.py::open_db_readonly` 打开，构建/
  评估均校验 DB 指纹不变。

## 吞吐：已结案 = 共享卡被抢占（2026-09-15）

冒烟那 25 s/步**不是代码问题**，是卡被别人占了。同一张 A800 上两次 bench 的对照：

| 运行 | bf16 matmul 基线（4096³） | 前向 @8K | 前向+反向 | 倍数 |
| --- | --- | --- | --- | --- |
| 卡被抢占时 | 中位 **9.6 TFLOPS** | —（脚本当时有 bug，已修） | — | — |
| 卡空闲时 | 中位 **250.5** / 最快 251.3 TFLOPS | 0.709 s（≈98 TFLOPS 等效） | **2.460 s** | **3.47×** |

- 250 TFLOPS = A800 峰值的 80%，**正常**；倍数 3.47× ≈ 理论值（多出的是 grad ckpt 重算），**正常**；
- 两次 matmul 相差 **26×** → 那 15 倍"变慢"来自资源竞争。`AISC-Ubuntu-Server` 是共享机，
  **任何吞吐测量/正式训练前先看 `nvidia-smi`**；
- ⚠️ **`gradient_checkpointing` 必须保持 `true`**：bench 的 12.88/85 GB 是「单样本单步」的峰值，
  真实训练关掉后**第 1 步就 OOM**（实测 `75.06 GiB allocated by PyTorch`、进程 78.72 GiB ——
  16:33 那次 run 就崩在这里）。16K × bs=1 的激活（MLP swiglu ≈0.6 GB/层 × 32 层）就是 30-50 GB；
- 外推：**1 epoch ≈ 12 h，2 epochs ≈ 1 天**（单优化步 ≈34 s × 2,484 步，见下）。

```bash
# 体检这张卡 + 单步计时（脚本自带"正常/偏低"判语；先看 matmul 那一行）
set -a; source .env; set +a
.venv-prm/bin/python scripts/bench_prm_step.py --length 8192 --json outputs/prm/bench.json
```

> **`--use-kernels` 在本机走不通（2026-09-15 实测）**：`kernels` 包能装，但解析 kernel
> repo revision 要连 HF Hub → `httpx.ConnectTimeout [Errno 110]`（本机不通 HF）。故 config
> 里 `use_kernels: false`；它对吞吐也不是必需项。要启用请先配 `HF_ENDPOINT` 镜像或
> `HTTPS_PROXY`。另外注意顺序：`from_pretrained` 不搬 GPU、Hub 内核按设备类型匹配，
> 必须先 `.to(cuda)` 再内核化（由 `VerdictScorer.to_device_and_kernelize` 统一处理），
> 否则 `use_kernels=True` 只是个空开关。

完整机理与判读表见 `docs/prm_training_plan.md` §14.6。

**评估成本已封顶**（原 `eval_max_samples: null` = 全量 4,864 ≈ 35 min/次）：

| 配置 | 效果 |
| --- | --- |
| `train.eval_max_samples: 500` | 每次 dev 评估 ≈3.6 min，且与 M4.0 probe 抽样口径对齐 |
| `train.smoke_eval_max_samples: 64` | 冒烟评估 ≈30 s（否则冒烟 45 min 里 35 min 都在评全量 dev） |
| `--eval-max-samples N`（CLI） | 临时覆盖两者；`<=0` = 全量 |

## 默认超参（config/prm.yaml，正式训练用）

| 组 | 值 |
| --- | --- |
| 基座 / 精度 | `Qwen3.5-4B`；bf16；`attn_implementation=flash_attention_2`（缺 flash-attn 自动降级 sdpa） |
| **LoRA** | **开**：r=16、alpha=32、dropout=0.05、bias=none、target=[q,k,v,o,gate,up,down]_proj |
| 有效 batch | per_device **1**（强制，无 padding）× grad_accum **16** = 16；dev 评估 `per_device_eval_batch_size=1` |
| 学习率 | 1e-4，cosine，warmup_ratio 0.05，max_grad_norm 1.0，weight_decay 0.0 |
| epochs / 步数 | 2 epochs ≈ 1,242 steps/epoch（19,862 样本） |
| 序列长度 | `max_length=16384`（§6 定稿，真实 tokenizer p95=16169） |
| 评估/保存 | 每 1,000 steps，`save_total_limit=2`，`metric_for_best_model=dev_auc`，早停 patience 3；`eval_max_samples: 500`（dev 只评前 500 条 ≈3.6 min；`null`=全量 4,864 ≈35 min），冒烟另有 `smoke_eval_max_samples: 64` |
| 内核 / attn | `use_kernels: false`（本机 HF Hub 不可达，见上节；开了也是 fail-open 回退）；`attn_implementation=flash_attention_2`（缺 flash-attn 自动降级 sdpa）；实际值记在 manifest 的 `*_effective` 字段 |
| 显存优化 | **`gradient_checkpointing: true`（硬约束）**：实测关掉后 16K/bs=1 第 1 步即 OOM（78.72 GiB / 80 GB）；bench 的 12.88 GB 是单样本单步、不可外推。`use_cache=false` |
| verdict | `["Correct", "Incorrect"]` → ids 39,130 / 31,995（F2 单 token 差） |
| 损失 | 加权 soft-BCE（w_pos=1.0、w_neg=2.8211，来自 build manifest） |
| 其它 | seed 42；probe 500 样本；best-of-**k=5**（`eval.best_of.k` 已接线，按 `rollout_idx` 取前 k）；bootstrap 1,000；γ=0.95 |
| run 产物 | `config.yaml`（有效配置快照）、`run_manifest.json`（含 `peak_gpu_memory`）、`truncation_stats.json`、adapter/tokenizer/checkpoints |

> probe（M4.0）是**零训练侦察**：它**不包 LoRA**，直接用未训练基座测三个 token 对
> 的方向性，因此它的显存特征与训练不同（训练有 LoRA 但 logits 同样只算末位）。

