# M4 PRM 训练 Runbook（实机 A800 交接）

> 阶段 2（M4）代码已全部落地（commit `0d666dc` → `4c7f19b`，见
> `docs/prm_training_plan.md` §13.2）。离线/单测路径已在沙箱完成；以下步骤
> 需要 GPU 与 CodeAgentRL-PRM 环境，在实机执行。执行前通读 §1.2 与本文件。

## 已完成（沙箱/离线，截至本文）

| 项 | 状态 |
| --- | --- |
| M4.1 数据构建（node_mc 21,379 冲突去重后 19,862/4,864/2,317） | ✅ `outputs/prm/{train,dev,test}.parquet` + manifest |
| 真实 tokenizer length_report（approx=False） | ✅ p95=16169，**max_length=16384 已定稿**（16K 覆盖 95.2%） |
| w_neg（train 统计，cap 4.0 未触顶） | 2.8211（pos 14,664 / neg 5,198） |
| M4.0 probe 代码 / eval 代码 / 训练代码 + 单测 | ✅ PRM 单测 121 项（含 torch 门控），基线 266 保持绿 |
| 离线训练冒烟（tiny 模型过完整 Trainer 循环） | ✅ `test/test_prm_train.py` |

## 步骤 1：CodeAgentRL-PRM 环境（§1.2）

```bash
conda create -n CodeAgentRL-PRM python=3.12 -y
conda activate CodeAgentRL-PRM
pip install torch==2.7.* --index-url https://download.pytorch.org/whl/cu128   # 按驱动
pip install "transformers==5.17.0" "peft==0.20.0" pandas pyarrow pyyaml \
            safetensors accelerate pytest
pip install flash-attn --no-build-isolation    # A800（SM80）预编译轮子
```

- 不要在该环境安装 mcts/agent 侧依赖；PRM 训练不依赖 minisweagent。
- `python -m pytest test/test_prm_model.py -q` 快速自检（有 GPU 时跑真 tokenizer 版）。

## 步骤 2：M4.0 零训练侦察（GPU1，§9.1）

```bash
nvidia-smi   # 确认显存空闲；被占用先询问用户，不自动停任何服务
python -m prm.probe --config config/prm.yaml --device cuda:1
```

- 判据 `AUC > 0.55`：通过 → 继续；≈0.5 → 看报告里 `Yes/No`、`对/错` 备选对，
  依 §12 决策后修改 `config/prm.yaml` 的 `train.verdict_pair` 再复测。
- 产出 `outputs/prm/probe_report.{json,md}`。

## 步骤 3：M4.2 冒烟训练（20 steps @ 8K，§8）

```bash
python -m prm.train_prm --config config/prm.yaml --run smoke --smoke --device cuda:1
```

- 观察：loss 下降、`dev_auc` 出现、无 OOM；数据管道端到端打通即通过。
- 产物 `outputs/prm/runs/smoke/`（可删除）。

## 步骤 4：正式训练（§8）

```bash
python -m prm.train_prm --config config/prm.yaml --run m4-v1 --device cuda:1
```

- 约 19,862 样本 ÷ 16（有效 batch）≈ 1,242 steps/epoch，2 epochs；
  每 1,000 steps 评估一次，dev_auc 停涨 3 次早停。
- 产物 `outputs/prm/runs/m4-v1/`：adapter 权重 + `run_manifest.json`
  （verdict ids / template_hash / w_neg / 数据版本）+ `truncation_stats.json`。

## 步骤 5：评估（§9.2）

```bash
python -m prm.eval_prm --config config/prm.yaml --run m4-v1 --device cuda:1
```

- 产出 `outputs/prm/eval_report.{md,json}` + `predictions.parquet`。
- 主验收（§13.1）：
  1. dev/test ROC-AUC ≫ 0.5；
  2. **node_mc 与 leaf_chain 两桶 AUC 差不大**（回填推定正样本单独盯）；
  3. Brier/ECE 校准可接受；
  4. PRM 聚合分与 correct/reward 相关为正，best-of-5 中 PRM 选择器
     selected_reward ≥ random 且差值 95% CI 不含 0（弱验收）。

## 常见问题

- **`Some tensors share memory`**：已由 `VerdictTrainer._save` 覆写修复
  （tie_word_embeddings 的共享矩阵须走模型 save_pretrained 去重）。
- **transformers 5.x 参数**：`warmup_ratio` 已移除 → 代码内换算
  `warmup_steps`；`eval_strategy` 是新名。
- **template_hash 不一致**：`load_scorer_for_run` 会拒绝评估（prompt 与权重
  错配）；改过 `prm/prompts.py` 后需重建数据并重训。
- **state.db 只读**：任何脚本都以 `prm/raw.py::open_db_readonly` 打开，构建/
  评估均校验 DB 指纹不变。
