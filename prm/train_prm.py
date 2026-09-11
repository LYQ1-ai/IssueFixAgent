# SPDX-License-Identifier: BSD-3-Clause

"""PRM 训练 CLI（docs/prm_training_plan.md §8，M4.2）。

- 基座 Qwen3.5-4B + PEFT LoRA（r=16/α=32/dropout=0.05，q/k/v/o/gate/up/down）；
- 损失 = 加权 soft-BCE（w_pos/w_neg 来自 build manifest 的 train 统计）；
- 评估 = dev AUC（``metric_for_best_model=dev_auc``，选优 + early stopping）；
- 产物 ``outputs/prm/runs/<name>/``：``model.safetensors``（= LoRA adapter 权重，
  另存 adapter_model.safetensors/adapter_config.json）+ ``run_manifest.json``
  （verdict ids / max_length / template_hash / w_neg / 数据版本）+ tokenizer
  + trainer_state.json + truncation_stats.json。

GPU 纪律（§1.2）：训练跑 A800 GPU1；启动前 ``nvidia-smi`` 确认显存空闲，被占用
先询问用户，不自动停任何服务。冒烟路径：``--smoke``（20 steps @ 8K）。
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from prm.data import PrmParquetDataset, VerdictCollator
from prm.metrics import brier_score, roc_auc
from prm.modeling import VerdictScorer, resolve_verdict_ids, verdict_loss
from prm.prompts import TEMPLATE_VERSION, template_hash

logger = logging.getLogger("prm.train")


# ---------------------------------------------------------------------------
# Trainer 子类：加权 soft-BCE + dev AUC 评估
# ---------------------------------------------------------------------------

def make_trainer_class():
    """惰性定义 Trainer 子类（torch/transformers 导入门控）。"""
    import torch
    from transformers import Trainer

    class VerdictTrainer(Trainer):
        """compute_loss = 加权 soft-BCE；prediction_step 输出 z 供 AUC 计算。"""

        def __init__(self, *args, w_pos: float = 1.0, w_neg: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.w_pos = float(w_pos)
            self.w_neg = float(w_neg)

        def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
            z = model(input_ids=inputs["input_ids"],
                      attention_mask=inputs["attention_mask"])
            loss = verdict_loss(z, inputs["labels"], self.w_pos, self.w_neg)
            return (loss, {"z": z}) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only: bool,
                            ignore_keys=None):
            with torch.no_grad():
                z = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"])
                loss = verdict_loss(z, inputs["labels"], self.w_pos, self.w_neg)
            if prediction_loss_only:
                return (loss.detach(), None, None)
            return (loss.detach(), z, inputs["labels"])

    return VerdictTrainer


def make_compute_metrics():
    """eval_pred → dev 指标（键名与 config.train.metric_for_best_model 对齐）。"""
    def compute_metrics(eval_pred) -> dict:
        preds = np.asarray(eval_pred.predictions, dtype=np.float64).reshape(-1)
        labels = np.asarray(eval_pred.label_ids, dtype=np.int64).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-preds))  # σ(z)
        return {
            "dev_auc": roc_auc(labels, probs),
            "dev_brier": brier_score(labels, probs),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------

def train(cfg: dict, run_name: str, *, smoke: bool = False,
          max_length: Optional[int] = None) -> Path:
    """执行训练并写 run 产物，返回 run 目录。"""
    import torch
    from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                              set_seed)

    train_cfg = cfg.get("train", {})
    data_dir = Path(cfg["output"]["dir"])
    run_dir = data_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    max_length = max_length or int(train_cfg.get("max_length", 16384))
    if smoke:
        max_length = min(max_length, 8192)  # §8：冒烟 = 20 steps @ 8K
    batch_cfg = train_cfg.get("batch", {})

    # 数据（构建产物；无则提示先跑 build_dataset）
    train_path, dev_path = data_dir / "train.parquet", data_dir / "dev.parquet"
    if not train_path.exists():
        raise SystemExit(f"缺少 {train_path}——先运行 python -m prm.build_dataset")
    train_ds = PrmParquetDataset(str(train_path))
    dev_ds = PrmParquetDataset(str(dev_path)) if dev_path.exists() else None

    # 类别权重：优先取 build manifest（train 统计口径一致），否则即时重算
    w_pos, w_neg = 1.0, 1.0
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        bm = json.loads(manifest_path.read_text(encoding="utf-8"))
        cw = bm.get("class_weights") or {}
        w_pos, w_neg = float(cw.get("w_pos", 1.0)), float(cw.get("w_neg", 1.0))
        logger.info("类别权重（build manifest）: w_pos=%.3f w_neg=%.3f", w_pos, w_neg)
    else:
        from prm.labeling import class_weights
        w_pos, w_neg = class_weights([int(s["label_binary"]) for s in train_ds])
        logger.info("类别权重（即时统计）: w_pos=%.3f w_neg=%.3f", w_pos, w_neg)

    # 模型 + collator
    base_model = train_cfg["base_model"]
    collator = VerdictCollator(tokenizer_path=base_model, max_length=max_length)
    verdict_pair = tuple(train_cfg.get("verdict_pair", ["Correct", "Incorrect"]))
    verdict_ids = resolve_verdict_ids(collator.tokenizer, verdict_pair)
    lora = train_cfg.get("lora")
    attn = train_cfg.get("attn_implementation", "flash_attention_2")
    if not torch.cuda.is_available() and "flash" in attn:
        attn = "sdpa"
        logger.info("无 CUDA → attn 实现降级 sdpa")
    scorer = VerdictScorer.from_pretrained(
        base_model, verdict_ids,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn, lora=lora)
    n_trainable = sum(p.numel() for p in scorer.trainable_parameters())
    logger.info("可训练参数: %.2fM（LoRA=%s）", n_trainable / 1e6, bool(lora))

    set_seed(int(train_cfg.get("seed", 42)))
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=int(batch_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(batch_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("lr", 1e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)) if not smoke else 1.0,
        max_steps=int(train_cfg.get("smoke_max_steps", 20)) if smoke else -1,
        bf16=bool(train_cfg.get("bf16", True)) and torch.cuda.is_available(),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
        eval_strategy="steps" if dev_ds else "no",
        eval_steps=int(train_cfg.get("eval_steps", 1000)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 1000)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        save_safetensors=True,
        load_best_model_at_end=bool(dev_ds),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "dev_auc")),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    callbacks = []
    if dev_ds and train_cfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(train_cfg["early_stopping_patience"])))

    trainer_cls = make_trainer_class()
    trainer = trainer_cls(
        model=scorer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        callbacks=callbacks,
        w_pos=w_pos, w_neg=w_neg,
    )
    trainer.train()

    # 保存最终产物（load_best_model_at_end → 此处权重 = dev AUC 最优）
    scorer.save_pretrained(str(run_dir))
    collator.tokenizer.save_pretrained(str(run_dir))

    best_metric = None
    state = getattr(trainer.state, "best_metric", None)
    if state is not None:
        best_metric = float(state)
    manifest = {
        "run": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_model,
        "verdict_pair": list(verdict_pair),
        "verdict_token_ids": list(verdict_ids),
        "max_length": max_length,
        "smoke": bool(smoke),
        "lora": lora,
        "w_pos": w_pos, "w_neg": w_neg,
        "template_version": TEMPLATE_VERSION,
        "template_hash": template_hash(),
        "train_data": str(train_path), "n_train": len(train_ds),
        "dev_data": str(dev_path) if dev_ds else None,
        "n_dev": len(dev_ds) if dev_ds else 0,
        "best_metric": best_metric,
        "seed": int(train_cfg.get("seed", 42)),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "truncation_stats.json").write_text(
        json.dumps(collator.stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("训练完成: %s（best auc=%s）", run_dir, best_metric)
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m prm.train_prm", description="PRM 训练（§8）")
    p.add_argument("--config", default="config/prm.yaml")
    p.add_argument("--run", required=True, help="run 名（outputs/prm/runs/<name>）")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：20 steps @ 8K（M4.2 验收路径）")
    p.add_argument("--max-length", type=int, default=None, help="覆盖 config 的 max_length")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    if args.run != "smoke" and torch.cuda.is_available():
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        total = torch.cuda.mem_get_info(0)[1] / 1e9
        logger.info("GPU0 显存空闲 %.1f / %.1f GB（若被占用请先询问用户，不自动停服务）",
                    free, total)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train(cfg, args.run, smoke=args.smoke, max_length=args.max_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
