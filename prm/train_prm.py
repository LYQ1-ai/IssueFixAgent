# SPDX-License-Identifier: BSD-3-Clause

"""PRM 训练 CLI（docs/prm_training_plan.md §8，M4.2）。

- 基座 Qwen3.5-4B + PEFT LoRA（r=16/α=32/dropout=0.05，q/k/v/o/gate/up/down）；
- 损失 = 加权 soft-BCE（w_pos/w_neg 来自 build manifest 的 train 统计）；
- 评估 = dev AUC（``metric_for_best_model=dev_auc``，选优 + early stopping）；
- 产物 ``outputs/prm/runs/<name>/``：``model.safetensors``（= LoRA adapter 权重，
  另存 adapter_model.safetensors/adapter_config.json）+ ``run_manifest.json``
  （verdict ids / max_length / template_hash / w_neg / 数据版本）+ tokenizer
  + trainer_state.json + truncation_stats.json。

GPU 纪律（§1.2/§14.5）：**用哪张卡由仓库根 ``.env`` 的 ``CUDA_VISIBLE_DEVICES``
决定**（``prm`` 包导入时装载，进程内 ``cuda:0`` 即选中卡；本 CLI 无 --device，
换卡请改 .env 或 ``CUDA_VISIBLE_DEVICES=N python -m prm.train_prm ...``）；
启动前 ``nvidia-smi`` 确认显存空闲，被占用
先询问用户，不自动停任何服务。冒烟路径：``--smoke``（20 steps @ 8K）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from prm.data import PrmParquetDataset, VerdictCollator, oversize_indices
from prm.env import describe as describe_env, load_project_env
from prm.metrics import brier_score, roc_auc
from prm.modeling import (VerdictScorer, kernelized_modules, resolve_verdict_ids,
                          verdict_loss)
from prm.prompts import TEMPLATE_VERSION, template_hash

logger = logging.getLogger("prm.train")

# 解析后的 SwanLab 配置（不写进 SWANLAB_PROJECT 环境变量，见 _setup_reporting）
_SWANLAB_PROJECT: Optional[str] = None
_SWANLAB_WORKSPACE: Optional[str] = None


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

        def _save(self, output_dir, state_dict=None):
            """覆写：走 VerdictScorer.save_pretrained（tie 权重去重 + adapter 镜像）。

            transformers 5.x 的默认 `_save` 直接 ``safetensors.save_file(state_dict)``
            不处理共享张量——``tie_word_embeddings=True``（F1）会报
            "Some tensors share memory"。模型的 save_pretrained 负责去重。
            """
            self.model.save_pretrained(output_dir)

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

def _warmup_steps(warmup_ratio: float, train_ds, per_device_bs: int, accum: int,
                  epochs: float, max_steps: int) -> int:
    """换算 warmup 步数（transformers 5.x 移除 warmup_ratio）。

    总步数与 Trainer 同口径：``ceil(N / (bs·accum)) · epochs``（或显式 max_steps）；
    warmup ≥ 1 步。
    """
    if max_steps > 0:
        total = max_steps
    else:
        import math
        steps_per_epoch = math.ceil(len(train_ds) / max(1, per_device_bs * accum))
        total = max(1, int(steps_per_epoch * epochs))
    return max(1, int(round(warmup_ratio * total)))


def _swanlab_credential_paths() -> list:
    """swanlab 登录凭据可能的落点（0.10 实测，按优先级）。

    ``swanlab login``（``save="root"``）→ ``~/.swanlab/.netrc``；
    ``swanlab login --local``（``save="local"``）→ ``./.swanlab/.netrc``；
    另兼容标准 ``~/.netrc``（``SWANLAB_API_KEY`` 之外的老写法）。
    """
    return [Path.home() / ".swanlab" / ".netrc",
            Path.cwd() / ".swanlab" / ".netrc",
            Path.home() / ".netrc"]


def _setup_reporting(train_cfg: dict, run_name: str, data_dir: Path) -> list[str]:
    """解析 ``train.report_to`` → HF ``report_to`` 列表，并初始化 SwanLab run。

    SwanLab 上报由 transformers 内置 ``SwanLabCallback`` 完成，但**必须由我们先
    ``swanlab.init``**，原因有两个上游缺陷（swanlab 0.10.0 + transformers 5.17.0，
    2026-09-11 实测）：

    1. 回调以 ``swanlab.get_run() is None`` 判断是否已初始化，而 0.10 的
       ``get_run()`` 在无 run 时**抛 RuntimeError** → ``on_train_begin`` 直接崩；
       预先建好 run 即可让回调走「已初始化」分支。
    2. ``SWANLAB_PROJECT`` 环境变量的字段类型是嵌套模型，写成普通字符串会让
       ``swanlab.init`` 抛 ``SettingsError``（只接受 ``'{"name": "x"}'`` 形式），
       故 project 只经 ``init(project=...)`` 传入，并清理环境里的非 JSON 写法。

    另外：初始化一旦失败，必须把 ``swanlab`` 从 ``report_to`` 摘掉，否则回调仍会
    因缺陷 1 抛异常、把数小时的训练直接打断——记录器绝不能拖垮训练。
    """
    report_to = [str(r) for r in (train_cfg.get("report_to") or []) if r]
    if "swanlab" not in report_to:
        return report_to
    global _SWANLAB_PROJECT, _SWANLAB_WORKSPACE
    if importlib.util.find_spec("swanlab") is None:
        raise SystemExit("config 里 report_to 含 swanlab，但环境未安装："
                         "pip install swanlab（或改回 report_to: []）")
    sl = train_cfg.get("swanlab") or {}
    project = str(sl.get("project") or "CodeAgentRL")
    workspace = sl.get("workspace")
    _SWANLAB_PROJECT = project
    _SWANLAB_WORKSPACE = str(workspace) if workspace else None
    # mode / log_dir 走环境变量（swanlab 认这两个）；project 见上方缺陷 2，不写环境变量
    os.environ.setdefault("SWANLAB_MODE", str(sl.get("mode") or "local"))
    os.environ.setdefault("SWANLAB_LOG_DIR",
                          str(sl.get("log_dir") or (data_dir / "swanlog")))
    env_project = os.environ.pop("SWANLAB_PROJECT", None)
    if env_project and not env_project.lstrip().startswith("{"):
        logger.warning("已移除环境变量 SWANLAB_PROJECT=%r（swanlab 0.10 只接受 JSON 对象，"
                       "普通字符串会 SettingsError）；project 取 config 的 %r",
                       env_project, project)
    for k in ("api_host", "web_host"):
        v = sl.get(k)
        if v:
            os.environ.setdefault(f"SWANLAB_{k.upper()}", str(v))
    mode = os.environ["SWANLAB_MODE"]
    log_dir = os.environ["SWANLAB_LOG_DIR"]
    logger.info("SwanLab 已启用：project=%s workspace=%s mode=%s log_dir=%s run=%s",
                project, workspace, mode, log_dir, run_name)
    if mode == "cloud" and not os.environ.get("SWANLAB_API_KEY") \
            and not any(p.exists() for p in _swanlab_credential_paths()):
        # 官方流程第 2 步是 `swanlab login`（凭据默认落 ~/.swanlab/.netrc）。两者都没有时
        # swanlab.init 会尝试交互式登录：nohup 下 EOF 报错、前台则可能一直等待输入。
        # 直接禁用更安全，且给出明确指引。
        logger.error("mode=cloud 但既无 SWANLAB_API_KEY 也无 swanlab 登录凭据 "
                     "(%s) → 本次禁用 SwanLab 上报，训练继续。请先执行 `swanlab login`，"
                     "或改用 swanlab.mode: local",
                     " / ".join(str(p) for p in _swanlab_credential_paths()))
        return [r for r in report_to if r != "swanlab"]
    try:
        import swanlab
        run = None
        try:
            run = swanlab.get_run()          # 0.10 在无 run 时抛异常，不是返回 None
        except Exception:
            run = None
        if run is None:
            init_cfg = {k: v for k, v in train_cfg.items()
                        if k not in ("swanlab", "report_to")}
            init_cfg["run"] = run_name
            init_kwargs = {"project": project, "name": run_name, "mode": mode,
                           "log_dir": log_dir, "config": init_cfg}
            if workspace:
                init_kwargs["workspace"] = str(workspace)
            swanlab.init(**init_kwargs)
            logger.info("SwanLab run 已初始化（project=%s 实验名=%s）", project, run_name)
    except Exception as e:
        logger.error("SwanLab 初始化失败，已禁用该记录器（训练继续）：%s", e)
        report_to = [r for r in report_to if r != "swanlab"]
    return report_to


def _finish_reporting(report_to: list[str]) -> Optional[dict]:
    """收尾记录器（SwanLab 需显式 finish），返回写入 manifest 的信息。"""
    if "swanlab" not in report_to:
        return None
    info: dict = {"project": _SWANLAB_PROJECT or os.environ.get("SWANLAB_PROJECT"),
                  "workspace": _SWANLAB_WORKSPACE,
                  "mode": os.environ.get("SWANLAB_MODE"),
                  "log_dir": os.environ.get("SWANLAB_LOG_DIR")}
    try:
        import swanlab
        run = swanlab.get_run()
        if run is not None:
            info["run_id"] = getattr(run, "id", None)
            public = getattr(run, "public", None)
            cloud = getattr(public, "cloud", None)
            info["url"] = getattr(cloud, "experiment_url", None)
        swanlab.finish()
    except Exception as e:  # pragma: no cover - 记录器失败不应影响训练产物
        logger.warning("SwanLab 收尾失败（不影响训练产物）：%s", e)
    return info


def _resolve_eval_max(train_cfg: dict, smoke: bool,
                      cli_value: Optional[int] = None) -> Optional[int]:
    """dev 评估条数上限的优先级：CLI > 冒烟上限 > config。

    - config ``train.eval_max_samples``（``null`` = 全量 4,864，bs=1 下约 35 min/次）；
    - 冒烟用它的小上限 ``train.smoke_eval_max_samples``（默认 64）——否则一次冒烟
      的钱几乎全花在"评全量 dev"上（实测 20 步训练 10 min + 全量评估 35 min）；
    - CLI ``--eval-max-samples`` 显式给出时**覆盖**以上两者（含冒烟）。
    """
    eval_max = train_cfg.get("eval_max_samples")
    if smoke:
        cap = int(train_cfg.get("smoke_eval_max_samples", 64) or 0)
        if cap > 0 and (not eval_max or int(eval_max) > cap):
            eval_max = cap
    if cli_value is not None:
        eval_max = None if int(cli_value) <= 0 else int(cli_value)
    return eval_max


def _drop_oversize(ds, collator, data_dir: Path, split: str):
    """剔除 §7.2 预算不足的样本；返回 ``(dataset, 跳过的 sample_id 列表)``。

    这些样本的**被判定步本身**就超过 ``max_length``（本次实测 19,862 条里 1 条，
    43,769 token、末步 31,230），在给定长度下无法使用。若放任 collator 抛错，异常
    发生在 DataLoader worker 内 → 整个训练进程崩掉（2026-09-15：step 1146，白跑 10 h）。
    扫描结果缓存在 ``<output.dir>/oversize_skip_<split>.json``（按 parquet 指纹校验）。
    """
    from torch.utils.data import Subset

    cache = data_dir / f"oversize_skip_{split}.json"
    bad = oversize_indices(ds, collator, cache_path=str(cache))
    if not bad:
        return ds, []
    keep = [i for i in range(len(ds)) if i not in set(bad)]
    logger.info("%s：剔除 %d 条无法截断的样本（%d → %d，§7.2 预算不足；清单见 %s）",
                split, len(bad), len(ds), len(keep), cache.name)
    return Subset(ds, keep), [ds[i]["sample_id"] for i in bad]


def _load_scorer(base_model: str, verdict_ids: tuple[int, int], *,
                 torch_dtype, attn: str, lora: Optional[dict],
                 use_kernels: bool, device: str = "cuda"):
    """建打分器；``use_kernels`` 失败即回退参考实现（返回 ``(scorer, 实际值)``）。

    ``kernels`` 未安装时 ``from_pretrained(use_kernels=True)`` 抛 ``ValueError``；
    Hub 内核下载/编译失败则抛其它异常。两种都不该让训练起不来——参考实现是
    对的、只是慢，故此处 fail-open（同 ``attn`` 的降级策略，§14.2）。

    加载后**显式搬到 device 并重新内核化**：Hub 内核按设备类型匹配，若在 CPU 上
    加载、之后只由 Trainer 搬张量，内核就永远不生效（§14.6.4）。
    """
    def _build(kernels: bool):
        scorer = VerdictScorer.from_pretrained(
            base_model, verdict_ids, torch_dtype=torch_dtype,
            attn_implementation=attn, lora=lora, use_kernels=kernels)
        return scorer.to_device_and_kernelize(device)

    if use_kernels:
        try:
            scorer = _build(True)
            n_kernel = len(kernelized_modules(scorer))
            logger.info("已启用 HF Hub 内核（use_kernels=True）：设备=%s 内核化模块类型=%d %s",
                        device, n_kernel, kernelized_modules(scorer)[:4])
            return scorer, True
        except Exception as e:      # noqa: BLE001 - 任何内核问题都退回参考实现
            logger.warning("启用 HF Hub 内核失败（%s: %s）→ 回退 torch 参考实现"
                           "（loss 不变，吞吐可能差一个数量级，§14.6）",
                           type(e).__name__, e)
    return _build(False), False


def train(cfg: dict, run_name: str, *, smoke: bool = False,
          max_length: Optional[int] = None, resume: bool = False,
          eval_max_samples: Optional[int] = None) -> Path:
    """执行训练并写 run 产物，返回 run 目录。

    ``resume=True`` 时从 ``runs/<name>/checkpoints/`` 下编号最大的、含
    ``trainer_state.json`` 的检查点续训；找不到则告警并从头训练（正式训练跑
    数小时，进程中断不该从头再来）。
    """
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
    batch_size = int(batch_cfg.get("per_device_train_batch_size", 1))
    if batch_size != 1:
        raise SystemExit(
            f"per_device_train_batch_size 必须为 1（当前 {batch_size}）：Qwen3.5 混合架构"
            "（causal conv + gated delta rule）对 pad 前缀敏感，多样本批的 padding 会"
            "改变打分；依据见 docs/prm_training_plan.md §14.2")

    # 数据（构建产物；无则提示先跑 build_dataset）
    train_path, dev_path = data_dir / "train.parquet", data_dir / "dev.parquet"
    if not train_path.exists():
        raise SystemExit(f"缺少 {train_path}——先运行 python -m prm.build_dataset")
    train_ds = PrmParquetDataset(str(train_path))
    dev_ds = PrmParquetDataset(str(dev_path)) if dev_path.exists() else None
    n_dev_full = len(dev_ds) if dev_ds is not None else 0

    base_model = train_cfg["base_model"]
    collator = VerdictCollator(tokenizer_path=base_model, max_length=max_length)
    # §7.2 边界样本（被判定步本身超预算）必须在开训前剔除：collator 是在 DataLoader
    # worker 里跑的，抛错会直接杀掉整个训练（2026-09-15 实测：19,862 条里 1 条，
    # step 1146 崩，白跑 10 h）。扫描结果按 parquet 指纹缓存，重复开训不再重扫。
    train_ds, skipped_train = _drop_oversize(train_ds, collator, data_dir, "train")
    skipped_dev: list[str] = []
    if dev_ds is not None:
        dev_ds, skipped_dev = _drop_oversize(dev_ds, collator, data_dir, "dev")

    # dev 评估子集（可选）：bs=1 下每次评估 = n_dev 次单条前向，全量 4.8k 条会显著
    # 拖慢训练；设 eval_max_samples 可只评前 N 条（确定性取前 N，便于跨 run 比较）。
    n_dev_usable = len(dev_ds) if dev_ds is not None else 0
    eval_max = _resolve_eval_max(train_cfg, smoke, eval_max_samples)
    if dev_ds is not None and eval_max:
        from torch.utils.data import Subset
        n_take = min(int(eval_max), len(dev_ds))
        dev_ds = Subset(dev_ds, list(range(n_take)))
        logger.info("dev 评估只用前 %d 条（train.eval_max_samples；可用 %d/%d，"
                    "另有 %d 条因 §7.2 预算不足被跳过）",
                    n_take, n_dev_usable, n_dev_full, len(skipped_dev))

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

    # 模型（collator 在数据段就已构造：它决定了哪些样本能进训练集）
    verdict_pair = tuple(train_cfg.get("verdict_pair", ["Correct", "Incorrect"]))
    verdict_ids = resolve_verdict_ids(collator.tokenizer, verdict_pair)
    lora = train_cfg.get("lora")
    attn = train_cfg.get("attn_implementation", "flash_attention_2")
    attn_requested = attn
    if "flash" in attn:
        # 降级条件二选一：无 CUDA，或环境未安装 flash-attn（config 注释承诺的
        # “不可用自动回退 sdpa”）。缺 flash_attn 时若不降级，from_pretrained 会直接抛错。
        has_cuda = torch.cuda.is_available()
        has_flash = importlib.util.find_spec("flash_attn") is not None
        if not has_cuda or not has_flash:
            logger.info("attn 实现降级 sdpa（CUDA=%s, flash_attn=%s）", has_cuda, has_flash)
            attn = "sdpa"
    scorer, use_kernels = _load_scorer(
        base_model, verdict_ids,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn=attn, lora=lora, use_kernels=bool(train_cfg.get("use_kernels", False)),
        device="cuda" if torch.cuda.is_available() else "cpu")
    n_trainable = sum(p.numel() for p in scorer.trainable_parameters())
    logger.info("可训练参数: %.2fM（LoRA=%s）", n_trainable / 1e6, bool(lora))
    attn_effective = str(getattr(getattr(scorer.backbone, "config", None),
                                 "_attn_implementation", attn))
    logger.info("attn 实现: 请求=%s 实际=%s | Hub 内核=%s", attn_requested, attn_effective,
                use_kernels)

    set_seed(int(train_cfg.get("seed", 42)))
    accum = int(batch_cfg.get("gradient_accumulation_steps", 16))
    epochs = float(train_cfg.get("num_train_epochs", 2)) if not smoke else 1.0
    smoke_max_steps = int(train_cfg.get("smoke_max_steps", 20)) if smoke else -1
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (batch_size * accum)))
    logger.info("计划：%d 优化步/epoch（bs=%d × accum=%d = %d 样本/步），epochs=%s，"
                "总优化步=%s，每 %d 步评 %s 条 dev",
                steps_per_epoch, batch_size, accum, batch_size * accum,
                epochs, smoke_max_steps if smoke else steps_per_epoch * epochs,
                int(train_cfg.get("eval_steps", 1000)),
                len(dev_ds) if dev_ds is not None else 0)
    report_to = _setup_reporting(train_cfg, run_name, data_dir)
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,   # 训练中 dev 评估同样受 padding 限制（§14.2）
        gradient_accumulation_steps=accum,
        learning_rate=float(train_cfg.get("lr", 1e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=_warmup_steps(float(train_cfg.get("warmup_ratio", 0.05)),
                                   train_ds, batch_size, accum, epochs, smoke_max_steps),
        num_train_epochs=epochs,
        max_steps=smoke_max_steps,
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
        load_best_model_at_end=bool(dev_ds),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "dev_auc")),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        logging_steps=10,
        report_to=report_to,
        run_name=run_name,
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

    resume_from: Optional[str] = None
    if resume:
        ckpt_root = run_dir / "checkpoints"
        cands = [p for p in ckpt_root.glob("checkpoint-*")
                 if (p / "trainer_state.json").exists()] if ckpt_root.exists() else []
        if cands:
            resume_from = str(max(cands, key=lambda p: int(p.name.rsplit("-", 1)[-1])))
            logger.info("断点续训：%s", resume_from)
        else:
            logger.warning("--resume 已指定但 %s 下无可用检查点 → 从头训练", ckpt_root)
    trainer.train(resume_from_checkpoint=resume_from)

    # 保存最终产物（load_best_model_at_end → 此处权重 = dev AUC 最优）
    scorer.save_pretrained(str(run_dir))
    collator.tokenizer.save_pretrained(str(run_dir))
    reporting_info = _finish_reporting(args.report_to)

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
        "resumed_from": resume_from,
        "report_to": list(args.report_to),
        "swanlab": reporting_info,
        "lora": lora,
        "w_pos": w_pos, "w_neg": w_neg,
        "template_version": TEMPLATE_VERSION,
        "template_hash": template_hash(),
        "train_data": str(train_path), "n_train": len(train_ds),
        "dev_data": str(dev_path) if dev_ds else None,
        "n_dev": n_dev_full,
        "n_dev_eval": len(dev_ds) if dev_ds else 0,
        "eval_max_samples": eval_max,
        # §7.2 边界样本：被判定步本身超预算 → 跳过（不跳会崩在 DataLoader worker 里）
        "oversize_skipped": {
            "n_train": len(skipped_train), "n_dev": len(skipped_dev),
            "train_sample_ids": skipped_train[:20], "dev_sample_ids": skipped_dev[:20],
        },
        "attn_implementation": attn_requested,
        "attn_implementation_effective": attn_effective,
        "use_kernels": use_kernels,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "steps_per_epoch": steps_per_epoch,
        "best_metric": best_metric,
        "seed": int(train_cfg.get("seed", 42)),
        "peak_gpu_memory": (
            {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
             "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
            if torch.cuda.is_available() else None),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
    }
    # 有效配置快照（§8 产出表）：run 目录自包含，便于复算/复现
    cfg_snapshot = json.loads(json.dumps(cfg))          # cfg 只含 JSON 化类型
    cfg_snapshot.setdefault("train", {})["max_length"] = max_length
    cfg_snapshot["run"] = {"name": run_name, "smoke": bool(smoke),
                           "resumed_from": resume_from,
                           "attn_implementation_effective": attn_effective,
                           "use_kernels": use_kernels,
                           "eval_max_samples_effective": eval_max}
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg_snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    p.add_argument("--resume", action="store_true",
                   help="从 runs/<name>/checkpoints/ 编号最大的检查点续训（无则从头）")
    p.add_argument("--eval-max-samples", type=int, default=None,
                   help="dev 评估条数上限（覆盖 config 与冒烟上限；<=0 = 全量 4,864）")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_project_env()          # 入口装载 .env（GPU/CUDA 选择），早于 torch/CUDA 初始化
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    logger.info("GPU 选择: %s", describe_env())          # .env / 环境变量决定的映射
    if args.run != "smoke" and torch.cuda.is_available():
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        total = torch.cuda.mem_get_info(0)[1] / 1e9
        logger.info("可见卡 0 显存空闲 %.1f / %.1f GB（若被占用请先询问用户，不自动停服务）",
                    free, total)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train(cfg, args.run, smoke=args.smoke, max_length=args.max_length,
          resume=args.resume, eval_max_samples=args.eval_max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
