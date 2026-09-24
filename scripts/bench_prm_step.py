#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""PRM 单步吞吐基准（诊断用；不写训练产物、不动 state.db）。

**为什么需要它**（docs/prm_training_plan.md §14.6）：

- 同卡同长度的 dev **评估** = 2.32 条/s（≈0.43 s/条 @8K ≈ 150 TFLOPS 前向，正常）；
- 冒烟**训练** = 25 s/步（bs=1 @8K，grad_accum=16），比"前向×3"的预期慢约 20×。

两者一比就知道：异常在**反向**侧，不是"卡太慢"。本脚本把
「纯前向」/「前向+反向」分开计时，并给出 bf16 matmul 的 TFLOPS 基线
（判断卡是否被抢占/本身弱），用于逐一排除：

| 开关 | 想回答的问题 |
| --- | --- |
| `--grad-ckpt` / `--no-grad-ckpt` | 显存峰值只有 13 GB，重算是否纯属浪费（甚至因 fla 内核不可重算而爆炸） |
| `--use-kernels` | HF Hub 内核（fla / mamba-ssm）相对本地 fla 0.5.2 还有多少收益 |
| `--no-lora` | LoRA + 冻结基座的反向是否有额外开销 |
| `--length 8192 16384` | 单步成本随长度的增长（注意力二次项 vs 线性项） |

**padding 硬约束**：本脚本只用单样本（bs=1），按目标长度从 dev 里挑"最接近且不超过"
的真实样本，绝不构造 padding（§14.2）。

**⚠️ 峰值显存不可外推到训练**：这里只跑 1 个样本、1 步、没有优化器状态。真实训练
（16K × bs=1 × 32 层）关掉 `gradient_checkpointing` 后**第 1 步就 OOM**
（实测 75.06 GiB allocated / 进程 78.72 GiB，80 GB 卡）。本脚本的 `--no-grad-ckpt`
只能用于**相对比较**，不能当作改 config 的依据。

用法（宿主机；先 `set -a; source .env; set +a` 让 .env 里的 CUDA_VISIBLE_DEVICES 生效）：

    .venv-prm/bin/python scripts/bench_prm_step.py --length 8192 16384
    .venv-prm/bin/python scripts/bench_prm_step.py --no-grad-ckpt --use-kernels
    .venv-prm/bin/python scripts/bench_prm_step.py --json outputs/prm/bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:            # 允许直接 python scripts/xxx.py
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from prm.data import PrmParquetDataset, VerdictCollator  # noqa: E402
from prm.env import load_project_env  # noqa: E402
from prm.modeling import (VerdictScorer, kernelized_modules, resolve_verdict_ids,
                          verdict_loss)  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn, iters: int, warmup: int) -> tuple[float, float]:
    """返回 ``(中位单次秒数, 峰值显存GB)``；warmup 次不计时。"""
    for _ in range(warmup):
        fn()
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return statistics.median(times), peak


def _matmul_tflops(size: int = 4096, iters: int = 10) -> tuple[float, float]:
    """bf16 方阵 matmul 基线（卡的"体检"）：返回 ``(中位TFLOPS, 最快一次的TFLOPS)``。

    **关键读法**：这个数远低于该卡标称值（A800 bf16 ≈ 312 TFLOPS 峰值，4096³ 实测
    应 >150）说明**卡正被别的进程占用/降频**——此时本脚本测出的 step_s 全部不可信，
    必须先换空闲卡。``最快一次`` 最接近"分到干净时间片"时的能力；两者差距大 =
    时有时无的抢占。
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    a = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(size, size, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):                      # cuBLAS 算法选择/预热
        a @ b
    _sync()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        a @ b
        _sync()
        times.append(time.perf_counter() - t0)
    best = 2 * size ** 3 / min(times) / 1e12
    median = 2 * size ** 3 / statistics.median(times) / 1e12
    return median, best


def _pick_sample(dataset: PrmParquetDataset, collator: VerdictCollator,
                 target: int, max_length: int) -> dict:
    """挑一条 token 数最接近 target 且 ≤ max_length 的真实样本（不 padding）。

    优先用 parquet 的 ``rendered_tokens``（构建期就用真 tokenizer 算好的），缺失时
    退回 collator 的编码路径，保证与训练期**同一口径**。
    """
    best, best_gap = None, None
    for i in range(min(len(dataset), 400)):
        row = dataset[i]
        n = int(row.get("rendered_tokens") or 0) or collator._render_len(row["messages"])
        if n > max_length:
            continue
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = {"row": row, "n_tokens": n}, gap
    if best is None:
        raise SystemExit(f"dev 里没有 ≤ max_length({max_length}) 的样本，无法基准")
    return best


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    load_project_env()                       # 与 CLI 入口一致：早于 CUDA 初始化
    p = argparse.ArgumentParser(prog="bench_prm_step", description="PRM 单步吞吐基准")
    p.add_argument("--config", default="config/prm.yaml")
    p.add_argument("--data", default=None, help="dev parquet（默认 <output.dir>/dev.parquet）")
    p.add_argument("--model", default=None, help="基座路径（默认 config.train.base_model）")
    p.add_argument("--length", type=int, nargs="+", default=[8192, 16384],
                   help="目标 token 长度（按真实样本匹配，不 padding）")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--no-grad-ckpt", action="store_true", help="关闭 gradient checkpointing")
    p.add_argument("--no-lora", action="store_true", help="不包 LoRA（全量微调口径）")
    p.add_argument("--use-kernels", action="store_true", help="启用 HF Hub 内核")
    p.add_argument("--device", default=None, help="默认 cuda（有）/cpu")
    p.add_argument("--json", default=None, help="把结果写成 JSON")
    args = p.parse_args(argv)

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    model_path = args.model or train_cfg["base_model"]
    data_path = Path(args.data or (Path(cfg["output"]["dir"]) / "dev.parquet"))
    max_length = int(train_cfg.get("max_length", 16384))
    grad_ckpt = not args.no_grad_ckpt
    lora = None if args.no_lora else train_cfg.get("lora")

    has_cuda = torch.cuda.is_available()
    device = args.device or ("cuda" if has_cuda else "cpu")
    free_gb = total_gb = 0.0
    if has_cuda:
        free, total = torch.cuda.mem_get_info(0)
        free_gb, total_gb = free / 1e9, total / 1e9
    print(f"device={device} count={torch.cuda.device_count()} "
          f"name={torch.cuda.get_device_name(0) if has_cuda else 'cpu'} "
          f"显存空闲 {free_gb:.1f}/{total_gb:.1f} GB")
    print(f"torch={torch.__version__} model={model_path} data={data_path} "
          f"max_length={max_length} lora={bool(lora)} grad_ckpt={grad_ckpt} "
          f"use_kernels={args.use_kernels}")
    mm, mm_best = _matmul_tflops()
    if mm:
        verdict = ("正常" if mm_best > 120 else
                   "偏低：卡可能被抢占/降频，本脚本的 step_s 不可信")
        print(f"bf16 matmul 基线: 中位 {mm:.1f} / 最快 {mm_best:.1f} TFLOPS"
              f"（4096³，A800 峰值≈312）→ {verdict}")
        print(f"  进程独占判断：另开一个终端跑 nvidia-smi 看是否已有其他 python 占卡")

    dataset = PrmParquetDataset(str(data_path))
    collator = VerdictCollator(tokenizer_path=model_path, max_length=max_length)
    verdict_pair = tuple(train_cfg.get("verdict_pair", ["Correct", "Incorrect"]))
    verdict_ids = resolve_verdict_ids(collator.tokenizer, verdict_pair)

    # 建模型：请求内核时先试内核版，任何失败（缺包 / HF 不可达 / 编译失败）都回退。
    # 注意 `to_device_and_kernelize` 必须在 **try 内**——内核化真正发生在搬设备那一刻，
    # 本机实测就是在这一步抛 httpx.ConnectTimeout（HF Hub 不可达，§14.6.4）。
    scorer, kernels_ok = None, False
    for want in ([True, False] if args.use_kernels else [False]):
        try:
            scorer = VerdictScorer.from_pretrained(
                model_path, verdict_ids,
                torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
                attn_implementation=train_cfg.get("attn_implementation", "flash_attention_2"),
                lora=lora, use_kernels=want)
            # from_pretrained 不搬设备（模型停在 CPU）：必须先搬到 GPU，再在 GPU 上
            # 重新内核化——否则"开了内核"只是空操作（§14.6.4）
            scorer = scorer.to_device_and_kernelize(device)
            kernels_ok = bool(getattr(scorer, "use_kernels", False))
            if want and not kernels_ok:
                print("[warn] 请求了内核但模型未标记 use_kernels")
            break
        except Exception as e:                               # noqa: BLE001
            print(f"[warn] 内核路径失败（{type(e).__name__}: {e}）→ 回退参考实现")
            scorer = None
    if scorer is None:
        raise SystemExit("模型加载失败：内核与参考实现两条路都不通")
    kmods = kernelized_modules(scorer)
    attn_eff = str(getattr(getattr(scorer.backbone, "config", None),
                           "_attn_implementation", "?"))
    print(f"attn_implementation(实际)={attn_eff} use_kernels(实际)={kernels_ok} "
          f"内核化模块类型={len(kmods)}")
    if kmods:
        print(f"  例：{kmods[:3]}")

    if grad_ckpt:
        scorer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        scorer.enable_input_require_grads()
    scorer.train()

    rows = []
    for target in args.length:
        pick = _pick_sample(dataset, collator, target, max_length)
        batch = collator([pick["row"]])
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = torch.tensor([0.5], dtype=torch.float32, device=device)
        n_tok = int(mask.sum().item())
        w_pos = w_neg = 1.0

        with torch.no_grad():
            fwd_dt, _ = _timeit(lambda: scorer(ids, mask), args.iters, args.warmup)

        def _step():
            scorer.zero_grad(set_to_none=True)
            z = scorer(ids, mask)
            loss = verdict_loss(z, labels, w_pos, w_neg)
            loss.backward()

        step_dt, peak = _timeit(_step, args.iters, args.warmup)
        rows.append({
            "target": target, "n_tokens": n_tok,
            "fwd_s": fwd_dt, "fwd_tok_s": n_tok / fwd_dt,
            "step_s": step_dt, "step_tok_s": n_tok / step_dt,
            "bwd_ratio": step_dt / fwd_dt, "peak_gb": peak,
        })
        print(f"target={target:>6} 实际={n_tok:>6} tok | 前向 {fwd_dt:7.3f}s "
              f"({n_tok / fwd_dt:7.1f} tok/s) | 前向+反向 {step_dt:8.3f}s "
              f"({n_tok / step_dt:6.1f} tok/s) | 倍数 {step_dt / fwd_dt:5.2f}× | "
              f"峰值 {peak:.2f} GB")

    if rows:
        # 线性外推：全量 19,862 样本 / (bs=1×accum=16) 的每步耗时 → epoch 时长
        mid = rows[len(rows) // 2]
        n_epoch = 19862
        hours = n_epoch * mid["step_s"] / 3600
        print(f"\n外推（用 target={mid['target']} 的 {mid['step_s']:.1f}s/步）："
              f"1 epoch ≈ {hours:.1f} h，2 epochs ≈ {2 * hours:.1f} h")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "device": {"type": device,
                       "count": torch.cuda.device_count(),
                       "name": torch.cuda.get_device_name(0) if has_cuda else "cpu",
                       "mem_free_gb": free_gb, "mem_total_gb": total_gb,
                       "matmul_tflops_median": mm, "matmul_tflops_best": mm_best,
                       # 卡被抢占时 step_s 不可信，下游解读要带上这个标记
                       "contended_suspect": bool(mm_best and mm_best < 120)},
            "torch": torch.__version__, "model": str(model_path),
            "max_length": max_length, "lora": bool(lora), "grad_ckpt": grad_ckpt,
            "use_kernels_requested": args.use_kernels, "use_kernels_effective": kernels_ok,
            "kernelized_module_types": kmods,
            "attn_implementation_effective": attn_eff, "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
