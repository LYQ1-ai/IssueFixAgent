# SPDX-License-Identifier: BSD-3-Clause

"""M4.0 零训练侦察（docs/prm_training_plan.md §9.1）。

基座（未训练）+ 构建好的 dev 样本抽样 → 批量前向取 P(correct) → 对
``label_binary`` 算 AUC/Brier → ``outputs/prm/probe_report.{json,md}``。

作用：免费验证 verdict 方向先验——预期 AUC 明显 >0.5 才说明 Correct/Incorrect
方向可 Hijack；≈0.5 时换 ``Yes/No``、``对/错`` token 对复测（§12）。一次前向
同时支持多 token 对（同一 last-position logits 上取不同两 token 差，边际成本低）。

GPU 要求：A800（GPU1）；运行前 ``nvidia-smi`` 确认显存空闲，若被占用先询问
用户，不自动停任何服务（§1.2）。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from prm.data import PrmParquetDataset, VerdictCollator, tensors_from
from prm.metrics import brier_score, classification_metrics, log_loss, roc_auc
from prm.modeling import VerdictScorer, resolve_verdict_ids
from prm.prompts import template_hash

logger = logging.getLogger("prm.probe")

# §12 备选 token 对（首个为主对，其余自动复测）
DEFAULT_PAIRS = [["Correct", "Incorrect"], ["Yes", "No"], ["对", "错"]]


def parse_pairs(raw: Optional[list[str]]) -> list[tuple[str, str]]:
    """``["Correct,Incorrect", ...]`` → ``[("Correct","Incorrect"), ...]``。"""
    if not raw:
        return [tuple(p) for p in DEFAULT_PAIRS]
    out = []
    for item in raw:
        parts = [p for p in item.split(",") if p]
        if len(parts) != 2:
            raise ValueError(f"--verdict-pairs 项需形如 'Correct,Incorrect'，得到 {item!r}")
        out.append((parts[0], parts[1]))
    return out


def score_dataset(scorer: VerdictScorer, collator: VerdictCollator, items: list[dict],
                  batch_size: int, device: str) -> np.ndarray:
    """批量前向 → p = σ(z)（predict_proba 内部 no_grad）。"""
    import torch
    probs = []
    for i in range(0, len(items), batch_size):
        batch = collator(items[i:i + batch_size])
        t = tensors_from(batch, device=device)
        probs.append(scorer.predict_proba(t["input_ids"], t["attention_mask"]).float().cpu())
    return torch.cat(probs).numpy() if probs else np.zeros(0)


def probe(model_path: str, dev_parquet: str, out_dir: str, *, n_samples: int = 500,
          seed: int = 42, pairs: Optional[list[tuple[str, str]]] = None,
          batch_size: int = 8, max_length: int = 16384,
          device: str = "cuda", attn_implementation: str = "sdpa",
          config: Optional[dict] = None) -> dict:
    """跑 M4.0 侦察并写报告，返回报告 dict。"""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs or [tuple(p) for p in DEFAULT_PAIRS]

    ds = PrmParquetDataset(dev_parquet)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    items = [ds[int(i)] for i in sorted(idx)]
    logger.info("probe: dev 抽样 %d/%d 条 → %s", len(items), len(ds), out)

    tok = VerdictCollator(tokenizer_path=model_path, max_length=max_length).tokenizer
    scorer = VerdictScorer.from_pretrained(
        model_path, (0, 1), torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation).to(device)
    scorer.eval()
    pair_ids = {p: resolve_verdict_ids(tok, p) for p in pairs}

    collator = VerdictCollator(tokenizer_path=model_path, max_length=max_length)
    y_true = np.array([it["label_binary"] for it in items], dtype=np.int64)

    results = {}
    for pair, (id_c, id_i) in pair_ids.items():
        scorer.set_verdict_ids((id_c, id_i))  # 换对并重注册冻结 hook
        scores = score_dataset(scorer, collator, items, batch_size, device)
        results[", ".join(pair)] = {
            "verdict_ids": [id_c, id_i],
            "auc": roc_auc(y_true, scores),
            "brier": brier_score(y_true, scores),
            "log_loss": log_loss(y_true, scores),
            "n": int(len(y_true)),
            "pos_rate": float(y_true.mean()),
            **{k: v for k, v in classification_metrics(y_true, scores).items()
               if k in ("accuracy", "precision", "recall", "f1")},
        }
        logger.info("probe[%s]: AUC=%.4f Brier=%.4f", pair, results[", ".join(pair)]["auc"],
                    results[", ".join(pair)]["brier"])

    best = max(results.items(), key=lambda kv: (kv[1]["auc"] if np.isfinite(kv[1]["auc"]) else -1))
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_path,
        "dev_parquet": dev_parquet,
        "n_samples": len(items),
        "seed": seed,
        "max_length": max_length,
        "template_hash": template_hash(),
        "pairs": results,
        "best_pair": best[0],
        "verdict_usable": bool(np.isfinite(best[1]["auc"]) and best[1]["auc"] > 0.55),
        "note": "AUC 明显 >0.5 说明该 token 对方向可 Hijack；≈0.5 时换备选对复测（§12）",
    }
    (out / "probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "probe_report.md").write_text(probe_report_md(report), encoding="utf-8")
    logger.info("probe 完成: best=%s AUC=%.4f（verdict_usable=%s）",
                report["best_pair"], best[1]["auc"], report["verdict_usable"])
    return report


def probe_report_md(report: dict) -> str:
    lines = [
        "# M4.0 零训练侦察报告",
        "",
        f"- 模型: `{report['model']}`（未训练基座）",
        f"- dev 抽样: {report['n_samples']} 条（seed={report['seed']}）"
        f" | max_length={report['max_length']} | template `{report['template_hash'][:12]}…`",
        "",
        "| token 对 | verdict ids | AUC | Brier | LogLoss | Acc | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in report["pairs"].items():
        ids = r["verdict_ids"]
        lines.append(f"| {name} | {ids} | {r['auc']:.4f} | {r['brier']:.4f} "
                     f"| {r['log_loss']:.4f} | {r['accuracy']:.4f} | {r['f1']:.4f} |")
    lines += [
        "",
        f"**结论**: 最佳 token 对 = `{report['best_pair']}`"
        f"（verdict_usable={report['verdict_usable']}，判据 AUC>0.55）。",
        "",
        f"> {report['note']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m prm.probe",
                                description="M4.0 零训练侦察（§9.1）")
    p.add_argument("--config", default="config/prm.yaml")
    p.add_argument("--model", default=None, help="基座路径（默认 config.train.base_model）")
    p.add_argument("--dev", default=None, help="dev parquet（默认 outputs/prm/dev.parquet）")
    p.add_argument("--out", default=None, help="报告输出目录（默认 outputs/prm）")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--verdict-pairs", nargs="*", default=None,
                   help="token 对列表，如 'Correct,Incorrect' 'Yes,No' '对,错'"
                        "（缺省 = 三对全测）")
    p.add_argument("--device", default=None, help="cuda / cuda:1 / cpu")
    p.add_argument("--attn", default=None, help="flash_attention_2 / sdpa（缺省 sdpa）")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model = args.model or train_cfg["base_model"]
    dev = args.dev or str(Path(cfg["output"]["dir"]) / "dev.parquet")
    out = args.out or cfg["output"]["dir"]
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    report = probe(
        model, dev, out,
        n_samples=args.n_samples or eval_cfg.get("probe", {}).get("n_samples", 500),
        seed=eval_cfg.get("probe", {}).get("seed", 42),
        pairs=parse_pairs(args.verdict_pairs),
        batch_size=args.batch_size or eval_cfg.get("batch_size", 8),
        max_length=args.max_length or train_cfg.get("max_length", 16384),
        device=device,
        attn_implementation=args.attn or "sdpa",
        config=cfg,
    )
    return 0 if report["verdict_usable"] else 1  # 不可用 → 非零退出提示换 token 对


if __name__ == "__main__":
    raise SystemExit(main())
