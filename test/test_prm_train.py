# SPDX-License-Identifier: BSD-3-Clause

"""prm/train_prm.py 离线冒烟（torch 门控）：tiny 模型过完整 HF Trainer 循环。

验证训练机制本身（compute_loss / prediction_step / dev_auc 选优 / 产物齐全 /
run_manifest 字段），不依赖 GPU 与真实基座。风险点 = transformers 版本 API
差异，此测试在 PRM 环境最先暴露。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent))

pytest.importorskip("torch")
pytest.importorskip("transformers")

import pyarrow as pa
import pyarrow.parquet as pq  # noqa: E402

from prm.build_dataset import SCHEMA  # noqa: E402
from prm.prompts import template_hash  # noqa: E402
from prm.train_prm import train  # noqa: E402
from prm_torch_fixtures import (  # noqa: E402
    build_tiny_tokenizer,
    make_tiny_messages,
)

CORRECT_ID, INCORRECT_ID = 2, 3  # tiny tokenizer 的 verdict token ids


def _sample_row(i: int, split: str, label_binary: int, token_len: int) -> dict:
    msgs = make_tiny_messages(1)
    for m in msgs:  # parquet 列 arguments = canonical dict 的 JSON 序列化
        for tc in m.get("tool_calls") or []:
            if isinstance(tc.get("function", {}).get("arguments"), dict):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    return {
        "sample_id": f"{split}-{i:04d}",
        "instance_id": f"inst-{i % 4:02d}",
        "repo": "repo",
        "split": split,
        "node_key": f"key-{i:04d}",
        "step_index": (i % 3) + 1,
        "step_count": 3,
        "messages": msgs,
        "label": float(label_binary),
        "label_binary": label_binary,
        "label_soft": float(label_binary),
        "label_source": "node_mc",
        "mc_score": float(label_binary),
        "n_rollouts": 5,
        "visits": 5,
        "in_pool": 1,
        "rendered_tokens": token_len,
    }


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    """tiny 模型目录 + train/dev parquet + 训练 config。"""
    from prm.modeling import VerdictScorer
    from prm_torch_fixtures import build_tiny_model

    tmp = tmp_path_factory.mktemp("prm_train_smoke")
    tok = build_tiny_tokenizer()
    model_dir = tmp / "tiny-model"
    scorer = VerdictScorer(build_tiny_model(vocab_size=64), (CORRECT_ID, INCORRECT_ID))
    model_dir.mkdir(parents=True)
    scorer.save_pretrained(str(model_dir))   # → model.safetensors + config.json
    tok.save_pretrained(str(model_dir))

    def write_parquet(name: str, rows: list[dict]) -> Path:
        path = tmp / name
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        return path

    train_rows = [_sample_row(i, "train", i % 2, 800) for i in range(8)]
    dev_rows = [_sample_row(i, "dev", i % 2, 800) for i in range(4)]
    write_parquet("train.parquet", train_rows)
    write_parquet("dev.parquet", dev_rows)

    cfg = {
        "output": {"dir": str(tmp)},
        "train": {
            "base_model": str(model_dir),
            "max_length": 512,
            "verdict_pair": ["Correct", "Incorrect"],
            "lora": None,
            "batch": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 1},
            "lr": 1e-3,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": 1,
            "bf16": False,
            "gradient_checkpointing": False,
            "attn_implementation": "sdpa",
            "seed": 42,
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 2,
            "metric_for_best_model": "dev_auc",
            "greater_is_better": True,
            "early_stopping_patience": 3,
            "smoke_max_steps": 4,
        },
    }
    return {"tmp": tmp, "cfg": cfg, "tokenizer": tok, "model_dir": model_dir}


class TestTrainSmoke:
    def test_offline_training_roundtrip(self, tiny_world):
        run_dir = train(tiny_world["cfg"], "smoketest", smoke=True)
        assert (run_dir / "model.safetensors").exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "truncation_stats.json").exists()
        assert (run_dir / "tokenizer_config.json").exists() or (
            run_dir / "tokenizer.json").exists()

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["verdict_pair"] == ["Correct", "Incorrect"]
        assert manifest["verdict_token_ids"] == [CORRECT_ID, INCORRECT_ID]
        assert manifest["template_hash"] == template_hash()
        assert manifest["n_train"] == 8 and manifest["n_dev"] == 4
        assert manifest["best_metric"] is not None      # dev 评估实际跑过
        assert 0.0 <= manifest["best_metric"] <= 1.0
        assert manifest["env"]["torch"]

    def test_reload_and_score(self, tiny_world):
        """训练产物可经 eval_prm.load_scorer_for_run 复原并前向打分。"""
        import torch
        from prm.eval_prm import load_scorer_for_run

        run_dir = tiny_world["tmp"] / "runs" / "smoketest"
        scorer, manifest, collator = load_scorer_for_run(str(run_dir), device="cpu")
        assert manifest["run"] == "smoketest"
        batch = collator([{"messages": make_tiny_messages(1), "label": 0.0}])
        p = scorer.predict_proba(batch["input_ids"], batch["attention_mask"])
        assert p.shape == (1,) and 0.0 <= float(p[0]) <= 1.0

    def test_class_weights_from_build_manifest(self, tiny_world):
        """存在 outputs/prm/manifest.json 时 w_neg 取 build 统计（一致性口径）。"""
        tmp = tiny_world["tmp"]
        (tmp / "manifest.json").write_text(
            json.dumps({"class_weights": {"w_pos": 1.0, "w_neg": 2.5}}), encoding="utf-8")
        run_dir = train(tiny_world["cfg"], "smoketest2", smoke=True)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["w_neg"] == pytest.approx(2.5)
