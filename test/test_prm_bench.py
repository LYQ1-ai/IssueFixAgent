# SPDX-License-Identifier: BSD-3-Clause

"""scripts/bench_prm_step.py 的离线自检（torch 门控，CPU tiny 模型）。

基准脚本是"要不要开正式训练"的判据工具，跑错了会误导决策，故在 PRM 环境里用
tiny 模型把主流程（选样本 → 建模型 → 前向/前向+反向计时 → 写 JSON）跑通一遍。
不断言任何时间数字（CPU 上无意义），只断言**结构与口径**：
bs=1 选样不 padding、JSON 字段齐全、LoRA/内核开关被如实记录。
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

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

from prm.build_dataset import SCHEMA  # noqa: E402
from prm_torch_fixtures import build_tiny_model, build_tiny_tokenizer, make_tiny_messages  # noqa: E402

CORRECT_ID, INCORRECT_ID = 2, 3


def _row(i: int) -> dict:
    msgs = make_tiny_messages(1)
    for m in msgs:  # parquet 列 arguments = canonical dict 的 JSON 序列化
        for tc in m.get("tool_calls") or []:
            if isinstance(tc.get("function", {}).get("arguments"), dict):
                tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    return {
        "sample_id": f"dev-{i:04d}", "instance_id": f"inst-{i % 2:02d}",
        "repo": "repo", "split": "dev", "node_key": f"key-{i:04d}",
        "step_index": (i % 3) + 1, "step_count": 3,
        "messages": msgs, "label": float(i % 2),
        "label_binary": i % 2, "label_soft": float(i % 2),
        "label_source": "node_mc", "mc_score": float(i % 2),
        "n_rollouts": 5, "visits": 5, "in_pool": 1, "rendered_tokens": 60,
    }


@pytest.fixture(scope="module")
def bench_world(tmp_path_factory):
    from prm.modeling import VerdictScorer

    tmp = tmp_path_factory.mktemp("prm_bench")
    model_dir = tmp / "tiny-model"
    model_dir.mkdir(parents=True)
    tok = build_tiny_tokenizer()
    VerdictScorer(build_tiny_model(vocab_size=64), (CORRECT_ID, INCORRECT_ID)
                  ).save_pretrained(str(model_dir))
    tok.save_pretrained(str(model_dir))
    pq.write_table(pa.Table.from_pylist([_row(i) for i in range(4)], schema=SCHEMA),
                   tmp / "dev.parquet")
    cfg = {"output": {"dir": str(tmp)},
           "train": {"base_model": str(model_dir), "max_length": 512,
                     "verdict_pair": ["Correct", "Incorrect"], "lora": None,
                     "attn_implementation": "sdpa"}}
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"tmp": tmp, "cfg_path": tmp / "cfg.yaml"}


class TestBenchCli:
    def test_main_writes_json_bs1(self, bench_world, tmp_path):
        from scripts.bench_prm_step import main

        out = tmp_path / "bench.json"
        rc = main(["--config", str(bench_world["cfg_path"]),
                   "--length", "64", "--iters", "1", "--warmup", "0",
                   "--no-grad-ckpt", "--no-lora", "--json", str(out)])
        assert rc == 0
        res = json.loads(out.read_text(encoding="utf-8"))
        assert res["device"]["name"] == "cpu"          # 沙箱无 GPU
        assert res["grad_ckpt"] is False and res["lora"] is False
        assert res["use_kernels_requested"] is False
        assert res["attn_implementation_effective"] == "sdpa"
        row = res["rows"][0]
        assert row["target"] == 64
        assert row["n_tokens"] <= res["max_length"]     # 选样不 padding、不超长
        assert row["step_s"] > 0 and row["fwd_s"] > 0
        assert row["bwd_ratio"] > 0

    def test_pick_sample_respects_max_length(self, bench_world):
        from prm.data import PrmParquetDataset, VerdictCollator
        from scripts.bench_prm_step import _pick_sample

        cfg = yaml.safe_load(bench_world["cfg_path"].read_text(encoding="utf-8"))
        collator = VerdictCollator(tokenizer_path=cfg["train"]["base_model"], max_length=8)
        ds = PrmParquetDataset(str(bench_world["tmp"] / "dev.parquet"))
        with pytest.raises(SystemExit, match="没有 ≤ max_length"):
            _pick_sample(ds, collator, target=64, max_length=8)
