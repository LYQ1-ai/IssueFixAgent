# SPDX-License-Identifier: BSD-3-Clause

"""prm/data.py 单测（docs/prm_training_plan.md §10 test_prm_data 行）。

覆盖：截断顺序（保上下文/指令、整步删除较早步保被判定步、marker）、
tool content 尾部截断（保 returncode+前 N token）、左 padding 逐条 vs 批量
token 序列一致、collator 张量形状、预算下限报错、截断统计。
需 torch/transformers（离线 tiny tokenizer，见 test/prm_torch_fixtures.py）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).parent))
from prm_torch_fixtures import (  # noqa: E402
    build_tiny_tokenizer,
    make_long_output,
    make_tiny_messages,
)

from prm.data import (  # noqa: E402
    PrmParquetDataset,
    VerdictCollator,
    load_messages,
)
from prm.prompts import TRUNCATION_MARKER  # noqa: E402


@pytest.fixture
def tok():
    return build_tiny_tokenizer()


def _make_collator(tok, max_length: int) -> VerdictCollator:
    """注入离线 tiny tokenizer（省去磁盘路径；正式训练经 tokenizer_path 构造）。"""
    return VerdictCollator(tokenizer_path=None, max_length=max_length, tokenizer=tok)


# ---------------------------------------------------------------------------
# load_messages（parquet → 可渲染）
# ---------------------------------------------------------------------------

class TestLoadMessages:
    def test_arguments_json_string_parsed(self):
        raw = [{"role": "assistant", "content": "",
                "tool_calls": [{"id": "c", "type": "function",
                                "function": {"name": "bash",
                                             "arguments": '{"command": "ls"}'}}]}]
        out = load_messages(raw)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}

    def test_idempotent_on_canonical_form(self):
        msgs = make_tiny_messages(1)
        assert load_messages(load_messages(msgs)) == load_messages(msgs)

    def test_plain_fields_untouched(self):
        raw = [{"role": "tool", "tool_call_id": "c", "content": "x"}]
        assert load_messages(raw) == raw


# ---------------------------------------------------------------------------
# 无截断路径 / 张量形状 / padding
# ---------------------------------------------------------------------------

class TestNoTruncation:
    def test_short_sample_passthrough(self, tok):
        c = _make_collator(tok, 4096)
        msgs = make_tiny_messages(2)
        out = c.truncate_messages(msgs)
        assert out is msgs or out == msgs  # 未截断，原样返回

    def test_single_sample_batch_no_padding(self, tok):
        """单样本批（唯一允许的形式）：无 padding，张量形状/标签正确。"""
        c = _make_collator(tok, 4096)
        item = {"messages": make_tiny_messages(3), "label": 0.2}
        out = c([item])
        n = len(c._encode(item["messages"]))
        assert out["input_ids"].shape == (1, n)
        assert out["attention_mask"].tolist() == [[1] * n]      # 无任何 pad
        assert out["labels"].tolist() == pytest.approx([0.2])

    def test_multi_sample_batch_rejected(self, tok):
        """多样本批必须报错（Qwen3.5 混合层对 pad 前缀敏感，见计划书 §14.2）。"""
        c = _make_collator(tok, 4096)
        with pytest.raises(ValueError, match="只接受单样本批"):
            c([{"messages": make_tiny_messages(1), "label": 0.8},
               {"messages": make_tiny_messages(3), "label": 0.2}])

    def test_single_vs_batch_token_ids_identical(self, tok):
        """单样本批的 token 序列与直接编码一致（§7.1-3 的前提）。"""
        c = _make_collator(tok, 4096)
        msgs = make_tiny_messages(2)
        single = c._encode(c.truncate_messages(msgs))
        batched = c([{"messages": msgs, "label": 1.0}])
        assert batched["input_ids"][0].tolist() == list(single)
        assert batched["attention_mask"][0].tolist() == [1] * len(single)

    def test_stats_counters(self, tok):
        c = _make_collator(tok, 4096)
        c([{"messages": make_tiny_messages(1), "label": 1.0}])
        c([{"messages": make_tiny_messages(2), "label": 0.0}])
        assert c.stats["n_batches"] == 2 and c.stats["n_samples"] == 2
        assert c.stats["n_truncated"] == 0


# ---------------------------------------------------------------------------
# 截断顺序（§7.2）
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_step_drop_keeps_context_instruction_and_judged_step(self, tok):
        """超限 → 从早到晚整步删除较早步：issue/末轮指令保留、被判定步保留、
        较早步消失、插入 marker（tiny tokenizer 解码不可读，按结构断言）。

        长度校准（tiny tokenizer 实测）：floor（context+marker+instruction）=64、
        单步（含 600 词 output）≈647 → max_length=900 触发"删前 3 步保被判定步"，
        不触发 tool 截断。
        """
        c = _make_collator(tok, 900)
        big = make_long_output(600)
        msgs = make_tiny_messages(4, output=big)
        out = c.truncate_messages(msgs)
        assert c._render_len(out) <= c.max_length
        assert out[0]["role"] == "system" and out[1]["role"] == "user"
        assert "issue: fix the bug please" in out[1]["content"]   # issue 不截断
        assert out[-1]["role"] == "user" \
            and "Judge and respond" in out[-1]["content"]          # 末轮指令不丢
        assert any(m.get("content") == TRUNCATION_MARKER for m in out)  # marker 插入
        assistants = [m for m in out if m["role"] == "assistant"]
        assert len(assistants) == 1                                # 只剩被判定步
        assert assistants[0]["reasoning_content"] == "step 4 reasoning"
        assert c.stats["steps_dropped"] >= 1

    def test_marker_not_inserted_when_no_drop(self, tok):
        c = _make_collator(tok, 4096)
        out = c.truncate_messages(make_tiny_messages(1))
        assert not any(m.get("content") == TRUNCATION_MARKER for m in out)

    def test_tool_content_tail_truncation(self, tok):
        """整步删完仍超限 → tool content 尾部截断（保 returncode + 前 N token）。

        校准：n=1 巨 output 样本 669 tokens；tool 截断（保 512 token）后 ≈600 →
        max_length=650 落在 (600, 669) 区间，恰好只触发 tool 截断。
        """
        c = _make_collator(tok, 650)
        big = make_long_output(600)
        msgs = make_tiny_messages(1, output=big)  # 单步：无步可删 → 直接 tool 截断
        out = c.truncate_messages(msgs)
        assert c._render_len(out) <= c.max_length
        assert "issue: fix the bug please" in out[1]["content"]    # issue 保留
        assert "Judge and respond" in out[-1]["content"]           # 指令保留
        tool = next(m for m in out if m["role"] == "tool")
        content = json.loads(tool["content"])
        assert content["returncode"] == 0                          # returncode 保留
        assert content["output"].endswith("…[truncated]")          # 尾部截断标记
        assert len(content["output"]) < len(big)                   # output 被裁剪

    def test_budget_floor_error(self, tok):
        """issue+instruction+marker 超预算 → 报错（禁止截断 issue/指令，§7.2）。"""
        c = _make_collator(tok, 30)
        with pytest.raises(RuntimeError, match="预算不足"):
            c.truncate_messages(make_tiny_messages(2))

    def test_still_over_after_tool_truncation_errors(self, tok):
        """tool 截断后仍超限 → 报错而非静默丢上下文。

        校准：floor=64 ≤ 300（过预算下限）；单步巨 output 669 > 300 → tool 截断
        （保 512 token）后 ≈64+23+512=599 > 300 → "仍超限" 报错。
        """
        c = _make_collator(tok, 300)
        msgs = make_tiny_messages(1, output=make_long_output(600))
        with pytest.raises(RuntimeError, match="仍超限"):
            c.truncate_messages(msgs)

    def test_judged_step_never_dropped(self, tok):
        """单步样本超限：步组不删（被判定步），走 tool 截断。"""
        c = _make_collator(tok, 650)
        msgs = make_tiny_messages(1, output=make_long_output(600))
        out = c.truncate_messages(msgs)
        assert any(m.get("reasoning_content") == "step 1 reasoning" for m in out)
        assert c.stats["steps_dropped"] == 0


# ---------------------------------------------------------------------------
# §7.2 边界样本：被判定步本身超预算 → 预检/跳过（2026-09-15 真机训练崩过）
# ---------------------------------------------------------------------------

def _unfittable_messages(n_words: int = 800) -> list[dict]:
    """末步 assistant 内容巨大且 tool 截断动不了它 → 任何 max_length 都放不下。"""
    msgs = make_tiny_messages(1)
    next(m for m in msgs if m["role"] == "assistant")["content"] = make_long_output(n_words)
    return msgs


class TestOversizeSkip:
    def test_fits_flags_unfittable_without_touching_stats(self, tok):
        c = _make_collator(tok, 700)
        before = dict(c.stats)
        assert c.fits(_unfittable_messages()) is False
        assert c.stats == before                      # 预检不污染截断统计
        assert c.fits(make_tiny_messages(1)) is True
        assert c.stats == before

    def test_oversize_indices_only_scans_candidates_and_caches(self, tok, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from prm.build_dataset import SCHEMA, messages_to_arrow
        from prm.data import PrmParquetDataset, oversize_indices

        def row(sid: str, msgs: list[dict], n_tok: int) -> dict:
            return {
                "sample_id": sid, "instance_id": "i", "repo": "r", "split": "train",
                "node_key": "n", "step_index": 1, "step_count": 1,
                "messages": messages_to_arrow(msgs), "label": 1.0, "label_binary": 1,
                "label_soft": 1.0, "label_source": "node_mc", "mc_score": 1.0,
                "n_rollouts": 5, "visits": 1, "in_pool": 1, "rendered_tokens": n_tok,
            }

        rows = [row("ok1", make_tiny_messages(1), 10),          # ≤ max_length → 快路径
                row("bad1", _unfittable_messages(), 99999),     # 候选 → 真扫 → 命中
                row("ok2", make_tiny_messages(2), 10)]
        path = tmp_path / "d.parquet"
        pq.write_table(pa.table({f.name: pa.array([r[f.name] for r in rows], type=f.type)
                                 for f in SCHEMA}, schema=SCHEMA), path)
        ds = PrmParquetDataset(str(path))
        cache = tmp_path / "oversize_skip_train.json"

        c = _make_collator(tok, 700)
        assert oversize_indices(ds, c, cache_path=str(cache)) == [1]
        assert c.stats["skipped_oversize"] == 1
        assert json.loads(cache.read_text(encoding="utf-8"))["sample_ids"] == ["bad1"]

        c2 = _make_collator(tok, 700)                  # 第二次：指纹命中，直接复用
        assert oversize_indices(ds, c2, cache_path=str(cache)) == [1]
        assert c2.stats["skipped_oversize"] == 1

    def test_collate_or_skip_counts_instead_of_raising(self, tok):
        from prm.data import collate_or_skip

        c = _make_collator(tok, 700)
        assert collate_or_skip(c, [{"messages": _unfittable_messages(), "label": 1.0,
                                    "sample_id": "bad1"}]) is None
        assert c.stats["skipped_oversize"] == 1
        ok = collate_or_skip(c, [{"messages": make_tiny_messages(1), "label": 1.0,
                                  "sample_id": "ok1"}])
        assert ok is not None and ok["input_ids"].shape[0] == 1

    def test_filter_oversize_items_keeps_alignment(self, tok):
        from prm.data import filter_oversize_items

        c = _make_collator(tok, 700)
        items = [{"messages": make_tiny_messages(1), "label_binary": 1, "sample_id": "a",
                  "rendered_tokens": 10},
                 {"messages": _unfittable_messages(), "label_binary": 0, "sample_id": "b",
                  "rendered_tokens": 99999},
                 {"messages": make_tiny_messages(2), "label_binary": 1, "sample_id": "c",
                  "rendered_tokens": 10}]
        kept, dropped = filter_oversize_items(c, items)
        assert [it["sample_id"] for it in kept] == ["a", "c"]   # 顺序/标签对齐保持
        assert dropped == ["b"]
        assert c.stats["skipped_oversize"] == 1


# ---------------------------------------------------------------------------
# PrmParquetDataset（parquet 往返）
# ---------------------------------------------------------------------------

class TestParquetDataset:
    def test_roundtrip_arguments_and_label(self, tok, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from prm.build_dataset import SCHEMA, messages_to_arrow
        msgs = make_tiny_messages(2)
        row = {
            "sample_id": "s1", "instance_id": "i1", "repo": "r", "split": "train",
            "node_key": "n", "step_index": 1, "step_count": 1,
            "messages": messages_to_arrow(msgs),
            "label": 0.96, "label_binary": 1, "label_soft": 0.8,
            "label_source": "node_mc", "mc_score": 0.8, "n_rollouts": 5,
            "visits": 1, "in_pool": 1, "rendered_tokens": 100,
        }
        path = tmp_path / "t.parquet"
        pq.write_table(pa.table({f.name: pa.array([row[f.name]], type=f.type)
                                 for f in SCHEMA}, schema=SCHEMA), path)
        ds = PrmParquetDataset(str(path))
        item = ds[0]
        assert item["label"] == pytest.approx(0.96)
        assert item["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
        assert item["label_source"] == "node_mc"
        assert len(ds) == 1
