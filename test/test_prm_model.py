# SPDX-License-Identifier: BSD-3-Clause

"""prm/modeling.py 单测（docs/prm_training_plan.md §10 test_prm_model 行）。

覆盖：verdict id 单 token 断言（异常 token 对报错）、verdict 行冻结（梯度清零）、
z = 两 token logit 差、加权 soft-BCE 与手算一致、padded 前后分数一致。
全部离线：tiny tokenizer + tiny 随机 Llama（tie_word_embeddings，F1）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prm_torch_fixtures import (  # noqa: E402
    build_tiny_model,
    build_tiny_tokenizer,
    make_tiny_messages,
)

from prm.modeling import (  # noqa: E402
    VerdictScorer,
    resolve_verdict_ids,
    verdict_loss,
)


# ---------------------------------------------------------------------------
# resolve_verdict_ids
# ---------------------------------------------------------------------------

class _MockTok:
    """encode → 配置映射（多 token 场景验证）。"""

    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping[text])


class TestResolveVerdictIds:
    def test_single_token_ok(self):
        tok = _MockTok({"Correct": [5], "Incorrect": [7]})
        assert resolve_verdict_ids(tok) == (5, 7)

    def test_multi_token_raises(self):
        tok = _MockTok({"Correct": [5], "Incorrect": [7, 8]})
        with pytest.raises(ValueError, match="单 token"):
            resolve_verdict_ids(tok)

    def test_same_id_raises(self):
        tok = _MockTok({"Correct": [5], "Incorrect": [5]})
        with pytest.raises(ValueError, match="同 id"):
            resolve_verdict_ids(tok)

    def test_pair_length_raises(self):
        with pytest.raises(ValueError, match="恰为 2"):
            resolve_verdict_ids(_MockTok({}), ("A",))

    def test_real_tokenizer_f2(self):
        """F2 固化：真实 tokenizer 中 Correct/Incorrect 均单 token（PRM 环境）。"""
        import os
        path = os.environ.get("PRM_TEST_TOKENIZER", "/media/shared_e/models/Qwen3.5-4B")
        if not Path(path).exists():
            pytest.skip(f"tokenizer 不存在: {path}")
        from transformers import AutoTokenizer
        real = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        ids = resolve_verdict_ids(real)
        assert ids[0] != ids[1] and all(isinstance(i, int) for i in ids)


# ---------------------------------------------------------------------------
# VerdictScorer（tiny 模型）
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    torch.manual_seed(0)
    tok = build_tiny_tokenizer()
    ids = resolve_verdict_ids(tok)  # Correct=2, Incorrect=3（WordLevel vocab 顺序）
    model = VerdictScorer(build_tiny_model(vocab_size=64), ids)
    model.eval()
    return model, tok


def _batch(scorer, tok, n_steps: int, label: float = 1.0):
    from prm.data import VerdictCollator, tensors_from
    from prm_torch_fixtures import make_tiny_messages
    coll = VerdictCollator(tokenizer=tok, max_length=2048)
    batch = coll([{"messages": make_tiny_messages(n_steps), "label": label}])
    return tensors_from(batch)


class TestVerdictScorer:
    def test_z_equals_logit_difference(self, scorer):
        """z = logits[last_non_pad, id_c] − logits[last_non_pad, id_i]（§7.1）。"""
        model, tok = scorer
        t = _batch(model, tok, 2)
        z = model(t["input_ids"], t["attention_mask"])
        out = model.backbone(input_ids=t["input_ids"], attention_mask=t["attention_mask"],
                             use_cache=False)
        last = t["attention_mask"].sum(1) - 1
        manual = (out.logits[0, last[0], model.id_correct]
                  - out.logits[0, last[0], model.id_incorrect])
        # 快路径只对末位做 GEMM，累加顺序与全量 GEMM 不同 → 允许浮点误差
        assert torch.allclose(z, torch.stack([manual]), atol=1e-5)

    def test_trailing_pad_fallback_picks_last_real_position(self, scorer):
        """绕过 collator 手造带尾随 pad 的批 → 回退全量 logits 路径且取位正确。

        生产路径已由 collator 强制单样本批（无 padding）；本测试只保证那条
        **兜底分支**（`trailing_pad > 0`）的取位语义没写错——注意它只修正"取哪个
        位置"，并不能消除 pad 本身对 Qwen3.5 混合层的污染（计划书 §14.2）。
        """
        model, tok = scorer
        from prm.data import VerdictCollator
        from prm_torch_fixtures import make_tiny_messages
        coll = VerdictCollator(tokenizer=tok, max_length=2048)
        b1 = coll([{"messages": make_tiny_messages(1), "label": 1.0}])
        z_single = model(b1["input_ids"], b1["attention_mask"])[0]

        n1 = int(b1["attention_mask"].sum())
        pad_id = tok.pad_token_id
        n2 = n1 + 7
        ids = torch.cat([b1["input_ids"][0], torch.full((7,), pad_id, dtype=torch.long)])[None]
        mask = torch.cat([torch.ones(n1, dtype=torch.long),
                          torch.zeros(7, dtype=torch.long)])[None]
        assert model._logits_to_keep_kw == "logits_to_keep"   # 快路径存在但不应被用
        z_pad = model(ids, mask)[0]
        assert torch.allclose(z_pad, z_single, atol=1e-5)

    def test_forward_uses_logits_to_keep(self, scorer):
        """只算末位 logits（全量 (B,L,V) 在 16K×248K 词表下 ≈65GB，必 OOM）。"""
        model, tok = scorer
        calls: dict = {}
        real_forward = model.backbone.forward

        def spy(*args, **kwargs):
            calls.update(kwargs)
            return real_forward(*args, **kwargs)

        model.backbone.forward = spy
        try:
            t = _batch(model, tok, 2)
            z = model(t["input_ids"], t["attention_mask"])
        finally:
            model.backbone.forward = real_forward
        assert calls.get("logits_to_keep") == 1, "快路径未启用（会 OOM）"
        assert "position_ids" in calls, "左 padding 下必须显式给位置编码"
        assert z.shape == (1,)

    def test_fallback_when_logits_to_keep_unsupported(self, scorer):
        """backbone 不支持该参数时回退全量 logits，结果与快路径一致。"""
        model, tok = scorer
        t = _batch(model, tok, 2)
        z_fast = model(t["input_ids"], t["attention_mask"])
        model._logits_to_keep_kw = None          # 强制走回退路径
        z_fallback = model(t["input_ids"], t["attention_mask"])
        assert torch.allclose(z_fast, z_fallback, atol=1e-5)

    def test_detect_logits_to_keep_kwarg(self, scorer):
        from prm.modeling import _detect_logits_to_keep_kwarg
        model, _ = scorer
        assert _detect_logits_to_keep_kwarg(model.backbone) == "logits_to_keep"

        class _Old(torch.nn.Module):
            def forward(self, input_ids, attention_mask, num_logits_to_keep=0):
                return None

        assert _detect_logits_to_keep_kwarg(_Old()) == "num_logits_to_keep"

        class _Bare(torch.nn.Module):
            def forward(self, input_ids):
                return None

        assert _detect_logits_to_keep_kwarg(_Bare()) is None

    def test_predict_proba_sigmoid(self, scorer):
        model, tok = scorer
        t = _batch(model, tok, 1)
        p = model.predict_proba(t["input_ids"], t["attention_mask"])
        z = model(t["input_ids"], t["attention_mask"])
        assert torch.allclose(p, torch.sigmoid(z))
        assert ((p >= 0) & (p <= 1)).all()

    def test_verdict_rows_gradient_frozen(self, scorer):
        """verdict 两行权重梯度恒为 0（F1 tie 共享矩阵；hook 清零，§7.1）。"""
        model, tok = scorer
        # 全量微调场景：显式放开全部参数（含 embedding），hook 应冻结 verdict 两行
        for p in model.parameters():
            p.requires_grad_(True)
        t = _batch(model, tok, 1, label=0.9)
        loss = verdict_loss(model(t["input_ids"], t["attention_mask"]), t["labels"])
        loss.backward()
        grad = model.backbone.get_input_embeddings().weight.grad
        assert grad is not None
        assert grad[model.id_correct].abs().sum() == 0
        assert grad[model.id_incorrect].abs().sum() == 0
        assert grad[0].abs().sum() > 0  # 其它行正常收到梯度

    def test_tied_embeddings_detected(self, scorer):
        model, _ = scorer
        assert model.tied_output is True  # F1：tie_word_embeddings 共享矩阵

    def test_trainable_parameters_excludes_frozen(self, scorer):
        model, _ = scorer
        # 默认全量参数可训（train_prm 用 LoRA 时仅 LoRA 参数 requires_grad）
        n_trainable = sum(1 for p in model.trainable_parameters())
        assert n_trainable > 0


# ---------------------------------------------------------------------------
# verdict_loss（§7.3）
# ---------------------------------------------------------------------------

class TestVerdictLoss:
    def test_matches_manual_weighted_soft_bce(self):
        z = torch.tensor([2.0, -1.0, 0.5])
        y = torch.tensor([1.0, 0.0, 0.7])
        w_pos, w_neg = 1.0, 4.0
        got = verdict_loss(z, y, w_pos, w_neg)
        sig = torch.sigmoid(z)
        manual = -(w_pos * y * torch.log(sig) + w_neg * (1 - y) * torch.log(1 - sig)).mean()
        assert torch.allclose(got, manual, atol=1e-6)

    def test_zero_for_perfect_direction(self):
        # z→+∞ 等价 σ(z)→1：正样本 logσ→0；y=1 → loss→0
        z = torch.tensor([100.0])
        assert verdict_loss(z, torch.tensor([1.0])) < 1e-6

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            verdict_loss(torch.zeros(3), torch.zeros(2))

    def test_soft_label_interpolates(self):
        """同一 z 下，soft 目标的损失介于 y=1 与 y=0 之间（soft 携带校准信号，§7.3）；
        且 soft 方向正确时损失低于方向错误。"""
        z = torch.tensor([1.0])
        l_y1 = verdict_loss(z, torch.tensor([1.0]), 1.0, 1.0)
        l_y0 = verdict_loss(z, torch.tensor([0.0]), 1.0, 1.0)
        l_soft = verdict_loss(z, torch.tensor([0.8]), 1.0, 1.0)
        assert l_y1 < l_soft < l_y0
        assert verdict_loss(z, torch.tensor([0.8])) < verdict_loss(-z, torch.tensor([0.8]))


# ---------------------------------------------------------------------------
# 真实 Qwen3.5-4B + GPU 端到端（CUDA 门控：无 GPU 自动跳过；A800 上 ~1 分钟）
# ---------------------------------------------------------------------------

class TestRealModelGpu:
    """M4.0 前置验证：真实基座在 GPU 上可加载、可前向、打分非退化。

    基座/tokenizer 路径经 ``PRM_TEST_TOKENIZER``（默认 /media/shared_e/models/
    Qwen3.5-4B）；模型文件缺失或无 GPU 时整类 skip，不误报失败。
    """

    REAL_MODEL = os.environ.get("PRM_TEST_TOKENIZER", "/media/shared_e/models/Qwen3.5-4B")

    def _load_scorer_on_gpu(self):
        from transformers import AutoTokenizer
        from prm.modeling import VerdictScorer

        path = self.REAL_MODEL
        if not Path(path, "config.json").exists():
            pytest.skip(f"基座不存在: {path}")
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        ids = resolve_verdict_ids(tok)
        scorer = VerdictScorer.from_pretrained(
            path, ids, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")
        scorer.eval()
        return scorer, tok

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
    def test_real_model_forward_on_gpu(self):
        scorer, tok = self._load_scorer_on_gpu()
        assert next(scorer.parameters()).dtype == torch.bfloat16
        assert next(scorer.parameters()).device.type == "cuda"

        # 真 tokenizer 渲染（F3：nothink 生成提示含空 think 块）→ collator → 前向
        # 单样本批（bs=1 强制；多样本会产生 padding，见计划书 §14.2）
        from prm.data import VerdictCollator
        collator = VerdictCollator(tokenizer=tok, max_length=4096)
        batch = collator([{"messages": make_tiny_messages(2), "label": 1.0}])
        with torch.no_grad():
            z = scorer(batch["input_ids"].to("cuda"),
                       batch["attention_mask"].to("cuda"))
        assert z.shape == (1,)
        assert torch.isfinite(z).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
    def test_real_model_scores_not_degenerate(self):
        """未训练基座缩微方向检查：不同前缀的 z 应有离散度而非常数（M4.0 前置）。"""
        scorer, tok = self._load_scorer_on_gpu()
        from prm.data import VerdictCollator
        collator = VerdictCollator(tokenizer=tok, max_length=4096)
        zs = []
        for n in (1, 2, 3):   # 逐条前向（bs=1，无 padding）
            b = collator([{"messages": make_tiny_messages(n), "label": 1.0}])
            with torch.no_grad():
                zs.append(float(scorer(b["input_ids"].to("cuda"),
                                       b["attention_mask"].to("cuda"))[0]))
        assert len(set(zs)) > 1, f"不同前缀给出相同 z={zs}（打分退化）"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
    def test_real_model_right_padding_gather_is_exact(self):
        """真实架构护栏：手造**右** padding 批 + 逐行取位 ≈ 单条评分。

        实测基准（2026-09-11，CPU 参考实现）：右 padding 在 1/8 pad 时 Δz=0.0000、
        46 pad 时 Δz=0.125；而**左** padding 的 Δz 可达 1.22（Δp 0.26~0.70）。
        阈值 0.26 足以拦住左 padding / 批大小放开导致的回归，同时容忍 bf16 与
        分块内核的数值抖动。tiny Llama 测不出这类问题（无卷积/线性注意力层）。
        """
        scorer, tok = self._load_scorer_on_gpu()
        from prm.data import VerdictCollator
        coll = VerdictCollator(tokenizer=tok, max_length=4096)
        b = coll([{"messages": make_tiny_messages(1), "label": 1.0}])
        with torch.no_grad():
            z_alone = float(scorer(b["input_ids"].to("cuda"),
                                   b["attention_mask"].to("cuda"))[0])
        n = int(b["attention_mask"].sum())
        pad = tok.pad_token_id
        ids = torch.cat([b["input_ids"][0],
                         torch.full((16,), pad, dtype=torch.long)])[None].to("cuda")
        mask = torch.cat([torch.ones(n, dtype=torch.long),
                          torch.zeros(16, dtype=torch.long)])[None].to("cuda")
        with torch.no_grad():
            z_pad = float(scorer(ids, mask)[0])   # trailing_pad>0 → 回退全量+gather
        assert abs(z_pad - z_alone) < 0.26, (
            f"右 padding 取位不精确：z_pad={z_pad} vs z_alone={z_alone}"
            "（若改回左 padding，此处会显著失败——见计划书 §14.2）")
