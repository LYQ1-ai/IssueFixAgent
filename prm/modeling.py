# SPDX-License-Identifier: BSD-3-Clause

"""verdict 打分模型（docs/prm_training_plan.md §7.1）。

**verdict 形式 ≡ 方向冻结为预训练语义的线性头**：

.. math:: z = logits[b, t, id_c] - logits[b, t, id_i]
.. math:: p = σ(z) = exp(l_c) / (exp(l_c) + exp(l_i))

- ``t`` = 序列**最后一个非 padding token**（右 padding；渲染以
  ``add_generation_prompt=True, enable_thinking=False`` 结尾——模板渲染完整空
  think 块，训练**不需要补 think 闭合标签**（F3），打分位置即该最后 token）；
- 无新增层 / 不 resize embedding：方向向量 ``(w_c − w_i)·h + (b_c − b_i)`` 直接
  取自 lm_head 两行；
- **verdict 两行权重冻结**（F1 ``tie_word_embeddings=True``，embedding 与
  lm_head 共享矩阵——冻结防止输出训练污染这两个 token 的输入语义）。PyTorch
  不支持单 Parameter 的逐行 requires_grad，因此用**梯度 hook 清零对应行**实现
  （LoRA 训练下 embedding 本就冻结，hook 是全量微调场景的保险）；
- 禁用 ``AutoModelForSequenceClassification``（last-token/pad 语义不可控），自写包装类。
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger("prm.model")


def resolve_verdict_ids(tokenizer, pair: Sequence[str] = ("Correct", "Incorrect")) -> tuple[int, int]:
    """verdict token 对 → ``(id_c, id_i)``；逐个 encode 并断言单 token（F2）。

    异常（多 token / 空 token 对）直接抛 ``ValueError``——方向头依赖单 token
    logit，歧义必须显式失败（§12：换 token 对复测由 probe 主导）。
    """
    if len(pair) != 2:
        raise ValueError(f"verdict token 对必须恰为 2 个词，得到 {pair!r}")
    ids: list[int] = []
    for word in pair:
        enc = tokenizer.encode(word, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"verdict 词 {word!r} 编码为 {len(enc)} 个 token（{enc}），要求单 token；"
                f"请更换 token 对（备选 Yes/No、对/错，§12）")
        ids.append(enc[0])
    if ids[0] == ids[1]:
        raise ValueError(f"verdict token 对两词同 id: {ids}")
    return ids[0], ids[1]


def verdict_loss(z: torch.Tensor, labels: torch.Tensor,
                 w_pos: float = 1.0, w_neg: float = 1.0) -> torch.Tensor:
    """加权 soft-BCE（§7.3）：``L = -mean(w_pos·y·logσ(z) + w_neg·(1−y)·log(1−σ(z)))``。

    ``y`` = 混合 soft 标签（collator 传入，∈[0,1]）；全序列**无 LM loss**（不传
    labels 给 LM head）；soft 目标本身携带校准信号，无校准辅助项。
    """
    if z.dim() != 1 or labels.shape != z.shape:
        raise ValueError(f"z/labels 形状不匹配: {tuple(z.shape)} vs {tuple(labels.shape)}")
    loss = -(w_pos * labels * torch.nn.functional.logsigmoid(z)
             + w_neg * (1.0 - labels) * torch.nn.functional.logsigmoid(-z))
    return loss.mean()


class VerdictScorer(nn.Module):
    """backbone（Qwen3.5-4B，可选 PEFT LoRA）→ z = 两 verdict token logit 差。

    Args:
        backbone: HF causal LM（或 vision-text 条件生成模型，仅用其文本侧 logits）。
        verdict_ids: ``(id_c, id_i)``（:func:`resolve_verdict_ids` 产出）。
    """

    def __init__(self, backbone: nn.Module, verdict_ids: tuple[int, int]):
        super().__init__()
        self.backbone = backbone
        self.id_correct, self.id_incorrect = int(verdict_ids[0]), int(verdict_ids[1])
        self.config = getattr(backbone, "config", None)
        self._freeze_verdict_rows()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, verdict_ids: tuple[int, int], *,
                        torch_dtype: torch.dtype = torch.bfloat16,
                        attn_implementation: str = "flash_attention_2",
                        lora: Optional[dict] = None) -> "VerdictScorer":
        """加载基座（可选包 LoRA）→ VerdictScorer。

        - 加载顺序回退：``AutoModelForCausalLM`` → ``AutoModelForImageTextToText``
          （Qwen3.5 官方权重是条件生成架构）；
        - ``attn_implementation`` 不可用（无 flash-attn）时由调用方降级 sdpa；
        - ``lora``: PEFT 配置 dict（r/alpha/dropout/bias/target_modules），为 None
          则不包装（全量微调 / 推理）。
        """
        from transformers import AutoConfig, AutoModelForCausalLM  # 惰性
        try:
            backbone = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        except (ValueError, KeyError) as e:
            logger.warning("AutoModelForCausalLM 加载失败（%s），回退 ImageTextToText 架构", e)
            config = AutoConfig.from_pretrained(model_path)
            from transformers import AutoModelForImageTextToText
            backbone = AutoModelForImageTextToText.from_pretrained(
                model_path, config=config, torch_dtype=torch_dtype,
                attn_implementation=attn_implementation)
        if lora:
            backbone = _apply_lora(backbone, lora)
        return cls(backbone, verdict_ids)

    # ------------------------------------------------------------------
    # verdict 行冻结（F1 tie 共享 → 梯度 hook 清零两行）
    # ------------------------------------------------------------------

    def _freeze_verdict_rows(self) -> None:
        embed = self.backbone.get_input_embeddings()
        rows = [self.id_correct, self.id_incorrect]

        def _zero_verdict_rows(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            for r in rows:
                grad[r].zero_()
            return grad

        # LoRA 场景 embedding 参数 requires_grad=False，hook 不会触发；
        # 全量微调场景 hook 保证 verdict 两行梯度恒为 0（等效 requires_grad=False）。
        handle = embed.weight.register_hook(_zero_verdict_rows)
        self._verdict_hook = handle
        self.verdict_rows = tuple(rows)
        # 共享矩阵（tie）时 lm_head.weight is embed.weight；记录断言信息
        lm_head = getattr(self.backbone, "get_output_embeddings", lambda: None)()
        self.tied_output = lm_head is not None and lm_head.weight is embed.weight

    def trainable_parameters(self):
        """仅返回 requires_grad=True 的参数（单参数组，§8 lr 表）。"""
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``(input_ids, attention_mask) -> z (B,)``；打分位置 = 最后一个非 pad token。"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # (B, L, V)
        last_non_pad = attention_mask.sum(dim=1) - 1          # 右 padding（§7.1-3）
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        step_logits = logits[batch_idx, last_non_pad]         # (B, V)
        return step_logits[:, self.id_correct] - step_logits[:, self.id_incorrect]

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """p = σ(z)（评估/probe 用）。"""
        return torch.sigmoid(self(input_ids, attention_mask))

    # ------------------------------------------------------------------
    # 保存 / 加载（LoRA adapter + manifest）
    # ------------------------------------------------------------------

    def save_pretrained(self, out_dir: str) -> None:
        """保存可复现加载的最小产物：LoRA adapter（若有）+ tokenizer 之外的元数据
        由 train_prm 的 manifest 记录。全量微调时保存完整权重。"""
        peft_model = getattr(self.backbone, "peft_config", None)
        if peft_model:
            self.backbone.save_pretrained(out_dir)  # adapter_model.safetensors
        else:
            self.backbone.save_pretrained(out_dir)


def _apply_lora(backbone: nn.Module, lora: dict) -> nn.Module:
    """PEFT LoRA 包装（§8：r=16, alpha=32, dropout=0.05, target=[q,k,v,o,gate,up,down]）。"""
    from peft import LoraConfig, get_peft_model  # 惰性
    peft_cfg = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias=lora.get("bias", "none"),
        target_modules=list(lora.get("target_modules",
                                     ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(backbone, peft_cfg)
    model.print_trainable_parameters()
    return model


__all__ = [
    "VerdictScorer",
    "resolve_verdict_ids",
    "verdict_loss",
]
