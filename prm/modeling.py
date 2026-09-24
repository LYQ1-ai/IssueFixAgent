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

import inspect
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


def _detect_logits_to_keep_kwarg(backbone: nn.Module) -> Optional[str]:
    """探测 backbone 支持的「只算末位 logits」参数名。

    transformers 5.x 用 ``logits_to_keep``、4.x 用 ``num_logits_to_keep``；
    PEFT 包装层（PeftModel/LoraModel）的 ``forward`` 不含该参数，须先解包到
    底层模型。返回 ``None`` 表示不支持（调用方回退全量 logits）。
    """
    mod = backbone
    getter = getattr(mod, "get_base_model", None)   # PeftModel → 底层模型
    if callable(getter):
        try:
            mod = getter()
        except Exception:  # pragma: no cover - 解包失败不应影响打分
            pass
    forward = getattr(mod, "forward", None)
    if forward is None:
        return None
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


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
        # HF Trainer 兼容属性（5.x save 路径读取）
        self._keys_to_ignore_on_save: list[str] = []
        self._logits_to_keep_kw = _detect_logits_to_keep_kwarg(backbone)
        self._freeze_verdict_rows()

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, verdict_ids: tuple[int, int], *,
                        torch_dtype: torch.dtype = torch.bfloat16,
                        attn_implementation: str = "flash_attention_2",
                        lora: Optional[dict] = None,
                        use_kernels: bool = False) -> "VerdictScorer":
        """加载基座（可选包 LoRA）→ VerdictScorer。

        - 加载顺序回退：``AutoModelForCausalLM`` → ``AutoModelForImageTextToText``
          （Qwen3.5 官方权重是条件生成架构）；
        - ``attn_implementation`` 不可用（无 flash-attn）时由调用方降级 sdpa；
        - ``lora``: PEFT 配置 dict（r/alpha/dropout/bias/target_modules），为 None
          则不包装（全量微调 / 推理）；
        - ``use_kernels``: 经 ``kernels`` 库启用 HF Hub 预编译内核
          （``chunk_gated_delta_rule`` ← ``kernels-community/fla``、
          ``causal_conv1d_fn/update`` ← ``kernels-community/mamba-ssm``）。
          transformers 自带的 torch 版只是**可读参考实现**，官方源码注释称
          ``chunk_gated_delta_rule`` 在 H100 上差**一个数量级以上**；本模型的
          linear-attention 层占多数，故这是吞吐主开关（§14.6）。
          要求 ``kernels`` 已安装（``0.16.x``），否则 ``from_pretrained`` 直接
          抛 ``ValueError``；调用方负责回退。
        """
        from transformers import AutoConfig, AutoModelForCausalLM  # 惰性
        load_kwargs = {"torch_dtype": torch_dtype,
                       "attn_implementation": attn_implementation}
        if use_kernels:
            load_kwargs["use_kernels"] = True
        try:
            backbone = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        except (ValueError, KeyError) as e:
            logger.warning("AutoModelForCausalLM 加载失败（%s），回退 ImageTextToText 架构", e)
            config = AutoConfig.from_pretrained(model_path)
            from transformers import AutoModelForImageTextToText
            backbone = AutoModelForImageTextToText.from_pretrained(
                model_path, config=config, **load_kwargs)
        if lora:
            backbone = _apply_lora(backbone, lora)
        scorer = cls(backbone, verdict_ids)
        scorer.use_kernels = bool(getattr(backbone, "use_kernels", False))
        return scorer

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

        # LoRA 场景 embedding 参数 requires_grad=False：此时 PyTorch 直接拒绝
        # register_hook（"cannot register a hook on a tensor that doesn't require
        # gradient"），故必须跳过注册；全量微调场景才注册，保证 verdict 两行
        # 梯度恒为 0（等效 requires_grad=False）。
        if embed.weight.requires_grad:
            self._verdict_hook = embed.weight.register_hook(_zero_verdict_rows)
        else:
            self._verdict_hook = None
        self.verdict_rows = tuple(rows)
        # 共享矩阵（tie）时 lm_head.weight is embed.weight；记录断言信息
        lm_head = getattr(self.backbone, "get_output_embeddings", lambda: None)()
        self.tied_output = lm_head is not None and lm_head.weight is embed.weight

    def set_verdict_ids(self, verdict_ids: tuple[int, int]) -> None:
        """更换 verdict token 对（M4.0 备选对复测），并重注册冻结 hook。"""
        self.id_correct, self.id_incorrect = int(verdict_ids[0]), int(verdict_ids[1])
        if getattr(self, "_verdict_hook", None) is not None:
            self._verdict_hook.remove()
        self._freeze_verdict_rows()

    def trainable_parameters(self):
        """仅返回 requires_grad=True 的参数（单参数组，§8 lr 表）。"""
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # 设备 / 内核
    # ------------------------------------------------------------------

    def to_device_and_kernelize(self, device) -> "VerdictScorer":
        """搬到目标设备；若启用了 Hub 内核，**在设备上重新内核化**。

        为什么必须重新内核化：``from_pretrained(use_kernels=True)`` 的 ``kernelize``
        发生在**加载时的设备**上——不传 ``device_map`` 时模型在 CPU，而 Hub 内核映射
        是按设备类型（``cuda``）匹配的，于是 CPU 上等于没换内核；之后 Trainer/调用方
        ``.to("cuda")`` 只说"搬张量"，**不会**把参考实现换成 cuda 内核。
        不设防的话就会得到"`use_kernels=True` 但一点没变快"的假结论（§14.6）。
        """
        self.to(device)
        if self.use_kernels and str(device).startswith("cuda"):
            self.backbone.set_use_kernels(True)   # kernelize(device=cuda)
        return self

    # ------------------------------------------------------------------
    # HF Trainer 兼容委托（gradient checkpointing / 保存 / 最优权重回载）
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        return self.backbone.gradient_checkpointing_disable()

    def enable_input_require_grads(self, **kwargs):
        return self.backbone.enable_input_require_grads(**kwargs)

    def state_dict(self, *args, **kwargs):
        """委托 backbone（Trainer 保存/回载最优权重的键空间一致）。"""
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.backbone.load_state_dict(*args, **kwargs)

    def save_pretrained(self, out_dir, **kwargs) -> None:
        """保存可复现加载的最小产物。

        - LoRA：``backbone.save_pretrained``（``adapter_model.safetensors`` +
          ``adapter_config.json``，计划 §8），并镜像一份 ``model.safetensors``
          供 Trainer 的 load_best_model_at_end 通用回载路径使用（adapter 很小，
          双写可忽略）；
        - 全量微调：backbone.save_pretrained 直接产出 ``model.safetensors``。
        """
        from pathlib import Path
        import shutil
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(out))
        is_peft = getattr(self.backbone, "peft_config", None) is not None
        adapter = out / "adapter_model.safetensors"
        if is_peft and adapter.exists():
            shutil.copyfile(adapter, out / "model.safetensors")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``(input_ids, attention_mask) -> z (B,)``；打分位置 = 最后一个非 pad token。

        **显存关键**：只计算末位 logits（``logits_to_keep=1``，切片发生在 lm_head
        之前）。若算全量 ``(B, L, V)``，16K × 248,044 词表在 batch 8 下需约 65GB，
        真机实测直接 OOM（2026-09-11 日志：`64353206272 bytes`）。

        **位置安全**：collator 强制单样本批 → 不产生 padding → 「序列最后一个物理
        位置」就是「最后一个非 pad 的真实 token」，位置切片因此精确。若批次含尾随
        pad（右 padding，理论上只有绕过 collator 才可能出现），则回退全量 logits +
        按 attention_mask 逐行取位（取位正确，但 pad 本身会污染混合层，见 §14.2）。

        ``position_ids`` 显式给出是**防御性**写法：Qwen3.5 实测三种写法（默认
        arange / cumsum−1 截断 / 严格递增）在同一样本上只差第 4 位小数，**不是**
        padding 敏感性的来源（早期注释把成因归给 RoPE 偏移，实测已否定）。
        """
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp_(min=0)
        # 尾随 0（右 padding）计数：>0 时末位可能是 pad，快路径位置切片不可信
        trailing_pad = int((attention_mask.flip(1).cumsum(dim=1) == 0).sum(dim=1).max().item())

        if self._logits_to_keep_kw is not None and trailing_pad == 0:
            try:
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, use_cache=False,
                                    **{self._logits_to_keep_kw: 1})
                step_logits = out.logits[:, -1]              # (B, V)
                return step_logits[:, self.id_correct] - step_logits[:, self.id_incorrect]
            except TypeError as e:  # 参数名不被接受（自定义/包装模型）→ 记录并回退
                logger.warning("backbone 不接受 %s（%s）→ 回退全量 logits（显存开销大）",
                               self._logits_to_keep_kw, e)
                self._logits_to_keep_kw = None

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False)
        logits = out.logits  # (B, L, V)
        last = (attention_mask.shape[1] - 1
                - (attention_mask.flip(1).cumsum(dim=1) == 0).sum(dim=1))
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        step_logits = logits[batch_idx, last]                   # (B, V)
        return step_logits[:, self.id_correct] - step_logits[:, self.id_incorrect]

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """p = σ(z)（评估/probe 用）。"""
        return torch.sigmoid(self(input_ids, attention_mask))


def kernelized_modules(model: nn.Module) -> list[str]:
    """列出"不是 torch/transformers/peft 实现"的子模块类型全名。

    用途：**验证 Hub 内核是否真的生效**（`use_kernels=True` 只是个意图，CPU 上加载
    时它会静默变成空操作，见 `VerdictScorer.to_device_and_kernelize`）。
    内核化会把这些层换成从 HF Hub 取来的实现，其类/函数的 ``__module__`` 落在
    ``kernels`` 缓存或仓库命名空间里，于是能被这个过滤器挑出来。
    """
    keep = ("torch.", "transformers.", "peft.", "accelerate.", "triton.", "prm.")
    names = set()
    for m in model.modules():
        mod = type(m).__module__
        if not mod.startswith(keep):
            names.add(f"{mod}.{type(m).__name__}")
    return sorted(names)


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
    "kernelized_modules",
    "resolve_verdict_ids",
    "verdict_loss",
]
