# SPDX-License-Identifier: BSD-3-Clause

"""PRM torch 侧单测共享 fixture（离线 tiny tokenizer / tiny model）。

依赖 torch + transformers（PRM 环境可跑；CodeAgentRL 环境的测试经
``pytest.importorskip`` 自动跳过）。**全部离线**：不下载任何模型/数据——

- tiny tokenizer：``tokenizers`` WordLevel + PreTrainedTokenizerFast + 自定义
  最小 chat template（同构 Qwen 的 im_start/im_end 结构，含空 think 块生成提示）；
- tiny model：transformers LlamaConfig（tie_word_embeddings=True，对齐 F1）
  随机初始化，仅用于梯度/logit 语义验证。
"""

from __future__ import annotations

VOCAB_WORDS = [
    "[UNK]", "[PAD]",
    "Correct", "Incorrect",
    "system", "user", "assistant", "tool",
    "issue", "fix", "bug", "run", "ls", "tests", "output", "error",
    "step", "reasoning", "command", "returncode",
] + [f"word{i}" for i in range(1000)]  # make_long_output 的词（保证编解码可逆）


def build_tiny_tokenizer():
    """最小离线 tokenizer：WordLevel（未知词 → 单 [UNK]，长度∝词数，确定性）。"""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(VOCAB_WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok,
                                   unk_token="[UNK]", pad_token="[PAD]")
    # 最小 chat template：与 Qwen 同构的 im_start/im_end + nothink 生成提示（空 think 块）
    fast.chat_template = (
        "{%- for m in messages %}"
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{%- endif %}"
    )
    return fast


def build_tiny_model(vocab_size: int = 64):
    """随机初始化 tiny causal LM（tie_word_embeddings=True，对齐 F1 共享矩阵语义）。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=512, tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config)


def make_tiny_messages(n_steps: int = 2, output: str = "ok output") -> list[dict]:
    """构造 §4.1 结构的最小 PRM prompt（canonical 形态，arguments 为 dict）。"""
    msgs = [
        {"role": "system", "content": "You are a process reward model for tests."},
        {"role": "user", "content": "issue: fix the bug please"},
    ]
    for i in range(1, n_steps + 1):
        msgs.append({
            "role": "assistant",
            "content": "",
            "reasoning_content": f"step {i} reasoning",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "bash", "arguments": {"command": "ls"}}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f'{{"returncode": 0, "output": "{output}"}}'})
    msgs.append({"role": "user",
                 "content": "Judge and respond Correct or Incorrect."})
    return msgs


def make_long_output(n_words: int = 400) -> str:
    return " ".join(f"word{i}" for i in range(n_words))
