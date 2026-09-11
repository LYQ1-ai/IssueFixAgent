# SPDX-License-Identifier: BSD-3-Clause

"""训练期数据处理（docs/prm_training_plan.md §7.2 / §10 test_prm_data）。

- :class:`PrmParquetDataset`：构建产物 parquet → ``{messages, label, ...}`` 样本
  （``arguments`` JSON 字符串就地还原为 dict——chat template 渲染要求 mapping）；
- :class:`VerdictCollator`：**训练/评估期动态截断** + 右 padding 张量化。
  截断顺序（§7.2）：

  1. 保 system+user 上下文（预算不足报错）；
  2. 保末轮指令（禁止丢弃）；
  3. 从早到晚**整步删除**轨迹中较早的步（被判定步 = 最后一步，永不删除）；
  4. 插入 marker ``[earlier steps omitted to fit the PRM context window]``；
  5. 仍超限：单条 tool content 尾部截断（保 returncode + 前 512 token）。

  截断**不在构建期执行**（parquet 存全量 messages），调整 max_length 无需重建数据。
  截断统计在 ``collator.stats`` 累计（train_prm 写入 trainer_state）。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pyarrow.parquet as pq

from prm.preprocess import CHAT_TEMPLATE_KWARGS, TrajectoryPreprocessor
from prm.prompts import TRUNCATION_MARKER

logger = logging.getLogger("prm.data")

# §7.2：tool content 尾部截断保留的 observation token 数
TOOL_KEEP_TOKENS = 512


def load_messages(raw_messages: list[dict]) -> list[dict]:
    """parquet 反序列化 → 可渲染 messages（幂等 canonicalization）。

    ``tool_calls[].function.arguments`` JSON 字符串 → dict；其余字段原样。
    """
    out: list[dict] = []
    for m in raw_messages:
        msg = dict(m)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            fixed = []
            for tc in tool_calls:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (TypeError, ValueError):
                        pass
                fn["arguments"] = args
                tc["function"] = fn
                fixed.append(tc)
            msg["tool_calls"] = fixed
        out.append(msg)
    return out


class PrmParquetDataset:
    """构建产物 parquet 的样本数据集（torch ``Dataset`` 协议，无 torch 依赖）。"""

    def __init__(self, path: str):
        self.path = path
        self.rows = pq.read_table(path).to_pylist()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "sample_id": row["sample_id"],
            "instance_id": row["instance_id"],
            "split": row["split"],
            "label": float(row["label"]),
            "label_binary": int(row["label_binary"]),
            "label_source": row["label_source"],
            "mc_score": row["mc_score"],
            "rendered_tokens": row["rendered_tokens"],
            "messages": load_messages(row["messages"]),
        }

    def iter_rows(self):
        for i in range(len(self)):
            yield self[i]


class VerdictCollator:
    """批量 collator：逐样本截断到 max_length → chat template tokenize → 右 padding。

    Args:
        tokenizer_path: HF tokenizer 目录（PRM 基座）。
        max_length: 单样本最大 token 数（§6 由 length_report 确定，写入 config）。
        tool_keep_tokens: tool content 尾部截断保留的 token 数（§7.2：512）。
    """

    def __init__(self, tokenizer_path: Optional[str] = None, max_length: int = 16384,
                 tool_keep_tokens: int = TOOL_KEEP_TOKENS, *, tokenizer=None):
        """
        Args:
            tokenizer_path: HF tokenizer 目录（PRM 基座）；与 ``tokenizer`` 二选一。
            max_length: 单样本最大 token 数（§6 由 length_report 确定，写入 config）。
            tool_keep_tokens: tool content 尾部截断保留的 token 数（§7.2：512）。
            tokenizer: 已加载的 HF tokenizer（测试/复用场景直接注入）。
        """
        if tokenizer is not None:
            self.pre = None
            self.tokenizer = tokenizer
        else:
            self.pre = TrajectoryPreprocessor(tokenizer_path)
            if self.pre.tokenizer is None:
                raise RuntimeError("VerdictCollator 需要 transformers tokenizer（CodeAgentRL-PRM 环境）")
            self.tokenizer = self.pre.tokenizer
        self.max_length = int(max_length)
        self.tool_keep_tokens = int(tool_keep_tokens)
        # 截断统计（§7.2：写入 trainer_state）
        self.stats = {"n_batches": 0, "n_samples": 0, "n_truncated": 0,
                      "steps_dropped": 0, "tools_truncated": 0, "errors": 0}

    # ------------------------------------------------------------------
    # 截断（§7.2）
    # ------------------------------------------------------------------

    def truncate_messages(self, messages: list[dict]) -> list[dict]:
        """把全 prompt 截断到 max_length 内（返回新 list，不改输入）。

        结构约定（§4.1）：``[system, user(context), *trajectory, user(instruction)]``。
        """
        if self._render_len(messages) <= self.max_length:
            return messages
        context, trajectory, instruction = self._split(messages)
        # 1) 预算下限检查：上下文 + 末轮指令 + marker 必须可容纳（禁止截断 issue/指令）
        floor = context + [{"role": "user", "content": TRUNCATION_MARKER}, instruction]
        floor_len = self._render_len(floor)
        if floor_len > self.max_length:
            raise RuntimeError(
                f"PRM 上下文预算不足：system+issue+instruction+marker 共 {floor_len} tokens "
                f"> max_length={self.max_length}（禁止截断 issue/指令，§7.2）")

        # 2) 从早到晚整步删除较早步（被判定步 = 最后一步，永不删除）
        #    轨迹按"步"分组：assistant 起头，到下一条 assistant 前（§3 步定义）
        step_groups = self._group_steps(trajectory)
        n_drop = self._min_drop(step_groups, context, instruction)
        kept = [m for g in step_groups[n_drop:] for m in g]
        dropped = sum(len(g) for g in step_groups[:n_drop])
        out = context
        if n_drop:
            out = out + [{"role": "user", "content": TRUNCATION_MARKER}]
        out = out + kept + [instruction]
        self.stats["steps_dropped"] += dropped

        if self._render_len(out) <= self.max_length:
            return out

        # 3) 仍超限：单条 tool content 尾部截断（保 returncode + 前 N token）
        out = self._truncate_tool_contents(out)
        self.stats["tools_truncated"] += 1
        final_len = self._render_len(out)
        if final_len > self.max_length:
            raise RuntimeError(
                f"截断后仍超限（{final_len} > {self.max_length}）："
                f"issue/末步过长，检查样本（§7.2 禁止截断 issue/指令）")
        return out

    def _split(self, messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """→ (context=[system,user], trajectory=中间段, instruction=末条 user)。"""
        if len(messages) < 4 or messages[-1].get("role") != "user":
            raise ValueError("prompt 结构异常：应为 [system, user, *trajectory, user(instruction)]")
        context = messages[:2]
        instruction = messages[-1]
        return context, list(messages[2:-1]), instruction

    @staticmethod
    def _group_steps(trajectory: list[dict]) -> list[list[dict]]:
        """轨迹消息 → 步组（assistant 起头 + 其 tail）；首条非 assistant 归入前置组。"""
        groups: list[list[dict]] = []
        for m in trajectory:
            if m.get("role") == "assistant" or not groups:
                groups.append([m])
            else:
                groups[-1].append(m)
        return groups

    def _min_drop(self, step_groups: list[list[dict]], context: list[dict],
                  instruction: list | dict) -> int:
        """最小整步删除数（二分：删除数单调 → token 数单调不增）。

        最后一步组（被判定步）不可删，故上界 = len(groups) - 1。
        """
        n = len(step_groups)
        lo, hi, best = 0, max(0, n - 1), None

        def fits(k: int) -> bool:
            msgs = list(context)
            if k:
                msgs.append({"role": "user", "content": TRUNCATION_MARKER})
            msgs.extend(m for g in step_groups[k:] for m in g)
            msgs.append(instruction)
            return self._render_len(msgs) <= self.max_length

        if not fits(hi):  # 只保留被判定步仍超限 → 交给 tool content 截断
            return hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best if best is not None else hi

    def _truncate_tool_contents(self, messages: list[dict]) -> list[dict]:
        """所有 tool 消息的 content 尾部截断：保 returncode + 前 tool_keep_tokens token。"""
        out: list[dict] = []
        for m in messages:
            if m.get("role") == "tool":
                m = dict(m)
                m["content"] = self._truncate_tool_content(str(m.get("content") or ""))
            out.append(m)
        return out

    def _truncate_tool_content(self, content: str) -> str:
        parsed: Optional[dict] = None
        try:
            d = json.loads(content)
            if isinstance(d, dict):
                parsed = d
        except (TypeError, ValueError):
            pass
        keep = self.tool_keep_tokens
        if parsed is not None:
            output = str(parsed.get("output", ""))
            kept = self._clip_tokens(output, keep)
            return json.dumps({"returncode": parsed.get("returncode"),
                               "output": kept + "…[truncated]"}, ensure_ascii=False)
        return self._clip_tokens(content, keep) + "…[truncated]"

    def _clip_tokens(self, text: str, n: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # 渲染 / 张量化
    # ------------------------------------------------------------------

    def _render_len(self, messages: list[dict]) -> int:
        return len(self._encode(messages))

    def _encode(self, messages: list[dict]) -> list[int]:
        """chat template → token id list（兼容 transformers 新旧返回类型：5.x 返回
        BatchEncoding，旧版返回 list[int]）。"""
        out = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                 **CHAT_TEMPLATE_KWARGS)
        if hasattr(out, "input_ids"):
            out = out.input_ids
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    def __call__(self, batch: list[dict]) -> dict:
        """``[{"messages": [...], "label": float}, ...]`` → 训练张量批（右 padding）。"""
        self.stats["n_batches"] += 1
        self.stats["n_samples"] += len(batch)
        seqs: list[list[int]] = []
        labels: list[float] = []
        for item in batch:
            messages = item["messages"]
            try:
                truncated = self.truncate_messages(messages)
            except RuntimeError:
                self.stats["errors"] += 1
                raise
            if len(truncated) != len(messages):
                self.stats["n_truncated"] += 1
            seqs.append(self._encode(truncated))
            labels.append(float(item["label"]))

        max_len = max(len(s) for s in seqs)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids, attention_mask = [], []
        for s in seqs:
            pad = max_len - len(s)
            # padding 全局右侧（§7.1-3）：attention-mask 取 last non-pad 语义依赖
            input_ids.append(s + [pad_id] * pad)
            attention_mask.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def tensors_from(batch: dict, device: Optional[str] = None):
    """collator 输出 → torch 张量（惰性导入 torch；list[int] 便于离线测试）。"""
    import torch  # 惰性（CodeAgentRL 环境无 torch）

    def _t(x, dtype):
        t = torch.tensor(x, dtype=dtype)
        return t.to(device) if device else t

    return {
        "input_ids": _t(batch["input_ids"], torch.long),
        "attention_mask": _t(batch["attention_mask"], torch.long),
        "labels": _t(batch["labels"], torch.float),
    }


__all__ = [
    "PrmParquetDataset",
    "VerdictCollator",
    "load_messages",
    "tensors_from",
    "TOOL_KEEP_TOKENS",
]
