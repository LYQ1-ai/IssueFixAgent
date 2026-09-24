# SPDX-License-Identifier: BSD-3-Clause

"""训练期数据处理（docs/prm_training_plan.md §7.2 / §10 test_prm_data）。

- :class:`PrmParquetDataset`：构建产物 parquet → ``{messages, label, ...}`` 样本
  （``arguments`` JSON 字符串就地还原为 dict——chat template 渲染要求 mapping）；
- :class:`VerdictCollator`：**训练/评估期动态截断** + 张量化（**强制单样本批**，
  不产生 padding；右 padding 仅为 API 兼容保留）。
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
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq

from prm.preprocess import CHAT_TEMPLATE_KWARGS, TrajectoryPreprocessor
from prm.prompts import TRUNCATION_MARKER, template_hash

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
            "step_index": int(row["step_index"]),
            "step_count": int(row["step_count"]),
            "messages": load_messages(row["messages"]),
        }

    def iter_rows(self):
        for i in range(len(self)):
            yield self[i]


class VerdictCollator:
    """collator：逐样本截断到 max_length → chat template tokenize → 张量化。

    **只接受单样本批**（``len(batch) == 1``）：Qwen3.5 混合架构对 pad 前缀敏感，
    多样本批的 padding 会让打分失真（见 :meth:`__call__` 与计划书 §14.2）。

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
                      "steps_dropped": 0, "tools_truncated": 0, "errors": 0,
                      # §7.2 边界样本：被判定步本身超预算 → 该样本在给定 max_length
                      # 下不可用，只能跳过（跳过的计入此处，不再抛错杀训练）
                      "skipped_oversize": 0}

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

    def fits(self, messages: list[dict]) -> bool:
        """该样本能否截断进 ``max_length``（§7.2 边界样本预检）。

        不改变 ``stats``（预扫不该污染截断计数）。抽出来是为了**在开训前**把
        "被判定步本身就超预算"的样本挑掉——这类样本在 collator 里抛错，而 collator
        跑在 DataLoader worker 内，异常会直接杀掉整个训练（2026-09-15 实测：
        19,862 条里仅 1 条 → step 1146 崩，白跑 10 h）。
        """
        snapshot = dict(self.stats)
        try:
            self.truncate_messages(messages)
            return True
        except RuntimeError:
            return False
        finally:
            self.stats.clear()
            self.stats.update(snapshot)

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
        """``[{"messages": [...], "label": float}, ...]`` → 训练张量（**强制单样本批**）。

        直接产出 torch 张量（HF Trainer 的 data_collator 约定）；离线无 torch
        环境的单测经 ``tensors_from`` / 列表访问另行处理。

        **为什么强制 len(batch)==1**：Qwen3.5 是混合架构（`causal_conv1d` +
        gated delta rule 线性注意力），其顺序递推**对 pad 前缀敏感**——真机实测
        （2026-09-11，`docs/prm_training_plan.md` §14.2）左 padding 会让 σ(z)
        偏移最多 +0.70 且随 pad 数剧烈跳变；右 padding 也有 0.02 量级偏差。
        单样本批不产生任何 padding（``pad = max_len - len(s) = 0``），与逐条
        推理完全一致，故唯一安全的批大小是 1。
        """
        import torch  # 惰性（CodeAgentRL 环境无 torch）

        if len(batch) != 1:
            raise ValueError(
                f"VerdictCollator 只接受单样本批（收到 {len(batch)} 条）——Qwen3.5 混合"
                "架构对 pad 前缀敏感（左 padding 实测 Δp 最多 +0.70），多样本批会引入 "
                "padding 使打分失真。请把 batch_size 设为 1；依据见 "
                "docs/prm_training_plan.md §14.2。"
            )

        self.stats["n_batches"] += 1
        self.stats["n_samples"] += len(batch)
        seqs: list[list[int]] = []
        labels: list[float] = []
        trunc_flags: list[bool] = []
        for item in batch:
            messages = item["messages"]
            try:
                truncated = self.truncate_messages(messages)
            except RuntimeError:
                self.stats["errors"] += 1
                raise
            # 逐样本截断标记（§9.2「truncated 与否」分桶用；总量统计在 stats）
            trunc_flags.append(truncated is not messages and truncated != messages)
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
            # 右 padding（与计划书 §7.1-3 一致）。bs=1 时 pad 恒为 0，此分支仅为
            # API 兼容保留；万一将来放开批大小，必须同时解决混合层的 pad 敏感问题
            # （per-row gather 只保证「取对位置」，不能消除 pad 前缀对递推的污染）。
            input_ids.append(s + [pad_id] * pad)
            attention_mask.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "truncated": torch.tensor(trunc_flags, dtype=torch.bool),
        }


def tensors_from(batch: dict, device: Optional[str] = None):
    """collator 输出 → torch 张量（惰性导入 torch；兼容已是张量的输入）。"""
    import torch  # 惰性（CodeAgentRL 环境无 torch）

    def _t(x, dtype):
        t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
        t = t.to(dtype)
        return t.to(device) if device else t

    out = {
        "input_ids": _t(batch["input_ids"], torch.long),
        "attention_mask": _t(batch["attention_mask"], torch.long),
        "labels": _t(batch["labels"], torch.float),
    }
    if "truncated" in batch:            # 逐样本截断标记（§9.2 分桶用）
        out["truncated"] = _t(batch["truncated"], torch.bool)
    return out


_OVERSIZE_CACHE_VERSION = 1


def oversize_fingerprint(dataset, collator: VerdictCollator) -> dict:
    """缓存指纹：parquet 身份 + 渲染/截断口径（任一变化则缓存作废）。"""
    st = Path(dataset.path).stat()
    return {"version": _OVERSIZE_CACHE_VERSION,
            "max_length": int(collator.max_length),
            "tool_keep_tokens": int(collator.tool_keep_tokens),
            "template_hash": template_hash(),
            "parquet_size": int(st.st_size),
            "parquet_mtime": int(st.st_mtime)}


def oversize_indices(dataset, collator: VerdictCollator,
                     cache_path: Optional[str] = None) -> list[int]:
    """扫出 **截断后仍放不下** 的样本下标（§7.2 边界样本），供训练/评估跳过。

    - 只检查 ``rendered_tokens > max_length`` 的候选（构建期长度与训练期同口径，
      其余必然能放下），所以对 2 万条训练集只有几百条需要真正走一遍策略；
    - ``cache_path`` 命中指纹时直接复用：全量扫要几分钟，每次开训重扫不划算；
    - 返回下标（升序），并把条数累加进 ``collator.stats["skipped_oversize"]``。
    """
    fp = oversize_fingerprint(dataset, collator)
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            try:
                cached = json.loads(cp.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fp:
                    idx = [int(i) for i in cached.get("indices", [])]
                    logger.info("oversize 缓存命中 %s：跳过 %d 条（max_length=%d）",
                                cp.name, len(idx), collator.max_length)
                    collator.stats["skipped_oversize"] += len(idx)
                    return idx
            except (OSError, ValueError, KeyError) as e:
                logger.warning("oversize 缓存不可用（%s）→ 重扫", e)

    out: list[int] = []
    n_cand = 0
    for i in range(len(dataset)):
        row = dataset[i]
        n_tok = int(row.get("rendered_tokens") or 0)
        if 0 < n_tok <= collator.max_length:
            continue
        n_cand += 1
        if not collator.fits(row["messages"]):
            out.append(i)
    if out:
        logger.warning("oversize 扫描：候选 %d 条，其中 **%d 条**无法截断进 max_length=%d → "
                       "跳过（该样本被判定步本身超预算，§7.2）：%s",
                       n_cand, len(out), collator.max_length,
                       [dataset[i]["sample_id"] for i in out[:5]])
    else:
        logger.info("oversize 扫描：候选 %d 条，全部可截断 ✅", n_cand)
    collator.stats["skipped_oversize"] += len(out)
    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(
            {"fingerprint": fp, "indices": out,
             "sample_ids": [dataset[i]["sample_id"] for i in out]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def filter_oversize_items(collator: VerdictCollator,
                          items: list[dict]) -> tuple[list[dict], list[str]]:
    """从**样本列表**里剔除 §7.2 预算不足的样本（保持原顺序）。

    用于抽样评估（``probe``）：那里分数数组与标签按下标对齐，不能"中途跳过"，
    只能先把不可用样本摘掉。返回 ``(保留的 items, 被剔除的 sample_id 列表)``。
    """
    kept, dropped = [], []
    for it in items:
        n_tok = int(it.get("rendered_tokens") or 0)
        if 0 < n_tok <= collator.max_length or collator.fits(it["messages"]):
            kept.append(it)
        else:
            dropped.append(it.get("sample_id", "?"))
    if dropped:
        logger.warning("剔除 %d 条无法截断的样本（§7.2 预算不足）：%s",
                       len(dropped), dropped[:5])
        collator.stats["skipped_oversize"] += len(dropped)
    return kept, dropped


def collate_or_skip(collator: VerdictCollator, items: list[dict]) -> Optional[dict]:
    """``collator(items)``；预算不足（§7.2 边界样本）→ 计数并返回 ``None``。

    评估路径（``eval_prm`` 的逐条前向 / best-of-k 聚合、``probe`` 抽样）用它包一层：
    遇到"被判定步本身就超预算"的样本就跳过，而不是中断整轮评估。
    """
    try:
        return collator(items)
    except RuntimeError as e:
        collator.stats["skipped_oversize"] += 1
        sid = items[0].get("sample_id") if items else "?"
        logger.warning("跳过无法截断的样本 %s：%s", sid, e)
        return None


__all__ = [
    "PrmParquetDataset",
    "VerdictCollator",
    "collate_or_skip",
    "filter_oversize_items",
    "load_messages",
    "oversize_indices",
    "tensors_from",
    "TOOL_KEEP_TOKENS",
]
