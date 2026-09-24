# SPDX-License-Identifier: BSD-3-Clause

"""PRM 训练集构建（docs/prm_training_plan.md §5）。

把 MCTS 树库（``outputs/batch500/state.db``，严格只读）的节点派生为步级训练
样本并落盘 ``outputs/prm/{train,dev,test}.parquet`` + manifest + build_report +
length_report。

核心概念（§5.1，先读计划书再读代码）：

- MCTS **节点** = "agent 执行到某步时的完整前缀状态"；``mc`` = 从该前缀续跑
  N 次的成功率（实测状态质量，由 rollouts 重算，§2.1）；
- 一条**训练样本** = 把某个前缀渲染成 PRM 输入，让模型给**前缀最后一步**打分；
  标签 = 该节点的 mc；
- 非 leaf 节点（0<mc<1 或 mc=1）→ 1 条 ``node_mc`` 样本；
- leaf 节点（非 root 且 mc=0）→ 二分定位认为前缀最后一步是首个错误步：末步
  标 0、之前链式回填 1 → L 条 ``leaf_chain`` 样本（第 i 条 = 前 i 步的前缀）；
- root（空前缀）不产样本。

去重键 ``(instance_id, prefix_node_key(prefix), step_index)`` 内容寻址
（``mcts/steps.py::prefix_node_key``，与引擎一致），优先级
``node_mc > leaf_chain`` —— 真实测量的节点样本永远压过链式回填的推定样本
（§5.1 例子：probe① 的 0.96 压过 leaf 回填的 1.0）。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from mcts.steps import prefix_node_key

from prm import labeling, raw
from prm.preprocess import TrajectoryPreprocessor
from prm.env import load_project_env
from prm.prompts import TEMPLATE_VERSION, template_hash

logger = logging.getLogger("prm.build")

SPLITS = ("train", "dev", "test")
SPLIT_ORDER = {s: i for i, s in enumerate(SPLITS)}
LABEL_SOURCE_PRIORITY = {"node_mc": 1, "leaf_chain": 0}  # 去重优先级（§5.1 例子）
FLUSH_ROWS = 1024          # §5.3：每 1024 条 flush writer（= 一个 row group）
MC_MISMATCH_TOL = 1e-6     # §2.1-4：重算值 vs 缓存差容忍

# ---------------------------------------------------------------------------
# parquet schema（§5.4）
# ---------------------------------------------------------------------------
# messages.struct.tool_calls[].function.arguments 以 JSON **字符串**存储（chat
# template 渲染需要 dict；加载端经 canonicalization 重新解析——prm/data.py 的
# 数据装载路径会跑一遍与构建端相同的白名单函数）。
_TOOL_CALL_TYPE = pa.list_(pa.struct([
    ("id", pa.string()),
    ("type", pa.string()),
    ("function", pa.struct([
        ("name", pa.string()),
        ("arguments", pa.string()),   # canonical dict 的 JSON 序列化
    ])),
]))
_MESSAGE_TYPE = pa.list_(pa.struct([
    ("role", pa.string()),
    ("content", pa.string()),
    ("reasoning_content", pa.string()),
    ("tool_calls", _TOOL_CALL_TYPE),
    ("tool_call_id", pa.string()),
]))

SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("instance_id", pa.string()),
    ("repo", pa.string()),
    ("split", pa.string()),
    ("node_key", pa.string()),
    ("step_index", pa.int32()),
    ("step_count", pa.int32()),
    ("messages", _MESSAGE_TYPE),
    ("label", pa.float32()),
    ("label_binary", pa.int8()),
    ("label_soft", pa.float32()),
    ("label_source", pa.string()),
    ("mc_score", pa.float32()),
    ("n_rollouts", pa.int16()),
    ("visits", pa.int16()),
    ("in_pool", pa.int8()),
    ("rendered_tokens", pa.int32()),
])

DEFAULT_CONFIG: dict[str, Any] = {
    "db": {"path": "outputs/batch500/state.db"},
    "splits": {"path": "outputs/mcts/splits.parquet"},
    "output": {"dir": "outputs/prm"},
    "tokenizer": None,
    "build": {
        "min_rollouts": 3,
        "leaf_chain": True,
        "soft_label_weight": 0.2,
        "class_weight_cap": 4.0,
        "workers": 1,
        "overwrite": False,
        "mc_mismatch_tol": MC_MISMATCH_TOL,
        "length_candidates": [8192, 16384, 32768],
    },
}


def deep_merge(base: dict, override: Optional[dict]) -> dict:
    """递归合并配置（override 优先；``None`` 值不覆盖）。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 消息 → parquet struct 序列化
# ---------------------------------------------------------------------------

def messages_to_arrow(messages: list[dict]) -> list[dict]:
    """canonical messages → SCHEMA.messages 对应的嵌套 dict 列表。

    ``function.arguments`` dict → JSON 字符串；缺省字段置 ``None``（struct 稀疏）。
    """
    out: list[dict] = []
    for m in messages:
        tool_calls = None
        if m.get("tool_calls"):
            tool_calls = [{
                "id": tc.get("id"),
                "type": tc.get("type"),
                "function": {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": json.dumps((tc.get("function") or {}).get("arguments"),
                                            ensure_ascii=False, sort_keys=True),
                },
            } for tc in m["tool_calls"]]
        out.append({
            "role": m.get("role"),
            "content": m.get("content"),
            "reasoning_content": m.get("reasoning_content"),
            "tool_calls": tool_calls,
            "tool_call_id": m.get("tool_call_id"),
        })
    return out


# ---------------------------------------------------------------------------
# 节点 → 样本（§5.1 规则；worker 与顺序路径共用）
# ---------------------------------------------------------------------------

_SKIP_KEYS = ("empty_prefix", "no_rollouts", "min_rollouts", "mc_mismatch", "key_mismatch")


def _new_skip_counter() -> dict[str, int]:
    return {k: 0 for k in _SKIP_KEYS}


def node_samples(instance_id: str, node: dict, head_user: str, mc: float,
                 n_rollouts: int, pre: TrajectoryPreprocessor, cfg: dict,
                 skipped: dict[str, int]) -> list[dict]:
    """单个节点 → 样本列表（空列表 = 该节点被跳过，原因计入 ``skipped``）。

    ``node`` 需含 ``node_key / prefix_json / visits / in_pool / mc_score``（缓存，
    仅供 §2.1-4 审计比对）。样本附带私有键 ``_prefix_key`` = 前缀内容寻址键
    （去重键的组成，写盘前剥离）与 ``repo / split``。
    """
    build_cfg = cfg["build"]
    steps = pre.parse_steps(node["prefix_json"])
    n_steps = len(steps)
    if n_steps == 0:
        skipped["empty_prefix"] += 1  # root 或空前缀：没有步可打分（§5.1）
        return []

    # 内容寻址完整性：节点 key 必须等于前缀内容哈希（引擎口径，§5.3 去重键的前提）
    if prefix_node_key(steps) != node["node_key"]:
        skipped["key_mismatch"] += 1
        logger.warning("节点 key 与内容哈希不一致（跳过）: %s/%s", instance_id, node["node_key"])
        return []

    # 重算 MC vs 缓存审计（§2.1-4：差 >1e-6 的节点跳过并列入构建报告）
    cached = node.get("mc_score")
    if cached is not None and abs(cached - mc) > build_cfg.get("mc_mismatch_tol", MC_MISMATCH_TOL):
        skipped["mc_mismatch"] += 1
        return []

    is_leaf = mc == 0.0  # 非 root 已由 n_steps>0 + key 检查保证（root 前缀为空）
    base = {
        "instance_id": instance_id,
        "node_key": node["node_key"],       # 来源节点（审计）
        "step_count": n_steps,
        "mc_score": float(mc),
        "n_rollouts": int(n_rollouts),
        "visits": int(node.get("visits") or 0),
        "in_pool": int(node.get("in_pool") or 0),
    }

    if is_leaf and build_cfg.get("leaf_chain", True):
        # leaf 链式回填（§5.1/§5.2）：L 条样本，第 i 条 = 前 i 步的前缀
        chain = labeling.leaf_chain_labels(n_steps)
        samples = []
        for i, soft_label in enumerate(chain, start=1):
            samples.append({
                **base,
                "_prefix_key": prefix_node_key(steps[:i]),
                "step_index": i,
                "messages": pre.build_messages(head_user, steps[:i]),
                "label": float(soft_label),        # chain 的 binary==soft，混合值不变
                "label_binary": int(soft_label > 0.5),
                "label_soft": float(soft_label),
                "label_source": "leaf_chain",
                "rendered_tokens": 0,  # iter_samples 内统一渲染计长
            })
        return samples

    # 非 leaf（0<mc<1 或 mc=1）→ 1 条真实测量样本；leaf 且未开链式回填 → 1 条负样本
    binary, soft = labeling.node_label(mc)
    label = labeling.mixed_label(binary, soft, build_cfg.get("soft_label_weight", 0.2))
    return [{
        **base,
        "_prefix_key": node["node_key"],  # 整前缀 = 节点本身（内容寻址一致）
        "step_index": n_steps,
        "messages": pre.build_messages(head_user, steps),
        "label": float(label),
        "label_binary": binary,
        "label_soft": float(soft),
        "label_source": "node_mc",
        "rendered_tokens": 0,
    }]


def _render_tokens(pre: TrajectoryPreprocessor, samples: list[dict]) -> None:
    for s in samples:
        s["rendered_tokens"] = pre.rendered_tokens(s["messages"])


def build_instance_samples(payload: dict) -> dict:
    """一个实例的全部节点 → 样本（multiprocessing worker 入口，模块级可 pickle）。

    ``payload["nodes"]`` 的每个节点已附 ``mc / n_rollouts``（主进程从重算表查出，
    避免把整张表 pickle 给每个 worker）。
    """
    skipped = _new_skip_counter()
    if payload["head_user"] is None:
        return {"samples": [], "skipped": skipped, "instance_skipped": True}
    pre = TrajectoryPreprocessor(payload.get("tokenizer"))
    samples: list[dict] = []
    for node in payload["nodes"]:
        if node["mc"] is None:
            skipped["no_rollouts"] += 1  # 无任何 rollout 行：有效证据为 0
            continue
        if node["n_rollouts"] < payload["min_rollouts"]:
            skipped["min_rollouts"] += 1  # §2.1-2：有效证据不足，不进训练集
            continue
        part = node_samples(payload["instance_id"], node, payload["head_user"], node["mc"],
                            node["n_rollouts"], pre, payload["cfg"], skipped)
        for s in part:
            s["repo"] = payload["repo"]
            s["split"] = payload["split"]
        samples.extend(part)
    _render_tokens(pre, samples)
    return {"samples": samples, "skipped": skipped, "instance_skipped": False}


# ---------------------------------------------------------------------------
# 构建器
# ---------------------------------------------------------------------------

class PRMDatasetBuilder:
    """编排：prepare → iter_samples → dedupe → write（§5.3 类规格）。

    Args:
        cfg: 全量配置（DEFAULT_CONFIG 结构；CLI/yaml 合并后传入）。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["output"]["dir"])
        self.samples: list[dict] = []
        self.dedupe_stats: dict = {}
        self.skip_stats: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.class_weights: dict = {}
        self._conn = None
        self._db_path = cfg["db"]["path"]
        self._db_mtime_before: Optional[float] = None
        self._db_fingerprint_before: dict = {}

    # ------------------------------------------------------------------
    # prepare（§5.3）：加载 heads/mc/splits，过滤 done 树，防泄漏断言
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        self._conn = raw.open_db_readonly(self._db_path)
        db_file = Path(self._db_path)
        self._db_mtime_before = db_file.stat().st_mtime
        self._db_fingerprint_before = raw.db_fingerprint(self._conn)
        logger.info("DB 只读打开: %s（行数快照 %s）", self._db_path, self._db_fingerprint_before)

        status = raw.load_tree_status(self._conn)
        done_ids = {iid for iid, st in status.items() if st == "done"}
        self._heads = raw.load_instance_heads(self._conn)
        mc_df = raw.load_mc_table(self._conn)
        self._mc_table: dict[tuple[str, str], tuple[float, int]] = {
            (r.instance_id, r.node_key): (float(r.mc), int(r.n_rollouts))
            for r in mc_df.itertuples(index=False)
        }
        splits_df = raw.load_splits(self.cfg["splits"]["path"])
        self._split_map = {r.instance_id: (r.repo, r.prm_split)
                           for r in splits_df.itertuples(index=False)}
        self._assert_no_leakage(splits_df)

        # done ∩ 有 split 的实例（无 split / 非 done 计入审计）
        self._instance_ids = sorted(done_ids & self._split_map.keys())
        self._audits = {
            "not_done_tree": len(status) - len(done_ids),
            "no_split": len(done_ids) - len(done_ids & self._split_map.keys()),
            "head_missing": 0,  # iter_samples 中按实例累计
        }
        logger.info("prepare: done∩split 实例 %d 个（非 done %d / 无 split %d）",
                    len(self._instance_ids), self._audits["not_done_tree"],
                    self._audits["no_split"])

    @staticmethod
    def _assert_no_leakage(splits_df) -> None:
        """§5.6：train/dev/test 两两 instance 与 repo 交集为空，不通过直接报错。"""
        by_split = {s: splits_df[splits_df["prm_split"] == s] for s in SPLITS}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            inst_a, inst_b = set(by_split[a]["instance_id"]), set(by_split[b]["instance_id"])
            repo_a, repo_b = set(by_split[a]["repo"]), set(by_split[b]["repo"])
            if inst_a & inst_b:
                raise ValueError(f"split 泄漏: {a}/{b} instance 交集非空（{len(inst_a & inst_b)} 个）")
            if repo_a & repo_b:
                raise ValueError(f"split 泄漏: {a}/{b} repo 交集非空（{sorted(repo_a & repo_b)[:5]}...）")

    # ------------------------------------------------------------------
    # iter_samples（§5.3）：遍历节点产出样本 dict
    # ------------------------------------------------------------------

    def _instance_payloads(self) -> Iterator[dict]:
        """逐实例装配 worker payload（按实例查询 nodes，避免整表物化）。

        每个节点就地附 ``mc / n_rollouts``（主进程从重算表查出；不在表中的节点
        ``mc=None``，worker 计入 ``no_rollouts`` 跳过）。
        """
        min_rollouts = int(self.cfg["build"].get("min_rollouts", 3))
        for iid in self._instance_ids:
            cur = self._conn.execute(
                "SELECT node_key, prefix_json, visits, in_pool, mc_score FROM nodes "
                "WHERE instance_id = ? AND node_key != 'root' ORDER BY node_key", (iid,))
            nodes = []
            for (nk, pj, v, ip, mcs) in cur.fetchall():
                mc, n_rollouts = self._mc_table.get((iid, nk), (None, 0))
                nodes.append({"node_key": nk, "prefix_json": pj, "visits": v, "in_pool": ip,
                              "mc_score": mcs, "mc": mc, "n_rollouts": n_rollouts})
            repo, split = self._split_map[iid]
            yield {
                "instance_id": iid,
                "repo": repo,
                "split": split,
                "head_user": raw.head_user_content(self._heads[iid]) if iid in self._heads else None,
                "min_rollouts": min_rollouts,
                "nodes": nodes,
                "cfg": self.cfg,
                "tokenizer": self.cfg.get("tokenizer"),
            }

    def iter_samples(self) -> Iterator[dict]:
        """遍历节点产出样本 dict（§5.3 字段清单；sample_id 在 dedupe 后分配）。"""
        workers = int(self.cfg["build"].get("workers", 1))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(build_instance_samples, self._instance_payloads(),
                                       chunksize=4):
                    yield from self._collect(result)
        else:
            for payload in self._instance_payloads():
                yield from self._collect(build_instance_samples(payload))

    def _collect(self, result: dict) -> Iterator[dict]:
        if result.get("instance_skipped"):
            self._audits["head_missing"] += 1
        for k, v in result["skipped"].items():
            self.skip_stats[k] = self.skip_stats.get(k, 0) + v
        yield from result["samples"]

    # ------------------------------------------------------------------
    # dedupe（§5.3）：键 (instance_id, prefix_node_key, step_index)；
    # 优先级 node_mc > leaf_chain（§5.1）
    # ------------------------------------------------------------------

    def dedupe(self) -> None:
        seen: dict[tuple, dict] = {}
        n_before = 0
        collisions = 0
        priority_wins = {"node_mc": 0, "leaf_chain": 0}
        for sample in self.iter_samples():
            n_before += 1
            key = (sample["instance_id"], sample["_prefix_key"], sample["step_index"])
            existing = seen.get(key)
            if existing is None:
                seen[key] = sample
                continue
            collisions += 1
            if LABEL_SOURCE_PRIORITY[sample["label_source"]] > \
                    LABEL_SOURCE_PRIORITY[existing["label_source"]]:
                seen[key] = sample  # 真实节点样本压过链式回填（§5.1 例子）
        self.samples = list(seen.values())
        for s in self.samples:
            priority_wins[s["label_source"]] += 1
        # 确定性排序（split → 实例 → 步位置 → 来源节点 → 标签来源）
        self.samples.sort(key=lambda s: (SPLIT_ORDER.get(s["split"], 9), s["instance_id"],
                                         s["step_index"], s["node_key"], s["label_source"]))
        # 分配全局唯一 sample_id + 剥离私有键
        for s in self.samples:
            s["sample_id"] = f"{s['instance_id']}::{s['node_key']}::{s['step_index']}"
            s.pop("_prefix_key", None)
        self.dedupe_stats = {
            "before": n_before,
            "after": len(self.samples),
            "collisions": collisions,
            "kept": priority_wins,
        }
        logger.info("dedupe: %d → %d（撞键 %d，保留 node_mc %d / leaf_chain %d）",
                    n_before, len(self.samples), collisions,
                    priority_wins["node_mc"], priority_wins["leaf_chain"])

    # ------------------------------------------------------------------
    # 类别权重（§5.2：train split 内按 label_binary 统计一次）
    # ------------------------------------------------------------------

    def compute_class_weights(self) -> None:
        train_bin = [s["label_binary"] for s in self.samples if s["split"] == "train"]
        w_pos, w_neg = labeling.class_weights(
            train_bin, cap=self.cfg["build"].get("class_weight_cap", 4.0))
        n_pos = sum(train_bin)
        self.class_weights = {
            "w_pos": w_pos, "w_neg": w_neg,
            "n_pos": n_pos, "n_neg": len(train_bin) - n_pos,
            "split": "train", "cap": self.cfg["build"].get("class_weight_cap", 4.0),
        }
        logger.info("class weights: w_pos=%.2f w_neg=%.2f（pos %d / neg %d）",
                    w_pos, w_neg, n_pos, len(train_bin) - n_pos)

    # ------------------------------------------------------------------
    # write（§5.3）：按 split 分写 parquet（zstd），每 FLUSH_ROWS 条 flush
    # ------------------------------------------------------------------

    def write(self) -> dict[str, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, Path] = {}
        for split in SPLITS:
            rows = [s for s in self.samples if s["split"] == split]
            path = self.out_dir / f"{split}.parquet"
            self._write_split(path, rows)
            files[split] = path
            logger.info("写出 %s: %d 条 → %s", split, len(rows), path)
        return files

    def _rows_to_table(self, rows: list[dict]) -> pa.Table:
        cols: dict[str, pa.Array] = {}
        for field in SCHEMA:
            if field.name == "messages":
                cols[field.name] = pa.array([messages_to_arrow(s["messages"]) for s in rows],
                                            type=field.type)
            else:
                cols[field.name] = pa.array([s[field.name] for s in rows], type=field.type)
        return pa.table(cols, schema=SCHEMA)

    def _write_split(self, path: Path, rows: list[dict]) -> None:
        writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        try:
            buf: list[dict] = []
            for s in rows:
                buf.append(s)
                if len(buf) >= FLUSH_ROWS:
                    writer.write_table(self._rows_to_table(buf))  # 每 1024 条 flush
                    buf = []
            if buf:
                writer.write_table(self._rows_to_table(buf))
        finally:
            writer.close()

    # ------------------------------------------------------------------
    # 长度报告（§6：全量渲染计长聚合 → 确定 max_length 的依据）
    # ------------------------------------------------------------------

    def length_stats(self) -> dict:
        tokens = np.array([s["rendered_tokens"] for s in self.samples], dtype=np.int64)
        if tokens.size == 0:
            return {"approx": True, "count": 0}
        candidates = self.cfg["build"].get("length_candidates", [8192, 16384, 32768])
        pct = {p: int(np.percentile(tokens, p))
               for p in (50, 75, 90, 95, 99)}
        coverage = {f"le_{L}": float((tokens <= L).mean()) for L in candidates}
        truncation = {f"over_{L}": float((tokens > L).mean()) for L in candidates}
        return {
            "approx": self.cfg.get("tokenizer") is None,
            "count": int(tokens.size),
            "min": int(tokens.min()), "max": int(tokens.max()), "mean": float(tokens.mean()),
            "percentiles": pct,
            "coverage": coverage,
            # §6：按 §7.2 规则的假想截断率上界——超限样本在 collator 中会先整步
            # 删除较早步（部分可收回），实际截断率 ≤ 此值；真正截断在训练期执行。
            "hypothetical_truncation_rate": truncation,
        }

    # ------------------------------------------------------------------
    # manifest（§4.2-3 / §5.2 / §11）
    # ------------------------------------------------------------------

    def manifest(self, files: dict[str, Path]) -> dict:
        after = raw.db_fingerprint(self._conn)
        mtime_after = Path(self._db_path).stat().st_mtime
        label_dist = {}
        for split in SPLITS:
            rows = [s for s in self.samples if s["split"] == split]
            bins = [s["label_binary"] for s in rows]
            label_dist[split] = {
                "count": len(rows),
                "pos_rate": (sum(bins) / len(bins)) if bins else 0.0,
                "mean_label": (float(np.mean([s["label"] for s in rows])) if rows else 0.0),
                "source": {
                    src: sum(1 for s in rows if s["label_source"] == src)
                    for src in ("node_mc", "leaf_chain")
                },
            }
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "template_version": TEMPLATE_VERSION,
            "template_hash": template_hash(),
            "tokenizer": self.cfg.get("tokenizer"),
            "tokenizer_backend": TrajectoryPreprocessor(self.cfg.get("tokenizer")).tokenizer_backend,
            "rendered_tokens_approx": self.cfg.get("tokenizer") is None,
            "db_path": self._db_path,
            "db_fingerprint_before": self._db_fingerprint_before,
            "db_fingerprint_after": after,
            "db_unchanged": (after == self._db_fingerprint_before
                             and mtime_after == self._db_mtime_before),
            "splits_path": self.cfg["splits"]["path"],
            "build": self.cfg["build"],
            "counts": {**{s: label_dist[s]["count"] for s in SPLITS},
                       "total": len(self.samples)},
            "label_distribution": label_dist,
            "dedupe": self.dedupe_stats,
            "skipped": {"prepare": self._audits, "nodes": self.skip_stats},
            "class_weights": self.class_weights,
            "length_stats": self.length_stats(),
            "files": {k: str(v) for k, v in files.items()},
        }

    # ------------------------------------------------------------------
    # build_report.md（构建审计报告）
    # ------------------------------------------------------------------

    def build_report_md(self, manifest: dict) -> str:
        ls = manifest["length_stats"]
        lines = [
            "# PRM 数据集构建报告",
            "",
            f"- 生成时间: {manifest['created_at']}  |  template: {manifest['template_version']}"
            f" (hash `{manifest['template_hash'][:12]}…`)",
            f"- tokenizer: `{manifest['tokenizer']}`"
            f"（backend={manifest['tokenizer_backend']}，approx={manifest['rendered_tokens_approx']}）",
            f"- DB: `{manifest['db_path']}`（构建前后行数一致: {manifest['db_unchanged']}）",
            f"- min_rollouts={self.cfg['build']['min_rollouts']}, leaf_chain="
            f"{self.cfg['build']['leaf_chain']}, soft_label_weight="
            f"{self.cfg['build']['soft_label_weight']}",
            "",
            "## 样本量",
            "",
            "| split | 条数 | 正类率 | node_mc | leaf_chain |",
            "| --- | --- | --- | --- | --- |",
        ]
        for split in SPLITS:
            d = manifest["label_distribution"][split]
            lines.append(
                f"| {split} | {d['count']} | {d['pos_rate']:.4f} "
                f"| {d['source']['node_mc']} | {d['source']['leaf_chain']} |")
        dw = manifest.get("class_weights") or {}
        lines += [
            "",
            f"## 类别权重（train split 统计）",
            "",
            f"- w_pos={dw.get('w_pos')}, w_neg={dw.get('w_neg')}"
            f"（pos {dw.get('n_pos')} / neg {dw.get('n_neg')}，cap {dw.get('cap')}）",
            "",
            "## 去重与跳过",
            "",
            f"- 去重: {manifest['dedupe']['before']} → {manifest['dedupe']['after']}"
            f"（撞键 {manifest['dedupe']['collisions']}）",
            f"- 节点级跳过: {manifest['skipped']['nodes']}",
            f"- 实例级审计: {manifest['skipped']['prepare']}",
            "",
            "## 长度分布（rendered_tokens，含 nothink 生成提示）",
            "",
        ]
        if ls.get("count"):
            lines += [
                f"- approx={ls['approx']}（无 tokenizer 时为 3.5 chars/token 近似）",
                f"- min/mean/max: {ls['min']} / {ls['mean']:.0f} / {ls['max']}",
                f"- p50/p75/p90/p95/p99: "
                + " / ".join(f"{ls['percentiles'][p]}" for p in (50, 75, 90, 95, 99)),
                f"- 覆盖率: " + ", ".join(f"≤{k[3:]}: {v:.4f}" for k, v in ls["coverage"].items()),
                f"- 假想截断率上界: "
                + ", ".join(f">{k[5:]}: {v:.4f}" for k, v in ls["hypothetical_truncation_rate"].items()),
                "",
                "> 依据本报告确定训练 max_length 后写入 `config/prm.yaml`（§6）；"
                "截断在训练/评估期由 collator 动态执行，parquet 存全量 messages。",
            ]
        else:
            lines.append("- （无样本）")
        return "\n".join(lines) + "\n"

    def length_report_md(self, ls: dict) -> str:
        """独立长度报告（§6：确定训练 max_length 的依据，先于训练产出）。"""
        lines = [
            "# PRM 长度分布报告（rendered_tokens，含 nothink 生成提示）",
            "",
            f"- approx = {ls.get('approx', True)}"
            "（`true` = 无 tokenizer，3.5 chars/token 近似；正式定 max_length 前应在"
            " CodeAgentRL-PRM 环境以真实 tokenizer 重建）",
        ]
        if not ls.get("count"):
            return "\n".join(lines + ["- （无样本）", ""]) + "\n"
        lines += [
            f"- 样本数: {ls['count']}  |  min / mean / max: {ls['min']} / {ls['mean']:.0f} / {ls['max']}",
            "",
            "| 分位 | p50 | p75 | p90 | p95 | p99 | max |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            "| tokens | "
            + " | ".join(str(ls["percentiles"][p]) for p in (50, 75, 90, 95, 99))
            + f" | {ls['max']} |",
            "",
            "| 候选 max_length | 覆盖率（≤L） | 假想截断率上界（>L） |",
            "| --- | --- | --- |",
        ]
        for k_cov, k_tr in zip(sorted(ls["coverage"]), sorted(ls["hypothetical_truncation_rate"])):
            L = k_cov.split("le_")[1]
            lines.append(f"| {L} | {ls['coverage'][k_cov]:.4f} "
                         f"| {ls['hypothetical_truncation_rate'][k_tr]:.4f} |")
        lines += [
            "",
            "> - 假想截断率上界 = rendered_tokens 超过候选值的样本占比；实际 §7.2 截断"
            "先整步删除较早步（部分超限样本可收回），真实截断率 ≤ 上界。",
            "> - 依据本表确定训练 max_length（§6 步骤 3），写入 `config/prm.yaml` 的"
            " `train.max_length` 后再开训。",
            "> - 截断不在构建期执行：parquet 存全量 messages，训练/评估期由 collator"
            " 按选定 max_length 动态截断，后续调整无需重建数据。",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # build（§5.3）：prepare → iter_samples → dedupe → write → 报告 → manifest
    # ------------------------------------------------------------------

    def build(self) -> dict:
        self.prepare()
        self.dedupe()
        if not self.samples:
            raise RuntimeError("构建结果为空：检查 DB/splits 配置与过滤条件")
        self.compute_class_weights()
        files = self.write()
        manifest = self.manifest(files)
        if not manifest["db_unchanged"]:
            raise RuntimeError("DB 在构建过程中发生变化（mtime/行数）——违反只读硬约束")
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out_dir / "build_report.md").write_text(
            self.build_report_md(manifest), encoding="utf-8")
        (self.out_dir / "length_report.md").write_text(
            self.length_report_md(manifest["length_stats"]), encoding="utf-8")
        spot = write_spot_check(self.out_dir, n=int(self.cfg["build"].get("spot_check_n", 20)))
        logger.info("构建完成: %s（manifest/build_report/length_report/%s 已写出）",
                    self.out_dir, spot.name)
        return manifest


def write_spot_check(out_dir: Path, n: int = 20, seed: int = 0) -> Path:
    """生成 `spot_check.md`：人工抽检报告（§13.2 步骤 3 的留档）。

    从构建产物里按**确定性**方式（split 分层 + 固定 seed 洗牌）抽 ``n`` 条样本，
    渲染出 PRM 实际会看到的 prompt 文本（system / issue 上下文 / 逐步轨迹 / 末轮
    指令）+ 标签来源与数值，供人眼核对：

    1. issue 是否**原文**进入 prompt、有没有 gold 泄漏（patch / 目标位置）；
    2. 轨迹的 assistant 动作（tool_calls 命令）与 tool observation 是否齐全；
    3. label / label_source / step_index 是否与被判定步一致（leaf 链回填的末步应为 0）。

    这是单测覆盖不到的一环（单测只能断言结构，看不出"渲染是否合理"）。
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    rows: list[dict] = []
    per_split = max(1, (n + 2) // 3)          # 向上取整后均分到 3 个 split
    for name in ("train", "dev", "test"):
        path = Path(out_dir) / f"{name}.parquet"
        if not path.exists():
            continue
        tbl = pq.read_table(path).to_pylist()
        if not tbl:
            continue
        rng.shuffle(tbl)
        rows.extend(tbl[:per_split])
    rows = rows[:n]

    lines = [
        "# PRM 样本抽检报告（`prm/build_dataset.py::write_spot_check`）", "",
        f"- 抽样 {len(rows)} 条（split 分层，seed={seed}）；字段与渲染口径见 §4/§5.4",
        "- 核对要点：① issue 原文且无 gold；② 动作/观察齐全；③ 标签与被判定步一致", "",
    ]
    for i, r in enumerate(rows, 1):
        lines += [
            f"## {i}. `{r['sample_id']}`", "",
            f"- split={r['split']} | label={r['label']:.4f}"
            f" | binary={r['label_binary']} | soft={r['label_soft']:.4f}"
            f" | source={r['label_source']}",
            f"- step_index={r['step_index']}/{r['step_count']}"
            f" | mc={r['mc_score']} | n_rollouts={r['n_rollouts']}"
            f" | rendered_tokens={r['rendered_tokens']}",
            "",
        ]
        for m in r["messages"]:
            role = m.get("role")
            body = m.get("content") or ""
            extra = ""
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                extra += f"\n    [tool_call] {fn.get('name')}({fn.get('arguments')})"
            if m.get("reasoning_content"):
                extra += f"\n    [reasoning] {str(m['reasoning_content'])[:200]}"
            lines += [f"- **{role}**{extra}", "", "  ```text",
                      "  " + str(body)[:2000].replace("\n", "\n  "), "  ```", ""]
    path = Path(out_dir) / "spot_check.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI（§5.5）
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m prm.build_dataset",
        description="从 MCTS 树库构建 PRM 训练集（只读 DB；docs/prm_training_plan.md §5）")
    p.add_argument("--config", default=None, help="yaml 配置（默认内置 DEFAULT_CONFIG）")
    p.add_argument("--db", default=None, help="MCTS 状态库（严格只读）")
    p.add_argument("--splits", default=None, help="splits.parquet 路径")
    p.add_argument("--output", default=None, help="输出目录")
    p.add_argument("--tokenizer", default=None, help="tokenizer 目录（缺省走 chars/token 近似）")
    p.add_argument("--min-rollouts", type=int, default=None, help="节点最少 rollout 数（§2.1-2）")
    p.add_argument("--leaf-chain", dest="leaf_chain", action="store_true", default=None,
                   help="启用 leaf 链式回填（§5.1）")
    p.add_argument("--no-leaf-chain", dest="leaf_chain", action="store_false",
                   help="禁用 leaf 链式回填（leaf 只产 1 条 node_mc 负样本）")
    p.add_argument("--soft-label-weight", type=float, default=None, help="混合 soft 的 soft 权重")
    p.add_argument("--class-weight-cap", type=float, default=None, help="w_neg 上限")
    p.add_argument("--workers", type=int, default=None, help="并行进程数")
    p.add_argument("--overwrite", action="store_true", default=None, help="覆盖已存在的输出")
    p.add_argument("--spot-check-only", action="store_true",
                   help="只从已有 parquet 生成 spot_check.md（不重建数据）")
    p.add_argument("--spot-check-n", type=int, default=20, help="抽检条数（默认 20）")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    cfg = DEFAULT_CONFIG
    if args.config:
        cfg = deep_merge(cfg, yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    cli = {
        "db": {"path": args.db},
        "splits": {"path": args.splits},
        "output": {"dir": args.output},
        "tokenizer": args.tokenizer,
        "build": {
            "min_rollouts": args.min_rollouts,
            "leaf_chain": args.leaf_chain,
            "soft_label_weight": args.soft_label_weight,
            "class_weight_cap": args.class_weight_cap,
            "workers": args.workers,
            "overwrite": args.overwrite,
        },
    }
    cfg = deep_merge(cfg, cli)
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    load_project_env()          # 入口装载 .env（GPU/CUDA 选择），早于 torch/CUDA 初始化
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args)
    out_dir = Path(cfg["output"]["dir"])
    if args.spot_check_only:
        # 仅从已有 parquet 生成人工抽检报告（不重建数据；§13.2 步骤 3 留档）
        path = write_spot_check(out_dir, n=int(args.spot_check_n))
        print(f"抽检报告已生成: {path}")
        return 0
    if not cfg["build"].get("overwrite", False):
        existing = [p for p in (f"{s}.parquet" for s in SPLITS) if (out_dir / p).exists()]
        if existing:
            raise SystemExit(f"输出已存在（{existing}），加 --overwrite 覆盖")
    builder = PRMDatasetBuilder(cfg)
    builder.build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
