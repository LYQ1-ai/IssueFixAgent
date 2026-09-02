# SPDX-License-Identifier: BSD-3-Clause

"""终态回报（PLAN §2.4 / D2 + docs/reward_design.md 综合方案）。

**两条判定模式**（``reward_mode``）：

1. **layered（主路径，2026-08-27 起，docs/reward_design.md 综合方案）**：
   把 ``file/class/func`` 视为同一条根因路径的不同完成深度 —— 每个位置只按**最深
   正确层级**计一次分（深度权重默认 ``{1:0.2, 2:0.5, 3:1.0}``），预测与 gold 做
   **一对一最大权匹配**（防蹭分/防刷分），聚合成 **Soft-F1** ``Reward = 2C/(M+N)``
   ∈ [0,1]（与 ``2·P·R/(P+R)`` 数学恒等，P=C/M、R=C/N）；gold 层级缺失按该位置
   最细可得层级 Lᵢ 折算（缺失实例可达满分）。二值化 ``correct = Reward ≥ τ``
   （默认 τ=0.6，落在 class 档 0.5 与 func 档 1.0 之间 ⇒ "正确 = 至少函数级精确
   定位"），N≥2 时可叠加 ``strict_multi_gate``（func 命中 ≥ ⌈N/2⌉）。
2. **independent（保留作对照）**：旧"三层独立 F1 相加"（∈[0,3]），供对比与回退。

**submission 协议**：提交方式为 ``submit_locations`` 工具（PLAN §2.2 改造），
submission = 结构化 locations JSON（``[{file, class_name, function_name}, ...]``）。
**无有效提交 → reward 0**（对齐 CodeScout：未调用 finish 工具即 0 分）。

diff 判定（``parse_diff_locations`` 等）保留为兼容/对照函数（旧魔法串提交路径）。
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from mcts.instances import Gold

# 文件头：+++ b/<path>（新/修改）或 --- a/<path>（删除）；/dev/null 是新建/删除的占位
_DIFF_HEADER_RE = re.compile(r"^(\+\+\+|---)\s+(?:[ab]/)?(/dev/null|.+)$", re.MULTILINE)
# hunk 头：@@ -a,b +c,d @@ <git function context>
_HUNK_RE = re.compile(r"^@@[^@\n]*@@\s*(.*)$", re.MULTILINE)
_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")

# ---------------------------------------------------------------------------
# layered 模式的默认深度权重（docs/reward_design.md §3）
# ---------------------------------------------------------------------------

DEFAULT_DEPTH_WEIGHTS: dict[int, float] = {1: 0.2, 2: 0.5, 3: 1.0}


def _norm_path(p: str) -> str:
    """去掉 diff 路径的 a/ b/ 前缀。"""
    p = p.strip()
    for prefix in ("a/", "b/"):
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def parse_diff_locations(patch: str) -> tuple[set[str], set[str], set[str]]:
    """从 git diff 文本解析 ``(files, modules, entities)`` 三粒度集合。

    规则（启发式，与 gold 的 ``path:name`` 口径对齐）：

    - files：全部 ``+++`` / ``---`` 头（排除 ``/dev/null``）；
    - 逐文件扫描 hunk：hunk 头（git function context）与 hunk 体内的
      ``class X`` / ``def f(`` 行共同确定当前 class 上下文与函数/方法名；
    - module = ``path:Class``（类内）或 ``path:func``（独立函数）；
    - entity = ``path:Class.func``（类内）或 ``path:func``（独立函数）。

    说明：gold 的 modules/entities 来自结构化 ``file_changes``；agent 侧只能从
    diff 推断（git function context + 体内容），属**可解释的近似** —— 单测锁定
    解析规则，M1 验收时人工抽检比对。
    """
    files: set[str] = set()
    for m in _DIFF_HEADER_RE.finditer(patch):
        path = _norm_path(m.group(2))
        if path and path != "/dev/null":
            files.add(path)

    modules: set[str] = set()
    entities: set[str] = set()
    current_file: Optional[str] = None
    current_class: Optional[str] = None

    lines = patch.splitlines()
    for i, line in enumerate(lines):
        m = _DIFF_HEADER_RE.match(line)
        if m:
            path = _norm_path(m.group(2))
            if path and path != "/dev/null" and m.group(1) == "+++":
                current_file = path
                current_class = None
            continue
        if current_file is None:
            continue
        hunk = _HUNK_RE.match(line)
        if hunk:
            # 新 hunk 重置 class 上下文：git function context 只对该 hunk 有效
            # （类体内的 def 由 hunk 体内扫描归属；跨 hunk 的 class 不沿用）
            current_class = None
            ctx = hunk.group(1).strip()
            cm = _CLASS_RE.search(ctx)
            if cm:
                current_class = cm.group(1)
            else:
                dm = _DEF_RE.search(ctx)
                if dm:
                    _add_func(modules, entities, current_file, None, dm.group(1))
            continue
        if line.startswith(("+", "-", " ")) and len(line) > 1:
            body = line[1:].lstrip()
            cm = _CLASS_RE.match(body)
            dm = _DEF_RE.match(body)
            if cm and not body.startswith("def"):
                current_class = cm.group(1)
            elif dm:
                _add_func(modules, entities, current_file, current_class, dm.group(1))
    return files, modules, entities


def _add_func(
    modules: set[str], entities: set[str],
    file: str, cls: Optional[str], func: str,
) -> None:
    """按 codescout 口径把函数/方法加入 modules 与 entities。"""
    if cls:
        modules.add(f"{file}:{cls}")
        entities.add(f"{file}:{cls}.{func}")
    else:
        modules.add(f"{file}:{func}")
        entities.add(f"{file}:{func}")


def compute_file_f1_score(
    predicted: set[str] | list[str],
    true: set[str] | list[str],
    beta: float = 1.0,
) -> float:
    """codescout ``compute_file_f1_score`` 同款：ground truth 为空 → 0。"""
    pred, truth = set(predicted), set(true)
    if not truth:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    if precision + recall <= 0:
        return 0.0
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def patch_localization_f1(
    agent_patch: str,
    gold: Gold,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, dict]:
    """agent diff vs gold 三粒度 F1 加权和（∈ [0, sum(weights)]）。

    返回 ``(reward, {"file_f1": ..., "module_f1": ..., "entity_f1": ..., ...})``。
    """
    if not agent_patch or not agent_patch.strip():
        return 0.0, {"file_f1": 0.0, "module_f1": 0.0, "entity_f1": 0.0,
                     "reward": 0.0, "reason": "empty_patch"}
    pred_files, pred_modules, pred_entities = parse_diff_locations(agent_patch)
    file_f1 = compute_file_f1_score(pred_files, gold.files)
    module_f1 = compute_file_f1_score(pred_modules, gold.modules)
    entity_f1 = compute_file_f1_score(pred_entities, gold.entities)
    w_file, w_module, w_entity = weights
    reward = file_f1 * w_file + module_f1 * w_module + entity_f1 * w_entity
    return reward, {
        "file_f1": file_f1,
        "module_f1": module_f1,
        "entity_f1": entity_f1,
        "reward": reward,
        "pred_files": sorted(pred_files),
        "pred_modules": sorted(pred_modules),
        "pred_entities": sorted(pred_entities),
    }


def is_diff_like(text: str) -> bool:
    """粗略判断一段文本是否是 git diff（submission 回退路径用）。"""
    if not text:
        return False
    head = text.lstrip()[:2000]
    return ("diff --git" in head or "+++ " in head or "@@ " in head)


def reward_from_patch(
    patch: str,
    gold: Gold,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
) -> tuple[float, bool, dict]:
    """patch → ``(reward, correct, details)``；``correct = reward ≥ θ``。"""
    reward, details = patch_localization_f1(patch, gold, weights)
    return reward, reward >= threshold, details


# ---------------------------------------------------------------------------
# 结构化 submission 解析（两模式共用）
# ---------------------------------------------------------------------------


def locations_from_submission(submission: str) -> Optional[list[dict]]:
    """把 ``Submitted`` 的 submission（JSON）宽松解析成规范化 locations。

    返回 ``list[{"file": str, "class_name": str|None, "function_name": str|None}]``；
    解析失败 / 结构非法返回 ``None``（调用方按 reward 0 处理）。判定侧保持宽松
    （agent 侧的严格校验在 ``agent.submit_tool.parse_submit_locations``），
    任何不符合 {file 必填非空} 的输入都视为无效提交。
    """
    if not submission or not submission.strip():
        return None
    try:
        data = json.loads(submission)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        file = item.get("file")
        if not isinstance(file, str) or not file.strip():
            return None
        class_name = item.get("class_name")
        function_name = item.get("function_name")
        if class_name is not None and not isinstance(class_name, str):
            return None
        if function_name is not None and not isinstance(function_name, str):
            return None
        out.append({"file": file.strip(),
                    "class_name": class_name, "function_name": function_name})
    return out


def parse_structured_locations(
    locations: list[dict],
) -> tuple[set[str], set[str], set[str]]:
    """结构化 locations → ``(files, modules, entities)`` 三粒度集合。

    对齐 codescout ``parse_structured_outputs`` 口径：

    - files：全部 ``file``；
    - modules：``file:Class``（有 class）或 ``file:func``（无 class 有 func）；
    - entities：``file:Class.func``（class+func）或 ``file:func``（仅 func）。
    """
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for loc in locations:
        file = loc["file"]
        files.add(file)
        class_name = loc.get("class_name")
        function_name = loc.get("function_name")
        if class_name:
            modules.add(f"{file}:{class_name}")
        elif function_name:
            modules.add(f"{file}:{function_name}")
        if class_name and function_name:
            entities.add(f"{file}:{class_name}.{function_name}")
        elif function_name:
            entities.add(f"{file}:{function_name}")
    return files, modules, entities


def locations_localization_f1(
    locations: list[dict],
    gold: Gold,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, dict]:
    """（independent 模式）结构化 locations vs gold 三粒度 F1 加权和。"""
    if not locations:
        return 0.0, {"file_f1": 0.0, "module_f1": 0.0, "entity_f1": 0.0,
                     "reward": 0.0, "reason": "empty_locations"}
    pred_files, pred_modules, pred_entities = parse_structured_locations(locations)
    file_f1 = compute_file_f1_score(pred_files, gold.files)
    module_f1 = compute_file_f1_score(pred_modules, gold.modules)
    entity_f1 = compute_file_f1_score(pred_entities, gold.entities)
    w_file, w_module, w_entity = weights
    reward = file_f1 * w_file + module_f1 * w_module + entity_f1 * w_entity
    return reward, {
        "file_f1": file_f1,
        "module_f1": module_f1,
        "entity_f1": entity_f1,
        "reward": reward,
        "pred_files": sorted(pred_files),
        "pred_modules": sorted(pred_modules),
        "pred_entities": sorted(pred_entities),
    }


def reward_from_locations(
    locations: list[dict],
    gold: Gold,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
) -> tuple[float, bool, dict]:
    """（independent 模式）locations → ``(reward, correct, details)``。"""
    reward, details = locations_localization_f1(locations, gold, weights)
    return reward, reward >= threshold, details


# ---------------------------------------------------------------------------
# layered 模式（docs/reward_design.md 综合方案，主路径）
# ---------------------------------------------------------------------------


def triples_from_gold(gold: Gold) -> list[tuple[str, Optional[str], Optional[str]]]:
    """把 gold 的三集合（files/modules/entities）还原为位置三元组列表。

    与 codescout ``file_changes`` 的结构对应：

    - ``entities``（``path:Class.method`` / ``path:func``）→ ``(path, class, func)``
      或 ``(path, None, func)``；
    - 剩余 ``modules``（``path:Class`` 或漏标 entity 的 ``path:func``）→
      ``(path, name, None)``（最细为 class/函数级位置）；
    - 剩余 ``files`` → ``(path, None, None)``（最细为文件级位置）。

    注：module 字符串无法区分类名与独立函数名（解析歧义，见 docs/reward_design.md
    §8），此处统一记为 class 级位置 —— 匹配时 ``match_depth`` 会按 gold 实际结构
    判定，不影响正确性。
    """
    files = set(gold.files or [])
    mods = set(gold.modules or [])
    ents = set(gold.entities or [])
    out: list[tuple[str, Optional[str], Optional[str]]] = []
    for e in ents:
        path, _, name = e.rpartition(":")
        if "." in name:
            cls, _, func = name.partition(".")
            if cls and func:
                out.append((path, cls, func))
                continue
        out.append((path, None, name))
    covered = {(f, c, fn) for f, c, fn in out}
    for m in mods:
        path, _, name = m.rpartition(":")
        if any(x[0] == path and (x[1] == name or x[2] == name) for x in covered):
            continue
        out.append((path, name, None))
    covered = {(f, c, fn) for f, c, fn in out}
    for f in files:
        if f not in {x[0] for x in covered}:
            out.append((f, None, None))
    return out


def pred_triples(
    locations: list[dict],
) -> list[tuple[str, Optional[str], Optional[str]]]:
    """结构化 locations → 位置三元组列表（与 gold 同口径）。"""
    return [(loc["file"], loc.get("class_name"), loc.get("function_name"))
            for loc in locations]


def g_max_depth(g: tuple) -> int:
    """gold 位置 g 的最细可得层级（有 func→3，有 class→2，否则 1）。"""
    _, gc, gf_ = g
    return 3 if gf_ else (2 if gc else 1)


def match_depth(
    p: tuple[str, Optional[str], Optional[str]],
    g: tuple[str, Optional[str], Optional[str]],
) -> int:
    """预测 p vs gold g 的最深正确层级（0/1/2/3；docs/reward_design.md §4）。

    级联判定：file 不同 → 0；file 同 → 判 class（gold 有 class 时，class 不匹配
    → 1）；class 匹配（或 gold 无 class）→ 判 func：gold 有 func 时 func 同 → 3、
    错/缺 → 2（或 1）；gold 无 func（最细即 class）→ 2；gold 仅文件 → 1。
    """
    pf, pc, pf_ = p
    gf, gc, gf_ = g
    if pf != gf:
        return 0
    if gc:  # gold 有 class
        if pc != gc:
            return 1          # class 不匹配/未给 → file 级
        if gf_:
            return 3 if pf_ == gf_ else 2   # class 对 → func 级或 class 级
        return 2              # gold 最细就是 class
    # gold 无 class（独立函数 / 文件级位置）
    if gf_:
        return 3 if pf_ == gf_ else 1       # 独立函数：func 对 → 3，否则 file 级
    return 1                  # gold 仅文件级


def max_weight_matching(
    preds: list[tuple],
    golds: list[tuple],
    depth_weights: dict[int, float],
) -> list[tuple[float, int, int, int]]:
    """深→浅贪心一对一最大权匹配（等价匈牙利最优，docs/reward_design.md §4-3）。

    返回 ``[(score, depth, pred_idx, gold_idx), ...]``，每个 pred/gold 至多出现
    一次；得分完全由深度决定 ⇒ 最深匹配对总是可进入最优解（可交换性）。
    """
    cands = []
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            d = match_depth(p, g)
            if d > 0:
                cands.append((depth_weights[d], d, i, j))
    cands.sort(key=lambda x: (-x[0], -x[1]))
    used_p, used_g = set(), set()
    pairs: list[tuple[float, int, int, int]] = []
    for score, d, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((score, d, i, j))
    return pairs


def locations_localization_f1_layered(
    locations: list[dict],
    gold: Gold,
    *,
    depth_weights: Optional[dict[int, float]] = None,
    missing_level_norm: bool = True,
    overpromise_penalty: float = 0.0,
    duplicate_penalty: float = 0.0,
) -> tuple[float, dict]:
    """（layered）结构化 locations vs gold 的 Soft-F1（∈ [0,1]）。

    ``Reward = 2C/(M+N)``；C = 匹配对折算分之和（默认按 gold 最细可得层级 Lᵢ
    折算 credit = s(d)/s(Lᵢ)，缺失实例可达满分）；惩罚项（默认 0）扣在 C 上。

    返回 ``(reward, details)``，details 含 M/N/C/深度分布/匹配对明细（供落库分析，
    reward 连续分数与分层细节一并记录在 rollout 数据中）。
    """
    w = dict(DEFAULT_DEPTH_WEIGHTS if depth_weights is None else depth_weights)
    if not locations:
        return 0.0, {"mode": "layered", "reward": 0.0, "correct": False,
                     "M": 0, "N": 0, "C": 0.0, "reason": "empty_locations"}
    preds = pred_triples(locations)
    golds = triples_from_gold(gold)
    if not golds:
        return 0.0, {"mode": "layered", "reward": 0.0, "correct": False,
                     "M": len(preds), "N": 0, "C": 0.0, "reason": "empty_gold"}

    pairs = max_weight_matching(preds, golds, w)
    C = 0.0
    matched_idx: set[int] = set()
    matched_detail: list[dict] = []
    depth_hist: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for score, d, i, j in pairs:
        L = g_max_depth(golds[j])
        credit = score / w[L] if missing_level_norm else score
        C += credit
        matched_idx.add(i)
        depth_hist[d] += 1
        matched_detail.append({
            "pred": list(preds[i]), "gold": list(golds[j]),
            "depth": d, "score": score, "credit": credit,
        })

    # 惩罚项（默认 0；docs/reward_design.md §3）
    penalty = 0.0
    unmatched = [i for i in range(len(preds)) if i not in matched_idx]
    if overpromise_penalty:
        for i in unmatched:
            _, c, fn = preds[i]
            penalty += overpromise_penalty * ((1 if c else 0) + (1 if fn else 0))
    if duplicate_penalty:
        matched_keys: set[tuple] = set()
        for _, _, i, _ in pairs:
            pf, pc, _ = preds[i]
            matched_keys.add((pf,))
            matched_keys.add((pf, pc))
        for i in unmatched:
            pf, pc, _ = preds[i]
            if (pf,) in matched_keys or (pf, pc) in matched_keys:
                penalty += duplicate_penalty
    C = max(0.0, C - penalty)

    M, N = len(preds), len(golds)
    reward = 2 * C / (M + N) if (M + N) else 0.0
    details: dict[str, Any] = {
        "mode": "layered",
        "reward": reward,
        "M": M, "N": N, "C": C,
        "exact_hits": depth_hist[3],
        "depth_hist": depth_hist,
        "matched": matched_detail,
        "unmatched_preds": [list(p) for i, p in enumerate(preds) if i not in matched_idx],
        "unmatched_golds": [list(g) for j, g in enumerate(golds)
                            if j not in {j2 for _, _, _, j2 in pairs}],
        "penalty": penalty,
    }
    return reward, details


def reward_from_locations_layered(
    locations: list[dict],
    gold: Gold,
    *,
    threshold: float = 0.6,
    depth_weights: Optional[dict[int, float]] = None,
    strict_multi_gate: bool = True,
    missing_level_norm: bool = True,
    overpromise_penalty: float = 0.0,
    duplicate_penalty: float = 0.0,
) -> tuple[float, bool, dict]:
    """（layered）locations → ``(reward, correct, details)``。

    ``correct = reward ≥ τ``；``strict_multi_gate`` 开启且 N≥2 时额外要求
    func 级命中数 ≥ ⌈N/2⌉（docs/reward_design.md §3）。
    """
    reward, details = locations_localization_f1_layered(
        locations, gold, depth_weights=depth_weights,
        missing_level_norm=missing_level_norm,
        overpromise_penalty=overpromise_penalty,
        duplicate_penalty=duplicate_penalty,
    )
    correct = reward >= threshold
    if strict_multi_gate and details["N"] >= 2:
        if details["exact_hits"] < math.ceil(details["N"] / 2):
            correct = False
    details["correct"] = correct
    details["threshold"] = threshold
    return reward, correct, details


def reward_from_trajectory_exit(
    exit_status: str,
    submission: str,
    gold: Gold,
    *,
    threshold: float = 0.6,
    mode: str = "layered",
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    depth_weights: Optional[dict[int, float]] = None,
    strict_multi_gate: bool = True,
    missing_level_norm: bool = True,
    overpromise_penalty: float = 0.0,
    duplicate_penalty: float = 0.0,
) -> tuple[float, bool, dict]:
    """**结构化判定统一入口**：仅 ``Submitted`` 且 submission 为合法 locations
    JSON 才计算 reward；其它情况（未提交 / 无效提交）→ ``(0, False, {reason})``。

    - ``mode="layered"``：docs/reward_design.md 综合方案（默认）；
    - ``mode="independent"``：旧三层独立 F1 相加（对照/回退）。

    对齐 CodeScout：未调用 finish 工具（或 sanity check 失败）即 reward 0。
    """
    if exit_status != "Submitted":
        return 0.0, False, {"mode": mode, "reason": "not_submitted",
                            "exit_status": exit_status, "reward": 0.0}
    locations = locations_from_submission(submission)
    if locations is None:
        return 0.0, False, {"mode": mode, "reason": "invalid_submission",
                            "exit_status": exit_status, "reward": 0.0}
    if mode == "independent":
        reward, correct, details = reward_from_locations(
            locations, gold, weights, threshold)
        details["mode"] = "independent"
        return reward, correct, details
    return reward_from_locations_layered(
        locations, gold, threshold=threshold, depth_weights=depth_weights,
        strict_multi_gate=strict_multi_gate,
        missing_level_norm=missing_level_norm,
        overpromise_penalty=overpromise_penalty,
        duplicate_penalty=duplicate_penalty,
    )


__all__ = [
    "DEFAULT_DEPTH_WEIGHTS",
    "parse_diff_locations",
    "compute_file_f1_score",
    "patch_localization_f1",
    "is_diff_like",
    "reward_from_patch",
    "locations_from_submission",
    "parse_structured_locations",
    "locations_localization_f1",
    "reward_from_locations",
    "triples_from_gold",
    "pred_triples",
    "g_max_depth",
    "match_depth",
    "max_weight_matching",
    "locations_localization_f1_layered",
    "reward_from_locations_layered",
    "reward_from_trajectory_exit",
]
