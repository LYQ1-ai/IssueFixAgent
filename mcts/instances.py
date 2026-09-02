# SPDX-License-Identifier: BSD-3-Clause

"""数据预处理（PLAN §1 / 里程碑 M0）：读取 → 过滤 → GT 抽取 → 落盘 + 数据报告。

迁移自 ``ref_papers/codescout`` 的数据处理部分：

- **读取**：``pd.read_parquet`` 直读本地 parquet —— 对应 codescout
  ``tests/test_single_file_localization.py`` / ``test_single_prompt.py`` 的读法；
  ``src/build_dataset.py`` 是这些 parquet 的**生成代码**（HF → 加工 → 落盘）。
- **GT 抽取（三粒度 gold）**：按 ``src/rewards/file_localization/file_localization.py``
  的 ``multilevel_localization_f1_reward`` 解析口径实现 —— files 取自每条
  ``change["file"]``；modules / entities 取自 ``change["changes"]`` 的
  ``edited_*`` 与 ``added_*``（None 兜底，对齐 PLAN §1.2）。本模块与后续
  ``mcts/reward.py`` 的 agent patch 解析**复用同一解析器**，保证 GT / 预测可比。
- **过滤**：对齐 PLAN §1.2 的五条规则（空语句 / patch 新建删除文件 / 非 Python
  GT / 重复与非法格式 / 语句长度上下限），输出过滤报告。

命令行用法（在项目根目录执行）::

    python -m mcts.instances build                     # 读 config.yaml 默认路径
    python -m mcts.instances build --data-dir <dir>    # 覆盖数据目录（读 {dir}/train|validation.parquet）
    python -m mcts.instances build --output outputs/mcts

产出（``outputs/mcts/``）：``instances.parquet``（每行附解析好的 gold）+
``data_report.json`` / ``data_report.md``（原数量 / 保留数量 / 各规则剔除数 /
repo 数，验收对照 CodeScout 的 39K / 128 repos 数量级）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from mcts.config import load_config, resolve_path

logger = logging.getLogger("mcts.instances")

_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# SWE-Smith 仓库名末尾的 commit8：``swesmith/kurtmckee__feedparser.cad965a3``
_COMMIT8_RE = re.compile(r"\.([0-9a-fA-F]{8})$")
# git diff 中“新建 / 删除文件”的标志（过滤规则 2）
_NEW_FILE_RE = re.compile(r"^new file mode", re.MULTILINE)
_DELETED_FILE_RE = re.compile(r"^deleted file mode", re.MULTILINE)
_DEV_NULL_RE = re.compile(r"^(---|\+\+\+) /dev/null\b", re.MULTILINE)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gold:
    """三粒度 ground truth（与 codescout ``multilevel_localization_f1_reward``
    的 GT 解析口径一致）。

    - ``files``：每条 change 的 ``file``（如 ``src/pptx/chart/plot.py``）；
    - ``modules``：``edited_modules`` + ``added_modules``（如
      ``src/pptx/chart/plot.py:PlotTypeInspector``）；
    - ``entities``：``edited_entities`` + ``added_entities``（如
      ``src/pptx/chart/plot.py:PlotTypeInspector._differentiate_xy_chart_type``）。
    """

    files: frozenset[str] = frozenset()
    modules: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()

    def is_empty(self) -> bool:
        return not (self.files or self.modules or self.entities)

    def to_dict(self) -> dict:
        return {
            "files": sorted(self.files),
            "modules": sorted(self.modules),
            "entities": sorted(self.entities),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Gold":
        if not d:
            return cls()
        return cls(
            files=frozenset(d.get("files", [])),
            modules=frozenset(d.get("modules", [])),
            entities=frozenset(d.get("entities", [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: Optional[str]) -> "Gold":
        if not s:
            return cls()
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class Instance:
    """统一内部实例记录（PLAN §1.1）。

    ``repo`` 列本身即仓库地址字符串（SWE-Smith：``swesmith/{owner}__{repo}.{commit8}``，
    克隆地址即 ``https://github.com/{repo}.git``）；``commit8`` 是仓库名末尾的
    快照 commit（**不是**环境准备要 checkout 的 commit —— 见 ``mcts/env.py``）。
    """

    instance_id: str
    repo: str
    owner: str
    name: str
    commit8: Optional[str]
    base_commit: Optional[str]
    problem_statement: str
    patch: str
    use_patch: bool
    gold: Gold
    source: str = ""

    @property
    def repo_url(self) -> str:
        """完整克隆地址（codescout ``clone_instance`` 的 URL 拼法）。"""
        return f"https://github.com/{self.repo}.git"

    def to_row(self) -> dict:
        """转成 DataFrame 行（gold 以 JSON 字符串落盘，便于 parquet 往返）。"""
        d = asdict(self)
        d["gold"] = self.gold.to_json()
        return d

    @classmethod
    def from_row(cls, row: dict) -> "Instance":
        row = dict(row)
        row["gold"] = Gold.from_json(row.get("gold"))
        return cls(**row)


# ---------------------------------------------------------------------------
# 解析辅助（codescout 口径）
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list:
    """把可能为 None / ndarray / list / JSON 字符串的值规整成 list。"""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return [value]
        return parsed if isinstance(parsed, list) else [value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _py_path_part(value: str) -> str:
    """module/entity 形如 ``path:name``，取路径部分用于 Python 文件判断。"""
    return value.split(":", 1)[0].strip()


def is_python_path(value: str) -> bool:
    """GT 条目是否为 Python 文件（过滤规则 3：忽略 README 等非 Python 条目）。"""
    p = _py_path_part(value).lower()
    return p.endswith((".py", ".pyi")) or ".py/" in p


def parse_repo(repo: str) -> tuple[str, str, Optional[str]]:
    """把 codescout 的 ``repo`` 列解析为 ``(owner, name, commit8)``。

    - SWE-Smith：``swesmith/theskumar__python-dotenv.2b8635b7`` ->
      ``("swesmith", "theskumar__python-dotenv", "2b8635b7")``；
    - 普通仓库：``django/django`` -> ``("django", "django", None)``。

    Raises:
        ValueError: 不含 ``/``（无法拆出 owner/repo）。
    """
    repo = repo.strip()
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "/" not in repo:
        raise ValueError(f"repo must be in 'owner/repo' format, got: {repo!r}")
    owner, rest = repo.split("/", 1)
    m = _COMMIT8_RE.search(rest)
    if m:
        name, commit8 = rest[: m.start()], m.group(1)
    else:
        name, commit8 = rest, None
    return owner, name, commit8


def extract_gold(
    file_changes: Any,
    *,
    python_only: bool = True,
    include_added_modules: bool = True,
    include_added_entities: bool = True,
) -> Gold:
    """从 ``file_changes`` 抽取三粒度 gold（迁移 codescout
    ``multilevel_localization_f1_reward`` 的 GT 解析部分）。

    - files：每条 ``change["file"]``；
    - modules：``changes["edited_modules"]``（+ ``added_modules``）；
    - entities：``changes["edited_entities"]``（+ ``added_entities``）；
    - 以上字段可能为 None / ndarray / list，统一 ``_as_list`` 兜底；
    - ``python_only=True`` 时忽略非 Python 文件条目（过滤规则 3）。
    """
    files: list[str] = []
    modules: list[str] = []
    entities: list[str] = []
    for change in _as_list(file_changes):
        if not isinstance(change, dict):
            continue
        f = change.get("file")
        if isinstance(f, str) and f:
            files.append(f)
        changes = change.get("changes")
        if not isinstance(changes, dict):
            continue
        for key in ("edited_modules", "added_modules") if include_added_modules else ("edited_modules",):
            modules.extend(_as_list(changes.get(key)))
        for key in ("edited_entities", "added_entities") if include_added_entities else ("edited_entities",):
            entities.extend(_as_list(changes.get(key)))
    if python_only:
        files = [f for f in files if is_python_path(f)]
        modules = [m for m in modules if is_python_path(m)]
        entities = [e for e in entities if is_python_path(e)]
    return Gold(frozenset(files), frozenset(modules), frozenset(entities))


def patch_creates_or_deletes_files(patch: str) -> bool:
    """gold patch 是否新建或删除文件（过滤规则 2：agent 无法预测新建文件名，
    删除文件没有 ground truth）。"""
    return bool(
        _NEW_FILE_RE.search(patch)
        or _DELETED_FILE_RE.search(patch)
        or _DEV_NULL_RE.search(patch)
    )


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = ("instance_id", "file_changes", "repo", "problem_statement", "patch")


def read_instances(
    parquet_path: str | Path,
    *,
    source: Optional[str] = None,
    gold_kwargs: Optional[dict] = None,
) -> list[Instance]:
    """读取一个 parquet 文件为 ``Instance`` 列表（PLAN §1.1）。

    ``source`` 缺省取文件名（如 ``train`` / ``validation``）；``file_changes``
    支持 ndarray / list / JSON 字符串三种形态（本仓库 parquet 实测为 ndarray）。
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"parquet not found: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}; "
            f"got {sorted(df.columns.tolist())}"
        )
    src = source if source is not None else path.stem
    gk = gold_kwargs or {}
    out: list[Instance] = []
    for record in df.to_dict("records"):
        repo = str(record["repo"])
        try:
            owner, name, commit8 = parse_repo(repo)
        except ValueError:
            owner, name, commit8 = "", repo, None  # 非法格式行 -> 过滤阶段剔除
        out.append(
            Instance(
                instance_id=str(record["instance_id"]),
                repo=repo,
                owner=owner,
                name=name,
                commit8=commit8,
                base_commit=record.get("base_commit"),
                problem_statement=str(record["problem_statement"]),
                patch=str(record["patch"]),
                use_patch=bool(record.get("use_patch", True)),
                gold=extract_gold(record["file_changes"], **gk),
                source=src,
            )
        )
    return out


def instances_from_parquet(path: str | Path) -> list[Instance]:
    """从 ``instances.parquet``（本模块落盘格式，gold 为 JSON 字符串）读回实例。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"parquet not found: {path}")
    df = pd.read_parquet(path)
    return [Instance.from_row(r) for r in df.to_dict("records")]


# ---------------------------------------------------------------------------
# 过滤（PLAN §1.2）
# ---------------------------------------------------------------------------


def filter_instances(
    instances: Iterable[Instance], cfg: Optional[dict] = None
) -> tuple[list[Instance], dict]:
    """按 PLAN §1.2 过滤实例，返回 ``(保留列表, 过滤报告)``。

    ``cfg`` 缺省时加载 ``mcts/config.yaml`` 的 ``filter`` 段（与
    ``build_dataset`` / CLI 行为一致）；显式传入的 dict 直接使用。

    报告含每条规则的剔除数、保留/剔除总量与 repo 数（验收对照数量级）。
    """
    if cfg is None:
        cfg = load_config()
    f = cfg.get("filter", {})
    instances = list(instances)
    drop_counts: dict[str, int] = {}
    seen: set[str] = set()
    kept: list[Instance] = []

    def drop(rule: str) -> None:
        drop_counts[rule] = drop_counts.get(rule, 0) + 1

    for inst in instances:
        if f.get("drop_empty_statement", True) and not inst.problem_statement.strip():
            drop("empty_statement")
            continue
        min_len = f.get("min_statement_len", 1)
        if min_len and len(inst.problem_statement) < min_len:
            drop("statement_too_short")
            continue
        max_len = f.get("max_statement_len")
        if max_len and len(inst.problem_statement) > max_len:
            drop("statement_too_long")
            continue
        if f.get("drop_created_deleted_files", True) and patch_creates_or_deletes_files(inst.patch):
            drop("created_or_deleted_files")
            continue
        if f.get("drop_no_valid_gold", True) and inst.gold.is_empty():
            drop("no_valid_gold")
            continue
        if f.get("drop_duplicates", True):
            if inst.instance_id in seen:
                drop("duplicate_instance_id")
                continue
            seen.add(inst.instance_id)
        if f.get("drop_invalid_format", True) and (
            "/" not in inst.repo or not inst.owner or not _INSTANCE_ID_RE.match(inst.instance_id)
        ):
            drop("invalid_format")
            continue
        kept.append(inst)

    kept_repos = len({i.repo for i in kept})
    report = {
        "original_total": len(instances),
        "kept_total": len(kept),
        "dropped_total": sum(drop_counts.values()),
        "rules": dict(sorted(drop_counts.items())),
        "n_repos_original": len({i.repo for i in instances}),
        "n_repos_kept": kept_repos,
    }
    return kept, report


# ---------------------------------------------------------------------------
# 数据统计（数据报告用）
# ---------------------------------------------------------------------------


def _repo_distribution(instances: list[Instance]) -> dict:
    counts = {}
    for i in instances:
        counts[i.repo] = counts.get(i.repo, 0) + 1
    vals = sorted(counts.values())
    return {
        "n_repos": len(vals),
        "min": vals[0] if vals else 0,
        "median": vals[len(vals) // 2] if vals else 0,
        "max": vals[-1] if vals else 0,
    }


def _statement_stats(instances: list[Instance]) -> dict:
    lens = sorted(len(i.problem_statement) for i in instances)
    if not lens:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    return {
        "min": lens[0],
        "p50": lens[len(lens) // 2],
        "p95": lens[int(len(lens) * 0.95)],
        "max": lens[-1],
    }


def _gold_stats(instances: list[Instance]) -> dict:
    return {
        "n_with_files": sum(1 for i in instances if i.gold.files),
        "n_with_modules": sum(1 for i in instances if i.gold.modules),
        "n_with_entities": sum(1 for i in instances if i.gold.entities),
        "n_empty": sum(1 for i in instances if i.gold.is_empty()),
    }


# ---------------------------------------------------------------------------
# 组装：build_dataset（读取 → 过滤 → 落盘 + 报告）
# ---------------------------------------------------------------------------


def build_dataset(
    parquet_paths: list[str | Path],
    *,
    output_dir: str | Path,
    cfg: Optional[dict] = None,
) -> dict:
    """M0 数据层主流程：读取全部 parquet -> 过滤 -> 写 ``instances.parquet``
    + ``data_report.json`` / ``data_report.md``，返回报告 dict。"""
    cfg = load_config() if cfg is None else cfg
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_instances: list[Instance] = []
    per_source: dict[str, int] = {}
    for path in parquet_paths:
        instances = read_instances(path, gold_kwargs=cfg.get("gold", {}))
        all_instances.extend(instances)
        for i in instances:
            per_source[i.source] = per_source.get(i.source, 0) + 1

    kept, report = filter_instances(all_instances, cfg)
    report["source_counts"] = dict(sorted(per_source.items()))
    report["repo_distribution"] = _repo_distribution(kept)
    report["statement_len"] = _statement_stats(kept)
    report["gold_stats"] = _gold_stats(kept)
    report["config"] = {
        "filter": cfg.get("filter", {}),
        "gold": cfg.get("gold", {}),
    }

    rows = [i.to_row() for i in kept]
    df = pd.DataFrame(rows, columns=[
        "instance_id", "repo", "owner", "name", "commit8", "base_commit",
        "problem_statement", "patch", "use_patch", "source", "gold",
    ])
    inst_path = output_dir / "instances.parquet"
    df.to_parquet(inst_path, index=False)
    report["instances_parquet"] = str(inst_path)

    (output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "data_report.md").write_text(
        _render_report_md(report), encoding="utf-8"
    )
    logger.info(
        "build_dataset: %d -> %d instances (%d repos); -> %s",
        report["original_total"], report["kept_total"], report["n_repos_kept"], inst_path,
    )
    return report


def _render_report_md(report: dict) -> str:
    rules = report.get("rules", {})
    lines = [
        "# M0 数据预处理报告（PLAN §1）",
        "",
        f"- 原始实例数：**{report['original_total']}**（分源："
        + "、".join(f"{k}={v}" for k, v in report.get("source_counts", {}).items())
        + f"）；保留：**{report['kept_total']}**；剔除：**{report['dropped_total']}**",
        f"- 仓库数：原始 **{report['n_repos_original']}** / 保留 **{report['n_repos_kept']}**"
        f"（验收对照 CodeScout ≈39K / 128 repos 数量级）",
        "",
        "## 各规则剔除数",
        "",
        "| 规则 | 剔除数 |",
        "| --- | ---: |",
    ]
    if rules:
        lines += [f"| {k} | {v} |" for k, v in rules.items()]
    else:
        lines.append("| （无剔除） | 0 |")
    rd = report.get("repo_distribution", {})
    sl = report.get("statement_len", {})
    gs = report.get("gold_stats", {})
    lines += [
        "",
        "## 保留集统计",
        "",
        f"- repo 内实例数：min={rd.get('min')} / median={rd.get('median')} / max={rd.get('max')}",
        f"- problem_statement 长度：min={sl.get('min')} / p50={sl.get('p50')} / p95={sl.get('p95')} / max={sl.get('max')}",
        f"- gold 三粒度：有 files 的实例 {gs.get('n_with_files')}，有 modules 的 {gs.get('n_with_modules')}，"
        f"有 entities 的 {gs.get('n_with_entities')}，空 gold {gs.get('n_empty')}",
        "",
        f"- 输出：`{report.get('instances_parquet')}`",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M0 data preprocessing: read parquet -> filter -> extract gold -> save",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser(
        "build", help="read parquet -> filter -> extract gold -> instances.parquet + data report",
    )
    p_build.add_argument(
        "--config", default=None,
        help="config yaml path (default: mcts/config.yaml)",
    )
    p_build.add_argument(
        "--data-dir", default=None,
        help="override data dir: read {data_dir}/train.parquet + {data_dir}/validation.parquet",
    )
    p_build.add_argument(
        "--output", default=None,
        help="output dir (default: outputs/mcts)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        paths = [data_dir / "train.parquet", data_dir / "validation.parquet"]
    else:
        paths = [resolve_path(p) for p in cfg.get("data", {}).get("parquet_paths", [])]
    if not paths:
        parser.error("no parquet paths: set data.parquet_paths in config or pass --data-dir")
    output = Path(args.output) if args.output else resolve_path(cfg.get("outputs", {}).get("dir", "outputs/mcts"))
    report = build_dataset(paths, output_dir=output, cfg=cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Gold",
    "Instance",
    "extract_gold",
    "parse_repo",
    "is_python_path",
    "patch_creates_or_deletes_files",
    "read_instances",
    "instances_from_parquet",
    "filter_instances",
    "build_dataset",
    "main",
]
