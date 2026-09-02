# SPDX-License-Identifier: BSD-3-Clause

"""PRM 数据三层划分（PLAN §1.3 / D6 repo 级防泄漏）。

1. **实例层（rollout 执行范围）**：按 repo 分组后取 60% repo（``gen_pool_ratio``，
   固定 seed）的实例作为**数据生成池**（阶段 1 先采样 500–2000 实例跑通全链路）；
2. **repo 级防泄漏划分（PRM 数据集）**：在数据生成池的 repo 上按
   ``PRM-train : PRM-dev : PRM-test = 80% : 10% : 10%`` 分组（**同 repo 不跨 split**，
   对齐 CodeScout §4.1 的 128 repos 无重叠原则）；只有生成池实例会产生 MCTS 标注，
   因此 PRM 划分只作用于生成池 repo；
3. **步级样本划分**：MCTS 标注展开成步样本后按 ``instance_id`` 分桶 ——
   本模块输出的 ``instance_id → split`` 映射即分桶依据（同实例的步全部落入同一 split）。

输出（``outputs/mcts/``）：``splits.parquet``（instance_id → split 映射）+
``splits_report.json`` / ``splits_report.md``。随机种子固定（config ``split.seed``），
划分可复现。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from mcts.config import load_config, resolve_path
from mcts.instances import Instance, instances_from_parquet

logger = logging.getLogger("mcts.splits")

SPLIT_COLUMNS = ["instance_id", "repo", "gen_pool", "prm_split"]


def make_splits(
    instances: list[Instance],
    *,
    seed: int = 42,
    gen_pool_ratio: float = 0.6,
    prm_train_ratio: float = 0.8,
    prm_dev_ratio: float = 0.1,
    prm_test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """对实例列表做三层划分，返回 ``(splits DataFrame, 报告 dict)``。

    ``prm_split`` 取值 ``train`` / ``dev`` / ``test``（生成池 repo）或 ``None``
    （不在生成池，不产生 MCTS 标注）。同一 repo 的所有实例落入同一 split。
    """
    if not 0.0 <= gen_pool_ratio <= 1.0:
        raise ValueError(f"gen_pool_ratio must be in [0, 1], got {gen_pool_ratio!r}")
    if not 0.0 <= prm_dev_ratio <= 1.0 or not 0.0 <= prm_test_ratio <= 1.0:
        raise ValueError("prm dev/test ratios must be in [0, 1]")

    repos = sorted({i.repo for i in instances})
    rng = random.Random(seed)
    rng.shuffle(repos)
    n_gen = int(round(len(repos) * gen_pool_ratio))
    pool_repos = repos[:n_gen]

    # PRM 80/10/10（按 repo 分组）；dev/test 至少 1 个 repo（数量允许时）
    n = len(pool_repos)
    n_dev = int(round(n * prm_dev_ratio))
    n_test = int(round(n * prm_test_ratio))
    if n >= 3:
        n_dev = max(1, n_dev)
        n_test = max(1, n_test)
        if n_dev + n_test > n - 1:
            n_dev, n_test = 1, 1
    n_train = n - n_dev - n_test
    if n_train < 0:  # 数量不足以分三份时全部归 train
        n_train, n_dev, n_test = n, 0, 0
    split_of_repo: dict[str, str] = {}
    for i, repo in enumerate(pool_repos):
        if i < n_train:
            split_of_repo[repo] = "train"
        elif i < n_train + n_dev:
            split_of_repo[repo] = "dev"
        else:
            split_of_repo[repo] = "test"
    pool_set = set(pool_repos)

    rows = [
        {
            "instance_id": i.instance_id,
            "repo": i.repo,
            "gen_pool": i.repo in pool_set,
            "prm_split": split_of_repo.get(i.repo),  # 非生成池 repo -> None
        }
        for i in instances
    ]
    df = pd.DataFrame(rows, columns=SPLIT_COLUMNS)

    def _n(mask: pd.Series) -> int:
        return int(mask.sum())

    report = {
        "seed": seed,
        "n_instances": len(df),
        "n_repos": len(repos),
        "gen_pool_repos": n_gen,
        "gen_pool_instances": _n(df["gen_pool"]),
        "prm_train_repos": n_train,
        "prm_train_instances": _n(df["prm_split"] == "train"),
        "prm_dev_repos": n_dev,
        "prm_dev_instances": _n(df["prm_split"] == "dev"),
        "prm_test_repos": n_test,
        "prm_test_instances": _n(df["prm_split"] == "test"),
        "excluded_instances": _n(~df["gen_pool"]),
    }
    return df, report


def _render_report_md(report: dict) -> str:
    lines = [
        "# M0 三层划分报告（PLAN §1.3）",
        "",
        f"- 种子：``{report['seed']}``（固定，划分可复现）",
        f"- 实例数：**{report['n_instances']}** / repo 数：**{report['n_repos']}**",
        "",
        "## 实例层（数据生成池 = 60% repo）",
        "",
        f"- 生成池 repo：**{report['gen_pool_repos']}** / 实例：**{report['gen_pool_instances']}**",
        f"- 池外（暂不 rollout）：{report['excluded_instances']} 实例",
        "",
        "## repo 级防泄漏划分（PRM 数据集，80/10/10，同 repo 不跨 split）",
        "",
        "| split | repos | instances |",
        "| --- | ---: | ---: |",
        f"| train | {report['prm_train_repos']} | {report['prm_train_instances']} |",
        f"| dev | {report['prm_dev_repos']} | {report['prm_dev_instances']} |",
        f"| test | {report['prm_test_repos']} | {report['prm_test_instances']} |",
        "",
        "> 步级样本划分（§1.3.3）：MCTS 标注展开成步样本后按 instance_id 分桶，",
        "> 同实例的步全部落入同一 split（本映射即分桶依据）。",
        "",
    ]
    return "\n".join(lines)


def build_splits(
    instances_parquet: str | Path,
    *,
    output_dir: str | Path,
    cfg: Optional[dict] = None,
) -> dict:
    """读取 ``instances.parquet`` -> 三层划分 -> 写 ``splits.parquet`` + 报告。"""
    cfg = cfg or {}
    sc = cfg.get("split", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_from_parquet(instances_parquet)
    df, report = make_splits(
        instances,
        seed=int(sc.get("seed", 42)),
        gen_pool_ratio=float(sc.get("gen_pool_ratio", 0.6)),
        prm_train_ratio=float(sc.get("prm_train_ratio", 0.8)),
        prm_dev_ratio=float(sc.get("prm_dev_ratio", 0.1)),
        prm_test_ratio=float(sc.get("prm_test_ratio", 0.1)),
    )
    splits_path = output_dir / "splits.parquet"
    df.to_parquet(splits_path, index=False)
    report["splits_parquet"] = str(splits_path)
    (output_dir / "splits_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "splits_report.md").write_text(_render_report_md(report), encoding="utf-8")
    logger.info(
        "build_splits: %d instances / %d repos -> gen_pool %d repos; %s",
        report["n_instances"], report["n_repos"], report["gen_pool_repos"], splits_path,
    )
    return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M0 three-layer split: instances.parquet -> splits.parquet (no repo leakage)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser(
        "build", help="read instances.parquet -> three-layer split -> splits.parquet + report",
    )
    p_build.add_argument("--config", default=None, help="config yaml path")
    p_build.add_argument("--instances", default=None, help="instances.parquet path")
    p_build.add_argument("--output", default=None, help="output dir (default: outputs/mcts)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out_dir = Path(args.output) if args.output else resolve_path(cfg.get("outputs", {}).get("dir", "outputs/mcts"))
    inst_path = (
        Path(args.instances) if args.instances else out_dir / "instances.parquet"
    )
    report = build_splits(inst_path, output_dir=out_dir, cfg=cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["make_splits", "build_splits", "SPLIT_COLUMNS", "main"]
