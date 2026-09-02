#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""批量拉取数据集中所有仓库到本地 zip 缓存（训练 / rollout 前的网络预热）。

背景：rollout 时 ``EnvManager.get_env`` 若未命中 zip 缓存，会在容器内实时
``git clone https://github.com/<repo>.git`` —— A800 上 github 网络不稳时这一步
经常失败 / 长时间挂起。本脚本在训练前把数据集中出现的**全部仓库**一次性拉取到
宿主机 zip 缓存（与 ``python -m agent.init_env cache <repo>`` 同一实现：
``EnvManager.populate_repo_cache``，完整 git 克隆含 ``.git`` 历史），之后
rollout 全部走 docker cp + 解压的离线路径，不再依赖实时外网。

数据来源（优先级从高到低）：

1. ``--repos a/b,c/d``：显式仓库列表（逗号分隔，跳过数据加载）；
2. ``--data-dir <dir>``：原始 SWE-Smith parquet（读 ``{dir}/train.parquet``
   + ``{dir}/validation.parquet``，不依赖 M0 输出）；
3. ``--instances <parquet>``（默认 ``outputs/mcts/instances.parquet``）：
   M0 产出的实例表（含全部 39,284 条过滤后实例 / 131 repos）。

仓库去重；SWE-Smith 快照仓库 ``base_commit=None``，拉取保持 HEAD（单 commit，
与 ``mcts/env.py`` 的"无 commit 不切换"一致）；若实例带 ``base_commit`` 则
打包前 checkout 该 commit。

用法（项目根目录，CodeAgentRL 环境）::

    # 1) 预生成 M0 实例表（可选，已有则跳过）
    python -m mcts.instances build

    # 2) 批量拉取全部仓库（缓存目录也可用环境变量 CODEAGENTRL_REPO_CACHE_DIR）
    python scripts/batch_repo_pull.py --cache-dir /data/repo_cache

    # 3) 常用选项：并发 / 重试 / 只列不拉 / 只拉前 N 个 / 保存报告
    python scripts/batch_repo_pull.py --cache-dir /data/repo_cache --workers 4 --retries 2
    python scripts/batch_repo_pull.py --dry-run
    python scripts/batch_repo_pull.py --limit 10
    python scripts/batch_repo_pull.py --report outputs/mcts/repo_cache_report.json

    # 4) 直接从原始 parquet 拉取（不依赖 M0 输出）
    python scripts/batch_repo_pull.py --data-dir ref_papers/codescout/data/swe_smith

    # 5) 显式仓库列表
    python scripts/batch_repo_pull.py --repos swesmith/a__b.1234abcd,swesmith/c__d.5678ef90

幂等：已存在的 zip 直接跳过（``populate_repo_cache`` 内部也有双重检查）；
失败仓库自动重试（``--retries``）并记录，其余仓库继续；任一仓库最终失败则
退出码为 1（便于脚本 / CI 判断），全部成功退出码为 0。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 保证 ``agent`` / ``mcts`` 包可导入（无论从项目根还是其它目录启动）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 必须先于 ``from agent import ...`` 加载 .env：agent.init_env 的模块级默认
# EnvConfig 在 import 时就会读取 CODEAGENTRL_REPO_CACHE_DIR 等环境变量。
load_dotenv()

from agent.init_env import EnvConfig, EnvManager, _sanitize  # noqa: E402

logger = logging.getLogger("batch_repo_pull")

DEFAULT_INSTANCES = PROJECT_ROOT / "outputs" / "mcts" / "instances.parquet"
DEFAULT_DATA_DIR = PROJECT_ROOT / "ref_papers" / "codescout" / "data" / "swe_smith"
DEFAULT_CACHE_REPORT = PROJECT_ROOT / "outputs" / "mcts" / "repo_cache_report.json"


# ---------------------------------------------------------------------------
# 数据加载：{repo: commit_or_None}（去重）
# ---------------------------------------------------------------------------


def load_repos(
    *,
    repos_arg: Optional[str] = None,
    instances_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> tuple[dict[str, Optional[str]], str]:
    """加载仓库集合，返回 ``({repo: commit}, 数据来源描述)``。

    - ``repos_arg``：逗号分隔的显式仓库列表（commit 一律 None）；
    - ``instances_path``：M0 实例表（``mcts.instances.instances_from_parquet``）；
    - ``data_dir``：原始 SWE-Smith parquet（``mcts.instances.read_instances``）。
    """
    if repos_arg:
        repos = {r.strip(): None for r in repos_arg.split(",") if r.strip()}
        return repos, f"--repos 显式列表（{len(repos)} 个）"

    if data_dir is not None and Path(data_dir).is_dir():
        from mcts.instances import read_instances

        insts = []
        for split in ("train", "validation"):
            p = Path(data_dir) / f"{split}.parquet"
            if p.is_file():
                insts.extend(read_instances(p))
        source = f"{data_dir}（train/validation，{len(insts)} 实例）"
    elif instances_path is not None and instances_path.is_file():
        from mcts.instances import instances_from_parquet

        insts = instances_from_parquet(instances_path)
        source = f"{instances_path}（{len(insts)} 实例）"
    else:
        raise FileNotFoundError(
            f"未找到可用数据：{instances_path} 不存在且 {data_dir} 不是目录。\n"
            f"请先运行 `python -m mcts.instances build` 生成实例表，或用 "
            f"`--data-dir ref_papers/codescout/data/swe_smith` 读原始 parquet，"
            f"或直接用 `--repos a/b,c/d` 指定仓库列表。"
        )

    repos: dict[str, Optional[str]] = {}
    for inst in insts:
        # 同一 repo 取首个非 None 的 base_commit（SWE-Smith 恒为 None -> 保持 HEAD）
        commit = repos.get(inst.repo)
        if commit is None and inst.base_commit:
            repos[inst.repo] = inst.base_commit
        else:
            repos.setdefault(inst.repo, None)
    return repos, source


# ---------------------------------------------------------------------------
# 拉取
# ---------------------------------------------------------------------------


def _expected_zip(cache_dir: str, repo: str) -> Path:
    return Path(cache_dir) / f"{_sanitize(repo)}.zip"


def pull_one(
    mgr: EnvManager, repo: str, commit: Optional[str], retries: int
) -> tuple[str, Optional[str], Optional[str]]:
    """拉取单个仓库，返回 ``(status, zip_path, error)``，status ∈ pulled/skipped/failed。"""
    zip_path = _expected_zip(mgr.config.repo_cache_dir or "", repo)
    if zip_path.is_file():
        return "skipped", str(zip_path), None
    last_err: Optional[str] = None
    for attempt in range(1, retries + 2):  # retries=0 时也尝试 1 次
        try:
            path = mgr.populate_repo_cache(repo, commit)
            return "pulled", path, None
        except Exception as e:  # 网络 / git 错误统一按失败处理并重试
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("repo %s 第 %d/%d 次尝试失败: %s",
                           repo, attempt, retries + 1, last_err)
            if attempt <= retries:
                time.sleep(2 * attempt)
    return "failed", None, last_err


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量拉取数据集中所有仓库到本地 zip 缓存（rollout 网络预热）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instances", type=Path, default=DEFAULT_INSTANCES,
        help="M0 实例表 parquet（instances.parquet）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="原始 SWE-Smith parquet 目录（读 train/validation.parquet，优先于 --instances）",
    )
    parser.add_argument(
        "--repos", default=None,
        help="逗号分隔的显式仓库列表（如 swesmith/a__b.1234abcd,swesmith/c__d.5678ef90；"
             "优先级最高，跳过数据加载）",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="repo zip 缓存目录（默认读环境变量 CODEAGENTRL_REPO_CACHE_DIR）",
    )
    parser.add_argument("--workers", type=int, default=4, help="并发拉取数")
    parser.add_argument("--retries", type=int, default=2, help="单个仓库失败重试次数")
    parser.add_argument(
        "--clone-timeout", type=int, default=None,
        help="单次 git clone 超时秒数（默认取 EnvConfig.clone_timeout=600）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个仓库（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只列出仓库，不实际拉取")
    parser.add_argument("--report", type=Path, default=DEFAULT_CACHE_REPORT,
                        help="报告 JSON 保存路径（--no-report 可关闭）")
    parser.add_argument("--no-report", dest="report", action="store_const", const=None,
                        help="不保存报告文件")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    repos, source = load_repos(
        repos_arg=args.repos,
        instances_path=args.instances,
        data_dir=args.data_dir,
    )
    repo_list = sorted(repos)
    if args.limit:
        repo_list = repo_list[: args.limit]
    print(f"[repo-pull] 数据来源: {source}")
    print(f"[repo-pull] 唯一仓库 {len(repos)} 个（本次处理 {len(repo_list)} 个）")

    if args.dry_run:
        for repo in repo_list:
            print(f"  - {repo}@{repos[repo] or 'HEAD'}")
        return 0

    cache_dir = args.cache_dir or EnvConfig().repo_cache_dir
    if not cache_dir:
        parser.error(
            "未指定缓存目录：用 --cache-dir 或设置环境变量 CODEAGENTRL_REPO_CACHE_DIR"
        )
    mgr = EnvManager(EnvConfig(repo_cache_dir=cache_dir, clone_timeout=args.clone_timeout or 600))

    stats = {"pulled": 0, "skipped": 0, "failed": 0}
    failures: dict[str, str] = {}
    t0 = time.time()
    if args.workers > 1 and len(repo_list) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(pull_one, mgr, repo, repos[repo], args.retries): repo
                for repo in repo_list
            }
            for fut in as_completed(futures):
                repo = futures[fut]
                status, path, err = fut.result()
                stats[status] += 1
                if status == "failed":
                    failures[repo] = err or "unknown"
                    print(f"  ✗ {repo}: {err}")
                else:
                    print(f"  ✓ {repo} [{status}] -> {path}")
    else:
        for repo in repo_list:
            status, path, err = pull_one(mgr, repo, repos[repo], args.retries)
            stats[status] += 1
            if status == "failed":
                failures[repo] = err or "unknown"
                print(f"  ✗ {repo}: {err}")
            else:
                print(f"  ✓ {repo} [{status}] -> {path}")
    elapsed = time.time() - t0

    report = {
        "source": source,
        "cache_dir": cache_dir,
        "total_repos": len(repos),
        "processed_repos": len(repo_list),
        "pulled": stats["pulled"],
        "skipped": stats["skipped"],
        "failed": stats["failed"],
        "failures": failures,
        "elapsed_s": round(elapsed, 1),
    }
    print("=" * 60)
    print(f"[repo-pull] 完成: 新增 {stats['pulled']} / 已缓存 {stats['skipped']} / "
          f"失败 {stats['failed']}，耗时 {elapsed:.1f}s")
    if failures:
        print("[repo-pull] 失败清单（可重跑本脚本续拉，已成功的会跳过）:")
        for repo, err in failures.items():
            print(f"  - {repo}: {err}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[repo-pull] 报告已保存: {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
