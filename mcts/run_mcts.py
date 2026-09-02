# SPDX-License-Identifier: BSD-3-Clause

"""MCTS 高并发 Rollout 管道入口（PLAN §2.3–2.4 / 里程碑 M1–M3）。

用法（在项目根目录、CodeAgentRL conda 环境执行）::

    # 生成池前 10 个实例冒烟（本地 vLLM gemma-4）
    python -m mcts.run_mcts --sample 10 --seed 42 \\
        --model openai/gemma-4 --base-url http://localhost:8010/v1 \\
        --concurrency 40 --create-concurrency 8

    # 断点续跑（复用已完成 rollout，跳过 done 实例）
    python -m mcts.run_mcts --sample 500 --resume --concurrency 8

    # 预算熔断：全局限 200 次 rollout
    python -m mcts.run_mcts --sample 200 --max-rollouts 200

    # 管道自检（不碰 Docker/LLM，用脚本化 FakeExecutor 跑通并发 + 落盘 + 续跑）
    python -m mcts.run_mcts --sample 5 --dry-run

数据源：``outputs/mcts/instances.parquet``（M0 产物）+ ``outputs/mcts/splits.parquet``
（仅取 ``gen_pool=True`` 的实例，PLAN §1.3）；``--instances`` 前 N 个与
``--sample N --seed`` 随机采样两种取法。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from mcts.config import load_config, resolve_path
from mcts.instances import instances_from_parquet

logger = logging.getLogger("mcts.run_mcts")


# ---------------------------------------------------------------------------
# 实例选择
# ---------------------------------------------------------------------------


def select_instances(
    instances_parquet: str | Path,
    splits_parquet: str | Path,
    *,
    count: Optional[int] = None,
    instance_ids: Optional[list[str]] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    gen_pool_only: bool = True,
) -> list:
    """从 M0 产物选择实例：gen_pool 过滤 → 指定 ids / 前 N / 随机采样。"""
    instances = instances_from_parquet(instances_parquet)
    by_id = {i.instance_id: i for i in instances}

    gen_pool: set[str] = set()
    if gen_pool_only:
        import pandas as pd

        df = pd.read_parquet(splits_parquet)
        gen_pool = set(df.loc[df["gen_pool"], "instance_id"].tolist())
        instances = [i for i in instances if i.instance_id in gen_pool]

    if instance_ids:
        missing = [iid for iid in instance_ids if iid not in by_id]
        if missing:
            raise ValueError(f"unknown instance ids: {missing[:5]}...")
        return [by_id[iid] for iid in instance_ids]

    if count is not None:
        return instances[:count]

    if sample is not None:
        rng = random.Random(seed)
        pool = list(instances)
        rng.shuffle(pool)
        return pool[:sample]

    return instances


# ---------------------------------------------------------------------------
# dry-run FakeExecutor（管道自检用：不碰 Docker / LLM）
# ---------------------------------------------------------------------------


class FakeRolloutExecutor:
    """脚本化 rollout 执行器：每个节点产出 ``n_rollouts`` 次结果，correct 由
    节点 key 的确定性哈希决定（可复现），步数随机 1..step_limit。"""

    def __init__(
        self,
        env_factory,
        *,
        n_rollouts: int = 5,
        step_limit: int = 8,
        correct_rate: float = 0.5,
        fail_rate: float = 0.0,
        delay: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.n_rollouts = n_rollouts
        self.step_limit = step_limit
        self.correct_rate = correct_rate
        self.fail_rate = fail_rate
        self.delay = delay
        self._rng = random.Random(seed)
        self.n_calls = 0

    def run(self, task) -> object:
        from mcts.tasks import RolloutResult

        if self.delay:
            time.sleep(self.delay)
        self.n_calls += 1
        from mcts.steps import Step

        n_steps = self._rng.randint(1, self.step_limit)
        steps = [
            Step(assistant={"role": "assistant", "content": f"step {j} of {task.node_key}",
                            "extra": {"actions": [{"command": f"echo {j}",
                                                   "tool_call_id": f"t{j}"}]}})
            for j in range(n_steps)
        ]
        correct = self._rng.random() < self.correct_rate
        error = self._rng.random() < self.fail_rate
        return RolloutResult(
            instance_id=task.instance_id, node_key=task.node_key,
            rollout_idx=task.rollout_idx,
            reward=1.0 if correct else 0.0, correct=correct,
            steps=steps, exit_status="Submitted" if correct else "LimitsExceeded",
            n_calls=n_steps, cost=0.0, duration=0.01,
            error=f"fake failure" if error else None,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MCTS high-concurrency rollout pipeline (M1-M3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="mcts/config.yaml path")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--instances", type=int, default=None, metavar="N",
                   help="取生成池前 N 个实例（按 M0 顺序）")
    g.add_argument("--instance-ids", default=None, metavar="ID1,ID2",
                   help="显式指定 instance_id 列表")
    g.add_argument("--sample", type=int, default=None, metavar="N",
                   help="从生成池随机采样 N 个（--seed 固定可复现）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-pool-only", action="store_true", default=True,
                   help="只取生成池实例（默认开启）")
    p.add_argument("--no-gen-pool", dest="gen_pool_only", action="store_false")

    p.add_argument("--resume", action="store_true", default=False,
                   help="断点续跑：跳过 done 实例、复用已完成节点 rollout")
    p.add_argument("--dry-run", action="store_true",
                   help="管道自检：用脚本化 FakeExecutor，不碰 Docker/LLM")
    p.add_argument("--report-only", action="store_true",
                   help="只从现有 state.db 生成数据报告（不跑 rollout）")
    p.add_argument("--no-report", action="store_true",
                   help="运行结束后不生成数据报告（默认生成 rollout_report.json/md）")
    p.add_argument("--fresh", action="store_true",
                   help="备份旧 state.db（state.db.bak.<ts>）后重建 v5 schema（旧库不自动迁移）")

    p.add_argument("--model", default=None, help="模型名（需 provider 前缀，如 openai/gemma-4）")
    p.add_argument("--base-url", default=None, help="OpenAI 兼容服务地址（本地 vLLM）")
    p.add_argument("--api-key", default=None)
    p.add_argument("--concurrency", type=int, default=8, help="全局 rollout 并发")
    p.add_argument("--create-concurrency", type=int, default=8,
                   help="容器**并发创建数**上限（只限同时创建的容器数，不限总创建次数）")
    p.add_argument("--n-rollouts", type=int, default=5, help="每节点 rollout 次数 N")
    p.add_argument("--step-limit", type=int, default=20, help="agent 回合上限（默认 20）")
    p.add_argument("--max-iterations", type=int, default=20, help="标注循环轮数上限")
    p.add_argument("--reward-threshold", type=float, default=None,
                   help="correct 阈值 τ（默认按 mcts/config.yaml mcts.reward.threshold=0.6）")
    p.add_argument("--magic-submit", action="store_true", default=None,
                   help="恢复 bash 魔法串提交（默认按 mcts/config.yaml submit.magic_submit）")
    p.add_argument("--no-magic-submit", dest="magic_submit", action="store_false",
                   help="禁用 bash 魔法串提交（仅 submit_locations 工具提交）")
    p.add_argument("--max-rollouts", type=int, default=0, help="全局 rollout 预算（0=不限）")
    p.add_argument("--max-rollouts-per-instance", type=int, default=0)
    p.add_argument("--max-llm-calls", type=int, default=0, help="全局 LLM 调用软预算")
    p.add_argument("--temperature-range", default="0.7,1.0", help="rollout 采样温度范围")
    p.add_argument("--output", default=None, help="输出目录（默认 outputs/mcts）")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--phoenix-tracing", action="store_true",
                   help="开启 Phoenix 追踪（假定 phoenix serve 已启动）")
    p.add_argument("--phoenix-project", default=None)
    p.add_argument("--phoenix-endpoint", default=None)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # .env 必须在任何 agent/minisweagent 导入之前加载（README §4：模块级默认
    # EnvManager 在 import 时固化 CODEAGENTRL_IMAGE 等环境变量）。
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv 缺失时退化为环境变量直供
        pass

    cfg = load_config(args.config)
    out_dir = Path(args.output) if args.output else resolve_path(
        cfg.get("outputs", {}).get("dir", "outputs/mcts"))
    instances_parquet = out_dir / "instances.parquet"
    splits_parquet = out_dir / "splits.parquet"
    if not instances_parquet.is_file():
        raise FileNotFoundError(
            f"instances.parquet not found: {instances_parquet} — run `python -m mcts.instances build` first")

    report_config = {
        "seed": args.seed,
        "trees": args.sample if args.sample is not None else args.instances,
        "n_rollouts": args.n_rollouts,
        "concurrency": args.concurrency,
        "create_concurrency": args.create_concurrency,
        "step_limit": args.step_limit,
        "max_iterations": args.max_iterations,
        "model": args.model,
        "resume": args.resume,
    }

    # --fresh / 旧库预检：v5 三表 schema 与旧 v2 库不兼容（施工文件 00 §6）
    db_path = out_dir / "state.db"
    if db_path.exists():
        if args.fresh:
            import shutil
            bak = db_path.with_name(f"state.db.bak.{int(time.time())}")
            shutil.copy2(db_path, bak)
            db_path.unlink()
            logger.info("fresh: 旧库已备份到 %s，将重建 v5 schema", bak)
        else:
            try:
                from mcts.store import LegacySchemaError, StateStore
                _probe = StateStore(db_path)
                _probe.close()
            except LegacySchemaError as e:
                logger.error("%s", e)
                print("[error] 旧版 state.db 与 v5 不兼容，请加 --fresh（自动备份后重建）。")
                return 1

    # --report-only：只读现有 state.db 生成报告，不跑 rollout
    if args.report_only:
        from mcts.report import build_report, write_report

        report = build_report(out_dir, config=report_config)
        path = write_report(out_dir, report)
        print(f"[report] 已生成: {path}（report-only，未运行 rollout）")
        return 0

    instance_ids = None
    if args.instance_ids:
        instance_ids = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    instances = select_instances(
        instances_parquet, splits_parquet,
        count=args.instances, instance_ids=instance_ids, sample=args.sample,
        seed=args.seed, gen_pool_only=args.gen_pool_only,
    )
    if not instances:
        raise SystemExit("no instances selected")
    logger.info("selected %d instances (%d repos)", len(instances),
                len({i.repo for i in instances}))

    temp_range = tuple(float(x) for x in args.temperature_range.split(","))
    if len(temp_range) != 2 or not (0.0 <= temp_range[0] <= temp_range[1] <= 2.0):
        raise ValueError(f"invalid --temperature-range {args.temperature_range!r}")

    from mcts.tasks import Budget, EnvFactory, MCTSPipeline

    env_factory = EnvFactory(creation_concurrency=args.create_concurrency)

    def make_executor(ef):
        if args.dry_run:
            return FakeRolloutExecutor(ef, n_rollouts=args.n_rollouts,
                                       step_limit=args.step_limit, seed=args.seed)
        from mcts.executor import AgentRolloutExecutor

        phoenix_tracing = None
        if args.phoenix_tracing:
            phoenix_tracing = {
                "project_name": args.phoenix_project,
                "endpoint": args.phoenix_endpoint,
            }
        # 提交协议配置（PLAN §2.2）：默认从 mcts/config.yaml mcts.submit 读，
        # 命令行 --magic-submit/--no-magic-submit 可覆盖魔法串开关。
        submit_cfg = cfg.get("mcts", {}).get("submit", {}) or {}
        magic_submit = submit_cfg.get("magic_submit", False)
        if args.magic_submit is not None:
            magic_submit = args.magic_submit
        # 奖励配置（docs/reward_design.md）：mcts.reward_* / mcts.reward 段
        reward_cfg = cfg.get("mcts", {}).get("reward", {}) or {}
        depth_weights = reward_cfg.get("depth_weights")
        return AgentRolloutExecutor(
            ef,
            model_name=args.model or "openai/gemma-4",
            base_url=args.base_url,
            api_key=args.api_key,
            config_file=cfg.get("mcts", {}).get("config_file", "config/mini_submit.yaml"),
            step_limit=args.step_limit,
            reward_threshold=(args.reward_threshold
                              if args.reward_threshold is not None
                              else reward_cfg.get("threshold", 0.6)),
            reward_mode=reward_cfg.get("mode", "layered"),
            depth_weights=depth_weights,
            strict_multi_gate=reward_cfg.get("strict_multi_gate", True),
            missing_level_norm=reward_cfg.get("missing_level_norm", True),
            overpromise_penalty=reward_cfg.get("overpromise_penalty", 0.0),
            duplicate_penalty=reward_cfg.get("duplicate_penalty", 0.0),
            temperature_range=temp_range,
            workdir=cfg.get("env", {}).get("workdir", "/repo"),
            phoenix_tracing=phoenix_tracing,
            magic_submit=magic_submit,
            model_class=submit_cfg.get(
                "model_class", "agent.submit_model.SubmitLocationsModel"),
            agent_class=submit_cfg.get(
                "agent_class", "agent.submit_agent.SubmitAgent"),
        )

    pipeline = MCTSPipeline(
        instances,
        executor_factory=make_executor,
        env_factory=env_factory,
        out_dir=out_dir,
        max_concurrency=args.concurrency,
        n_rollouts=args.n_rollouts,
        temperature_range=temp_range,
        max_iterations=args.max_iterations,
        budget=Budget(
            max_rollouts=args.max_rollouts,
            max_rollouts_per_instance=args.max_rollouts_per_instance,
            max_llm_calls=args.max_llm_calls,
        ),
        resume=args.resume,
        seed=args.seed,
        config_file=cfg.get("mcts", {}).get("config_file", "config/mini_submit.yaml"),
    )

    summary = asyncio.run(pipeline.run())
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.no_report:
        from mcts.report import build_report, write_report

        report = build_report(out_dir, summary=summary, config=report_config)
        path = write_report(out_dir, report)
        print(f"[report] 数据报告已生成: {path}（+ rollout_report.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
