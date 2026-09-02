# SPDX-License-Identifier: BSD-3-Clause

"""``mcts.report`` 数据报告测试（纯逻辑，离线可跑）。"""

import json

from mcts.report import build_report, write_report
from mcts.store import StateStore
from mcts.tasks import Budget, EnvFactory, MCTSPipeline

from mcts.tests.helpers import REARTE_R_SEQUENCES, ScriptedExecutor, make_instance


def _run_small_pipeline(workdir) -> dict:
    exe = ScriptedExecutor(REARTE_R_SEQUENCES)
    pipeline = MCTSPipeline(
        [make_instance()],
        executor_factory=lambda ef: exe,
        env_factory=EnvFactory(creation_concurrency=2),
        out_dir=workdir,
        max_concurrency=4,
        n_rollouts=5,
        max_iterations=4,
        resume=False,
        seed=1,
        stats_interval=999.0,
    )
    import asyncio

    return asyncio.run(pipeline.run())


def test_report_structure_and_content(workdir):
    summary = _run_small_pipeline(workdir)
    report = build_report(workdir, summary=summary, config={"seed": 1, "trees": 1})
    assert report["db_counts"]["rollouts"] == 35
    assert report["db_counts"]["annotations"] > 0
    assert report["statuses"] == {"done": 1}
    rs = report["rollout_stats"]
    assert rs["total"] == 35 and rs["failed"] == 0
    assert 0.0 < rs["avg_reward"] <= 1.0
    as_ = report["annotation_stats"]
    assert as_["best"] >= 4 and as_["leaf"] == 2 and as_["add"] == 3
    assert report["per_instance"][0]["instance_id"] == "inst1"
    assert report["per_instance"][0]["annotations"]["leaf"] == 2
    assert report["pipeline"]["throughput_rollouts_min"] is not None


def test_report_only_mode(workdir):
    """--report-only：只读现有库生成报告（不跑 rollout）。"""
    _run_small_pipeline(workdir)
    report = build_report(workdir, config={"seed": 1, "trees": 1})
    assert report["pipeline"] == {}
    assert report["db_counts"]["rollouts"] == 35


def test_write_report_files(workdir):
    _run_small_pipeline(workdir)
    report = build_report(workdir, config={"trees": 1})
    json_path = write_report(workdir, report)
    assert json_path.name == "rollout_report.json"
    assert (workdir / "rollout_report.md").is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["db_counts"]["rollouts"] == 35
    md = (workdir / "rollout_report.md").read_text(encoding="utf-8")
    assert "rollout 总数" in md and "inst1" in md
