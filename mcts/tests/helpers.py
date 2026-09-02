# SPDX-License-Identifier: BSD-3-Clause

"""MCTS 引擎测试共享设施：实例/步构造 + 脚本化 FakeExecutor。

测试为**纯逻辑**（不依赖 Docker / LLM / agent / minisweagent）：引擎的
``executor_factory`` 注入点用脚本化 executor 复现 ReARTeR docs/01 §11 的
数值示例与并发语义。
"""

import sys
from pathlib import Path

# 保证 ``mcts`` 包可导入（无论从 mcts/tests 还是项目根启动）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcts.instances import Gold, Instance  # noqa: E402
from mcts.steps import Step  # noqa: E402

SAMPLE_PATCH = (
    "diff --git a/x.py b/x.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/x.py\n"
    "+++ b/x.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def foo():\n"
    "-    return 1\n"
    "+    return 2\n"
)


def make_step(content: str, commands: list[str] | None = None) -> Step:
    """构造一步：assistant 消息（content + actions），无 tail。"""
    actions = [
        {"command": c, "tool_call_id": f"t{i}"} for i, c in enumerate(commands or [])
    ]
    return Step(assistant={
        "role": "assistant", "content": content,
        "extra": {"actions": actions},
    })


def make_steps(contents: list[str]) -> list[Step]:
    return [make_step(c) for c in contents]


def make_instance(
    instance_id: str = "inst1",
    repo: str = "swesmith/owner__repo.abc12345",
    gold_files: frozenset | None = None,
    problem_statement: str = "Fix the bug in x.py",
) -> Instance:
    """构造最小 Instance（gold 默认命中 x.py 的 file 粒度）。"""
    return Instance(
        instance_id=instance_id,
        repo=repo,
        owner="swesmith",
        name="owner__repo",
        commit8="abc12345",
        base_commit=None,
        problem_statement=problem_statement,
        patch=SAMPLE_PATCH,
        use_patch=True,
        gold=Gold(files=gold_files if gold_files is not None else frozenset({"x.py"}),
                  modules=frozenset({"x.py:foo"}),
                  entities=frozenset({"x.py:foo"})),
        source="test",
    )


class ScriptedExecutor:
    """按节点**首次出现顺序**脚本化 rollout 结果（内容寻址节点 key）。

    ``sequences``：``[(flags, lens), ...]`` —— flags 为该节点各 rollout 的
    correct 标记（0/1），lens 为各 rollout 的续跑步数；同一节点被再次访问时
    复用首次分配的脚本（复现 ReARTeR 的确定性示例）。

    rollout 的续跑步 content 含 ``{node_key}#{rollout_idx}#{j}``，保证前缀
    内容寻址稳定（不同节点/不同 rollout 生成的步内容互不相同）。
    """

    def __init__(self, sequences: list[tuple[list[int], list[int]]], env_factory=None,
                 include_trajectory: bool = True):
        self.sequences = list(sequences)
        self.by_key: dict[str, tuple[list[int], list[int]]] = {}
        self.executed: list[tuple[str, int]] = []   # (node_key, rollout_idx) 执行序
        self.env_factory = env_factory
        # True：rollout 带最小轨迹（system+user），TreeDriver 可算出 probe 前缀
        # head（默认行为与既有测试一致）；False：模拟历史实例 head 缺失场景。
        self.include_trajectory = include_trajectory

    def run(self, task) -> "RolloutResult":
        from mcts.tasks import RolloutResult

        key = task.node_key
        if key not in self.by_key:
            if len(self.by_key) < len(self.sequences):
                self.by_key[key] = self.sequences[len(self.by_key)]
            else:
                # 超出脚本范围的节点：防御性默认（全对、1 步 → MC=1，树搜索快速收敛）
                self.by_key[key] = ([1] * 5, [1] * 5)
        flags, lens = self.by_key[key]
        idx = task.rollout_idx % len(flags)
        correct = bool(flags[idx])
        n = lens[idx % len(lens)]
        steps = [make_step(f"{key}#{task.rollout_idx}#{j}") for j in range(n)]
        self.executed.append((key, task.rollout_idx))
        trajectory = None
        if self.include_trajectory:
            trajectory = {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "a", "extra": {"actions": []}},
                ],
                "trajectory_format": "mini-swe-agent-1.1",
            }
        return RolloutResult(
            instance_id=task.instance_id,
            node_key=key,
            rollout_idx=task.rollout_idx,
            reward=1.0 if correct else 0.0,
            correct=correct,
            steps=steps,
            exit_status="Submitted" if correct else "LimitsExceeded",
            n_calls=n,
            cost=0.0,
            duration=0.001,
            trajectory=trajectory,
        )


# ReARTeR docs/01 §11.1–11.6 的脚本化结果（按节点首次出现顺序）：
# root → n1 → n2 → n3 → n4 → n5 → n6
REARTE_R_SEQUENCES = [
    # root：t1..t5 = [对,错,对,错,错]，步数 [5,6,4,4,5] → MC=0.4
    ([1, 0, 1, 0, 0], [5, 6, 4, 4, 5]),
    # n1（前缀 [s3a,s3b]）：[1,1,1,1,0]，步数 [3,4,2,3,4] → MC=0.8
    ([1, 1, 1, 1, 0], [3, 4, 2, 3, 4]),
    # n2（前缀 [s3a,s3b,s3c]）：[1,0,0,0,0]，步数全 1 → MC=0.2
    ([1, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
    # n3（前缀 [s3a,s3b,u1]）：全对 → MC=1.0（探测后停止，不扩展）
    ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
    # n4（前缀 [s3a,s3b,v1]）：全错 → MC=0.0（leaf）
    ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
    # n5（前缀 [s3a,s3b,w1]）：[1,1,0,0,1] → MC=0.6（扩展）
    ([1, 1, 0, 0, 1], [1, 1, 1, 1, 1]),
    # n6（前缀 [s3a,s3b,w1,w2]）：全错 → MC=0.0（leaf）
    ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
]
