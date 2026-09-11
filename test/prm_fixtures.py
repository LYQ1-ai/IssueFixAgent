# SPDX-License-Identifier: BSD-3-Clause

"""PRM 单测共享 fixture 工厂（迷你 v5 DB + splits.parquet，离线构造）。

构造口径与真实库一致（docs/construction/01 DDL；mcts/steps.py::Step.to_json
形态的 prefix_json；node_key 内容寻址）。仅 ``test_prm_*`` 使用。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from mcts.steps import Step, prefix_node_key

_DDL = """
CREATE TABLE IF NOT EXISTS tree_instances (
    instance_id        TEXT PRIMARY KEY,
    status             TEXT NOT NULL,
    n_rounds           INTEGER NOT NULL DEFAULT 0,
    messages_head_json TEXT,
    root_mc            REAL,
    updated_at         REAL
);
CREATE TABLE IF NOT EXISTS nodes (
    instance_id TEXT NOT NULL,
    node_key    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'rollout',
    prefix_json TEXT NOT NULL,
    mc_score    REAL,
    visits      INTEGER NOT NULL DEFAULT 0,
    in_pool     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL,
    PRIMARY KEY (instance_id, node_key)
);
CREATE TABLE IF NOT EXISTS rollouts (
    instance_id        TEXT NOT NULL,
    node_key           TEXT NOT NULL,
    rollout_idx        INTEGER NOT NULL,
    is_consumed        INTEGER NOT NULL DEFAULT 0,
    consumed_iteration INTEGER,
    result_json        TEXT NOT NULL,
    n_steps            INTEGER,
    correct            INTEGER,
    reward             REAL,
    submitted          INTEGER,
    replay_drift       INTEGER NOT NULL DEFAULT 0,
    exit_status        TEXT,
    created_at         REAL,
    PRIMARY KEY (instance_id, node_key, rollout_idx)
);
"""

_HEAD = [
    {"role": "system", "content": "You are a helpful coding agent."},
    {"role": "user", "content": "Consider the issue:\n<issue_description>\nFix the bug about widgets.\n</issue_description>\nYou have bash access."},
]


def make_step(i: int, command: str = "ls", output: str = "ok") -> dict:
    """一步 = assistant(tool_calls/reasoning/冗余字段) + tool observation tail。"""
    assistant = {
        "content": "\n\n",
        "role": "assistant",
        "tool_calls": [{
            "index": 0,
            "function": {"arguments": json.dumps({"command": command}), "name": "bash"},
            "id": f"call_{i}",
            "type": "function",
        }],
        "function_call": None,
        "reasoning_content": f"step {i} reasoning",
        "provider_specific_fields": {"refusal": None},
        "extra": {"actions": [{"command": command, "tool_call_id": f"call_{i}"}],
                  "response": {"id": f"resp_{i}"}},
    }
    tail = [{
        "content": json.dumps({"returncode": 0, "output": output}),
        "extra": {"raw_output": output},
        "tool_call_id": f"call_{i}",
        "role": "tool",
    }]
    return {"assistant": assistant, "tail": tail}


def make_steps(n: int) -> list[dict]:
    return [make_step(i) for i in range(1, n + 1)]


def add_tree(conn: sqlite3.Connection, instance_id: str, *, status: str = "done",
             head: list | None = None, root_correct: list[int] | None = None,
             nodes: dict[str, dict] | None = None) -> None:
    """注册一棵树。

    Args:
        root_correct: root rollout 的 correct 序列（如 [1,1,0,0,0]）。
        nodes: ``{node_key: {"prefix": [step dicts], "corrects": [...], "mc": 缓存,
               "visits": int, "in_pool": int}}``；node_key 为 None 时自动内容寻址。
    """
    conn.execute(
        "INSERT INTO tree_instances (instance_id, status, n_rounds, messages_head_json, root_mc) "
        "VALUES (?, ?, 0, ?, NULL)",
        (instance_id, status, json.dumps(head if head is not None else _HEAD)))
    conn.execute(
        "INSERT INTO nodes (instance_id, node_key, status, prefix_json, mc_score, visits, in_pool) "
        "VALUES (?, 'root', 'ready', '[]', NULL, 0, 1)", (instance_id,))
    for idx, correct in enumerate(root_correct or []):
        _add_rollout(conn, instance_id, "root", idx, correct, n_steps=10)
    for key, spec in (nodes or {}).items():
        prefix = spec["prefix"]
        if key is None:
            key = prefix_node_key([Step.from_json(s) for s in prefix])
        mc_cache = spec.get("mc")
        conn.execute(
            "INSERT INTO nodes (instance_id, node_key, status, prefix_json, mc_score, visits, in_pool) "
            "VALUES (?, ?, 'ready', ?, ?, ?, ?)",
            (instance_id, key, json.dumps(prefix), mc_cache,
             spec.get("visits", 0), spec.get("in_pool", 0)))
        for idx, correct in enumerate(spec["corrects"]):
            _add_rollout(conn, instance_id, key, idx, correct, n_steps=len(prefix) + 3)
    conn.commit()


def _add_rollout(conn: sqlite3.Connection, instance_id: str, node_key: str, idx: int,
                 correct: int, n_steps: int = 5) -> None:
    result = {"instance_id": instance_id, "node_key": node_key, "rollout_idx": idx,
              "reward": 1.0 if correct else 0.0, "correct": bool(correct),
              "steps": make_steps(n_steps), "exit_status": "Submitted" if correct else "Submitted",
              "submission": "diff", "n_calls": n_steps, "cost": 0.1, "duration": 1.0,
              "replay_drift": False, "trace_id": None, "reward_details": {}, "error": None}
    conn.execute(
        "INSERT INTO rollouts (instance_id, node_key, rollout_idx, is_consumed, result_json, "
        "n_steps, correct, reward, submitted, exit_status) VALUES (?, ?, ?, 0, ?, ?, ?, ?, 1, 'Submitted')",
        (instance_id, node_key, idx, json.dumps(result), n_steps, correct,
         1.0 if correct else 0.0))


def make_splits_parquet(path: Path, rows: list[tuple[str, str, str | None]]) -> Path:
    """``[(instance_id, repo, prm_split)]`` → splits.parquet（``None`` split = 生成池外）。"""
    df = pd.DataFrame([
        {"instance_id": r[0], "repo": r[1],
         "prm_split": r[2], "gen_pool": r[2] is not None}
        for r in rows
    ])
    df.to_parquet(path, index=False)
    return path


def make_builder_config(db_path: str, splits_path: str, out_dir: str, **overrides) -> dict:
    """最小 build 配置（DEFAULT_CONFIG 结构）。"""
    from prm.build_dataset import DEFAULT_CONFIG, deep_merge
    cfg = deep_merge(DEFAULT_CONFIG, {
        "db": {"path": str(db_path)},
        "splits": {"path": str(splits_path)},
        "output": {"dir": str(out_dir)},
        "tokenizer": None,
        "build": {"workers": 1, "overwrite": True, **overrides},
    })
    return cfg
