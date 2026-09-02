# SPDX-License-Identifier: BSD-3-Clause

"""``submit_locations`` 工具纯逻辑测试：schema / 严格解析 / submission 宽松解析。

测试为**纯逻辑**（不 import agent / minisweagent —— ``agent.submit_tool`` 与
``mcts.reward`` 均无重依赖）。
"""

import json

import pytest

from agent.submit_tool import (
    SUBMIT_REMINDER,
    SUBMIT_TOOL,
    SUBMIT_TOOL_NAME,
    parse_submit_locations,
)
from mcts.reward import locations_from_submission


class FakeToolCall:
    def __init__(self, arguments: str, id: str = "tc1"):
        self.function = type("F", (), {"arguments": arguments})()
        self.id = id


def test_tool_schema_shape():
    assert SUBMIT_TOOL["function"]["name"] == SUBMIT_TOOL_NAME
    params = SUBMIT_TOOL["function"]["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["locations"]
    items = params["properties"]["locations"]["items"]
    assert items["required"] == ["file"]
    assert set(items["properties"]) == {"file", "class_name", "function_name"}


def test_parse_valid_single_location():
    call = FakeToolCall(json.dumps({
        "locations": [{"file": "src/a.py", "class_name": "A", "function_name": "f"}],
    }))
    action = parse_submit_locations(call)
    assert action["tool"] == SUBMIT_TOOL_NAME
    assert action["tool_call_id"] == "tc1"
    assert action["locations"] == [
        {"file": "src/a.py", "class_name": "A", "function_name": "f"}
    ]


def test_parse_valid_multiple_and_optional_fields():
    call = FakeToolCall(json.dumps({
        "locations": [
            {"file": "src/a.py", "class_name": "A", "function_name": "f"},
            {"file": "src/b.py", "function_name": "g"},
            {"file": "src/c.py"},
            {"file": "src/d.py", "class_name": "D", "function_name": None},
        ],
    }))
    action = parse_submit_locations(call)
    assert len(action["locations"]) == 4
    assert action["locations"][2] == {"file": "src/c.py", "class_name": None,
                                      "function_name": None}


@pytest.mark.parametrize("payload, expect", [
    ("not json", "parsing"),
    ("{}", "locations"),
    ('{"locations": {}}', "must be a list"),
    ('{"locations": []}', "at least one"),
    ('{"locations": [{}]}', "non-empty 'file'"),
    ('{"locations": [{"file": ""}]}', "non-empty 'file'"),
    ('{"locations": [{"file": "a.py", "class_name": 3}]}', "class_name"),
    ('{"locations": [{"file": "a.py", "function_name": []}]}', "function_name"),
    ('{"locations": [5]}', "object"),
])
def test_parse_invalid(payload, expect):
    with pytest.raises(ValueError, match=expect):
        parse_submit_locations(FakeToolCall(payload))


def test_parse_rejects_duplicates():
    payload = json.dumps({
        "locations": [
            {"file": "a.py", "class_name": "A", "function_name": "f"},
            {"file": "a.py", "class_name": "A", "function_name": "f"},
        ],
    })
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        parse_submit_locations(FakeToolCall(payload))


def test_parse_allows_same_file_different_entries():
    payload = json.dumps({
        "locations": [
            {"file": "a.py", "class_name": "A", "function_name": "f"},
            {"file": "a.py", "class_name": "B", "function_name": "g"},
        ],
    })
    action = parse_submit_locations(FakeToolCall(payload))
    assert len(action["locations"]) == 2


def test_reminder_message_mentions_submit_tool():
    assert SUBMIT_TOOL_NAME in SUBMIT_REMINDER


def test_locations_from_submission_valid():
    locs = locations_from_submission(json.dumps([
        {"file": "a.py", "class_name": "A", "function_name": "f"},
        {"file": "b.py"},
    ]))
    assert locs == [
        {"file": "a.py", "class_name": "A", "function_name": "f"},
        {"file": "b.py", "class_name": None, "function_name": None},
    ]


@pytest.mark.parametrize("submission", [
    "",
    "   ",
    "not json",
    "42",
    '{"file": "a.py"}',
    '[{"file": "a.py"}, "x"]',
    '[{"class_name": "A"}]',
    '[{"file": "a.py", "class_name": 1}]',
])
def test_locations_from_submission_invalid(submission):
    assert locations_from_submission(submission) is None
