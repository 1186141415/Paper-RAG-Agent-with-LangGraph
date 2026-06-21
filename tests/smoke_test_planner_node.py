"""
Smoke test for planner_node helpers and node behavior.

Run:
    python tests/smoke_test_planner_node.py
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from app.graph.agents import (
    MAX_SUBTASKS,
    _fallback_plan,
    _normalize_subtasks,
    _parse_planner_json,
    build_planner_node,
)

AVAILABLE_SOURCES = ["Paper1.pdf", "Paper2.pdf", "Paper3.pdf"]
QUERY = "What is the difference between paper1 and paper2?"


def test_comparison_subtasks_keep_valid_sources() -> None:
    raw_subtasks = [
        {
            "sub_id": "s1",
            "sub_question": "What method does paper1 use?",
            "target_source": "Paper1.pdf",
        },
        {
            "sub_id": "s2",
            "sub_question": "What method does paper2 use?",
            "target_source": "Paper2.pdf",
        },
    ]

    subtasks = _normalize_subtasks(raw_subtasks, QUERY, AVAILABLE_SOURCES)

    assert len(subtasks) == 2
    assert subtasks[0]["target_source"] == "Paper1.pdf"
    assert subtasks[1]["target_source"] == "Paper2.pdf"


def test_invalid_target_source_becomes_none() -> None:
    raw_subtasks = [
        {
            "sub_id": "s1",
            "sub_question": "What does paper9 discuss?",
            "target_source": "Paper9.pdf",
        }
    ]

    subtasks = _normalize_subtasks(raw_subtasks, QUERY, AVAILABLE_SOURCES)

    assert len(subtasks) == 1
    assert subtasks[0]["target_source"] is None


def test_subtasks_truncated_to_max() -> None:
    raw_subtasks = [
        {
            "sub_id": f"s{i}",
            "sub_question": f"Question {i}",
            "target_source": AVAILABLE_SOURCES[i % len(AVAILABLE_SOURCES)],
        }
        for i in range(1, MAX_SUBTASKS + 3)
    ]

    subtasks = _normalize_subtasks(raw_subtasks, QUERY, AVAILABLE_SOURCES)

    assert len(subtasks) == MAX_SUBTASKS
    assert subtasks[0]["sub_id"] == "s1"
    assert subtasks[-1]["sub_id"] == f"s{MAX_SUBTASKS}"


def test_empty_subtasks_fallback_to_single_subtask() -> None:
    subtasks = _normalize_subtasks([], QUERY, AVAILABLE_SOURCES)

    assert subtasks == _fallback_plan(QUERY)["subtasks"]


def test_json_parse_failure_uses_fallback_plan() -> None:
    planner_node = build_planner_node()

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not-json"))]
        )
        result = planner_node({
            "query": QUERY,
            "chat_history": [],
            "available_sources": AVAILABLE_SOURCES,
            "workflow_path": ["choose_tool"],
        })

    assert result["question_type"] == "UNKNOWN"
    assert result["subtasks"] == _fallback_plan(QUERY)["subtasks"]
    assert "error" not in result


def test_planner_node_appends_workflow_path() -> None:
    planner_node = build_planner_node()
    llm_payload = {
        "question_type": "COMPARISON",
        "subtasks": [
            {
                "sub_id": "s1",
                "sub_question": "What method does paper1 use?",
                "target_source": "Paper1.pdf",
            },
            {
                "sub_id": "s2",
                "sub_question": "What method does paper2 use?",
                "target_source": "Paper2.pdf",
            },
        ],
    }

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(llm_payload)))]
        )
        result = planner_node({
            "query": QUERY,
            "chat_history": [],
            "available_sources": AVAILABLE_SOURCES,
            "workflow_path": ["choose_tool"],
        })

    assert result["workflow_path"] == ["choose_tool", "planner"]
    assert result["question_type"] == "COMPARISON"
    assert len(result["subtasks"]) == 2


def test_planner_failure_does_not_write_error() -> None:
    planner_node = build_planner_node()

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.side_effect = RuntimeError("llm unavailable")
        result = planner_node({
            "query": QUERY,
            "chat_history": [],
            "available_sources": AVAILABLE_SOURCES,
        })

    assert result["question_type"] == "UNKNOWN"
    assert result["subtasks"] == _fallback_plan(QUERY)["subtasks"]
    assert "error" not in result
    assert result["workflow_path"] == ["planner"]


def test_parse_planner_json_strips_markdown_fence() -> None:
    content = """```json
{"question_type": "SPECIFIC", "subtasks": []}
```"""
    parsed = _parse_planner_json(content)
    assert parsed["question_type"] == "SPECIFIC"


def main() -> int:
    tests = [
        test_comparison_subtasks_keep_valid_sources,
        test_invalid_target_source_becomes_none,
        test_subtasks_truncated_to_max,
        test_empty_subtasks_fallback_to_single_subtask,
        test_json_parse_failure_uses_fallback_plan,
        test_planner_node_appends_workflow_path,
        test_planner_failure_does_not_write_error,
        test_parse_planner_json_strips_markdown_fence,
    ]

    passed = 0
    for test in tests:
        name = test.__name__
        print(f"RUN {name}")
        test()
        print(f"PASS {name}")
        passed += 1

    print(f"\nAll {passed}/{len(tests)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
