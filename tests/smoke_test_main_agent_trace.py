"""
Smoke test for /ask agent_trace compatibility.

Run:
    python tests/smoke_test_main_agent_trace.py
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

from app.main import _build_agent_trace

LONG_ANSWER = "x" * 400


def test_legacy_path_uses_tool_result_for_tool_used_and_input() -> None:
    result = {
        "decision": {"tool": "rag", "input": "original query", "reason": "papers"},
        "tool_result": {
            "tool_name": "time",
            "tool_input": "What time is it now?",
            "tool_output": "2026-06-21 17:00:00",
        },
        "workflow_path": ["choose_tool", "execute_tool", "generate_answer"],
    }

    trace = _build_agent_trace(result)

    assert trace["tool_used"] == "time"
    assert trace["tool_input"] == "What time is it now?"
    assert trace["workflow"] == ["choose_tool", "execute_tool", "generate_answer"]


def test_multi_agent_path_falls_back_to_decision_without_tool_result() -> None:
    result = {
        "decision": {
            "tool": "rag",
            "input": "What is the difference between paper1 and paper2?",
            "reason": "papers",
        },
        "question_type": "COMPARISON",
        "subtasks": [
            {
                "sub_id": "s1",
                "sub_question": "What method does paper1 use?",
                "target_source": "Paper1.pdf",
            }
        ],
        "workflow_path": [
            "choose_tool",
            "planner",
            "answer_worker",
            "synthesizer",
            "generate_answer",
        ],
    }

    trace = _build_agent_trace(result)

    assert trace["tool_used"] == "rag"
    assert trace["tool_input"] == "What is the difference between paper1 and paper2?"
    assert "planner" in trace["workflow"]


def test_plan_contains_question_type_and_subtasks() -> None:
    subtasks = [
        {
            "sub_id": "s1",
            "sub_question": "Q1",
            "target_source": "Paper1.pdf",
        }
    ]
    trace = _build_agent_trace({
        "question_type": "COMPARISON",
        "subtasks": subtasks,
    })

    assert trace["plan"] == {
        "question_type": "COMPARISON",
        "subtasks": subtasks,
    }


def test_sub_answers_are_summaries_without_full_chunks() -> None:
    trace = _build_agent_trace({
        "sub_answers": [
            {
                "sub_id": "s1",
                "target_source": "Paper1.pdf",
                "sub_question": "Q1",
                "answer": LONG_ANSWER,
                "context_sufficient": True,
                "error": "",
                "retrieved_chunks": [
                    {"source": "Paper1.pdf", "text": "chunk text", "distance": 1.0},
                ],
            }
        ],
    })

    assert len(trace["sub_answers"]) == 1
    summary = trace["sub_answers"][0]
    assert summary["retrieved_chunk_count"] == 1
    assert len(summary["answer_preview"]) == 300
    assert "retrieved_chunks" not in summary
    assert summary["sub_id"] == "s1"
    assert summary["target_source"] == "Paper1.pdf"


def test_missing_fields_do_not_raise() -> None:
    trace = _build_agent_trace({})

    assert trace["tool_used"] == ""
    assert trace["tool_input"] == ""
    assert trace["plan"] == {"question_type": "", "subtasks": []}
    assert trace["sub_answers"] == []
    assert trace["available_sources"] == []
    assert trace["workflow"] == ["choose_tool", "execute_tool", "generate_answer"]


def test_available_sources_passthrough() -> None:
    trace = _build_agent_trace({
        "available_sources": ["Paper1.pdf", "Paper2.pdf"],
    })

    assert trace["available_sources"] == ["Paper1.pdf", "Paper2.pdf"]


def test_ask_top_level_response_structure_unchanged() -> None:
    mcp_client_module = MagicMock()
    mcp_client_module.MultiServerMCPClient = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "langchain_mcp_adapters": MagicMock(),
            "langchain_mcp_adapters.client": mcp_client_module,
        },
    ):
        import app.main as main_module

    mock_workflow = MagicMock()
    mock_workflow.invoke.return_value = {
        "final_answer": "final",
        "retrieved_chunks": [{"source": "Paper1.pdf", "text": "chunk"}],
        "decision": {"tool": "rag", "input": "hello", "reason": "papers"},
        "question_type": "BROAD",
        "subtasks": [],
        "sub_answers": [],
        "available_sources": ["Paper1.pdf"],
        "context_sufficient": True,
        "context_metrics": {"multi_agent": True},
        "workflow_path": [
            "choose_tool",
            "planner",
            "answer_worker",
            "synthesizer",
            "generate_answer",
        ],
    }

    with patch.object(main_module, "_init_rag_and_workflow", return_value=(1, 1)):
        with patch.object(main_module, "workflow", mock_workflow):
            from fastapi.testclient import TestClient

            with TestClient(main_module.app) as client:
                response = client.post(
                    "/ask",
                    json={"session_id": "trace-test", "question": "hello"},
                )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "session_id",
        "question",
        "answer",
        "chunks",
        "agent_trace",
    }
    assert payload["answer"] == "final"
    assert payload["chunks"] == [{"source": "Paper1.pdf", "text": "chunk"}]
    assert payload["agent_trace"]["tool_used"] == "rag"
    assert payload["agent_trace"]["plan"]["question_type"] == "BROAD"
    assert "planner" in payload["agent_trace"]["workflow"]


def main() -> int:
    tests = [
        test_legacy_path_uses_tool_result_for_tool_used_and_input,
        test_multi_agent_path_falls_back_to_decision_without_tool_result,
        test_plan_contains_question_type_and_subtasks,
        test_sub_answers_are_summaries_without_full_chunks,
        test_missing_fields_do_not_raise,
        test_available_sources_passthrough,
        test_ask_top_level_response_structure_unchanged,
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
