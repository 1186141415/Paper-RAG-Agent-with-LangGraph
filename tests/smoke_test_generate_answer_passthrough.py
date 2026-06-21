"""
Smoke test for generate_answer_node passthrough behavior.

Run:
    python tests/smoke_test_generate_answer_passthrough.py
"""

from __future__ import annotations

import sys
from typing import Any

from app.graph.nodes import generate_answer_node

QUERY = "What is the difference between paper1 and paper2?"


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "query": QUERY,
        "workflow_path": ["planner", "answer_worker", "synthesizer"],
    }
    state.update(overrides)
    return state


def test_error_takes_priority_over_final_answer_passthrough() -> None:
    result = generate_answer_node(_base_state(
        error="upstream failed",
        final_answer="should not win",
        retrieved_chunks=[{"source": "Paper1.pdf", "text": "x"}],
    ))

    assert result["final_answer"] == "系统执行过程中出现问题：upstream failed"
    assert result["workflow_path"] == [
        "planner",
        "answer_worker",
        "synthesizer",
        "generate_answer",
    ]


def test_passthrough_final_answer_without_tool_result() -> None:
    result = generate_answer_node(_base_state(
        final_answer="synthesized comparison answer",
    ))

    assert result["final_answer"] == "synthesized comparison answer"
    assert "error" not in result


def test_passthrough_preserves_retrieval_fields() -> None:
    result = generate_answer_node(_base_state(
        final_answer="synthesized answer",
        retrieved_chunks=[{"source": "Paper1.pdf", "text": "chunk-a"}],
        context_sufficient=True,
        context_metrics={"multi_agent": True, "question_type": "COMPARISON"},
    ))

    assert result["retrieved_chunks"] == [{"source": "Paper1.pdf", "text": "chunk-a"}]
    assert result["context_sufficient"] is True
    assert result["context_metrics"] == {
        "multi_agent": True,
        "question_type": "COMPARISON",
    }


def test_legacy_dict_tool_output_unchanged() -> None:
    result = generate_answer_node(_base_state(
        tool_result={
            "tool_name": "rag",
            "tool_input": QUERY,
            "tool_output": {
                "answer": "rag answer",
                "retrieved_chunks": [{"source": "Paper1.pdf", "text": "chunk"}],
                "context_sufficient": False,
                "context_metrics": {"distance_gate_passed": False},
            },
        },
    ))

    assert result["final_answer"] == "rag answer"
    assert result["retrieved_chunks"] == [{"source": "Paper1.pdf", "text": "chunk"}]
    assert result["context_sufficient"] is False
    assert result["context_metrics"] == {"distance_gate_passed": False}


def test_legacy_string_tool_output_unchanged() -> None:
    result = generate_answer_node(_base_state(
        tool_result={
            "tool_name": "time",
            "tool_input": "What time is it now?",
            "tool_output": "2026-06-21 16:00:00",
        },
    ))

    assert result["final_answer"] == "2026-06-21 16:00:00"
    assert result["retrieved_chunks"] == []


def test_missing_final_answer_and_tool_result_returns_safe_message() -> None:
    result = generate_answer_node(_base_state())

    assert result["final_answer"] == "系统没有生成可用答案。"
    assert result["retrieved_chunks"] == []
    assert "error" not in result


def test_workflow_path_appends_generate_answer() -> None:
    result = generate_answer_node(_base_state(
        final_answer="ok",
        workflow_path=["planner", "synthesizer"],
    ))

    assert result["workflow_path"] == [
        "planner",
        "synthesizer",
        "generate_answer",
    ]


def test_tool_result_takes_priority_when_both_present() -> None:
    result = generate_answer_node(_base_state(
        final_answer="from synthesizer",
        tool_result={
            "tool_name": "rag",
            "tool_input": QUERY,
            "tool_output": {
                "answer": "from tool_result",
                "retrieved_chunks": [],
                "context_sufficient": True,
                "context_metrics": {},
            },
        },
    ))

    assert result["final_answer"] == "from tool_result"


def main() -> int:
    tests = [
        test_error_takes_priority_over_final_answer_passthrough,
        test_passthrough_final_answer_without_tool_result,
        test_passthrough_preserves_retrieval_fields,
        test_legacy_dict_tool_output_unchanged,
        test_legacy_string_tool_output_unchanged,
        test_missing_final_answer_and_tool_result_returns_safe_message,
        test_workflow_path_appends_generate_answer,
        test_tool_result_takes_priority_when_both_present,
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
