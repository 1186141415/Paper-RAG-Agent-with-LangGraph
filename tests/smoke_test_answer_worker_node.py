"""
Smoke test for answer_worker_node.

Run:
    python tests/smoke_test_answer_worker_node.py
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

from app.graph.agents import build_answer_worker_node

QUERY = "What is the difference between paper1 and paper2?"
CHAT_HISTORY = [{"role": "user", "content": "hello"}]

SUBTASKS = [
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


class FakeRAG:
    def __init__(self, responses: list[dict[str, Any]] | None = None, fail_on: set[str] | None = None):
        self.responses = responses or []
        self.fail_on = fail_on or set()
        self.calls: list[dict[str, Any]] = []

    def ask_with_trace(self, question, chat_history=None, source=None):
        self.calls.append({
            "question": question,
            "chat_history": chat_history,
            "source": source,
        })

        for subtask in SUBTASKS:
            if subtask["sub_question"] == question and subtask["sub_id"] in self.fail_on:
                raise RuntimeError(f"subtask {subtask['sub_id']} failed")

        if self.responses:
            index = len(self.calls) - 1
            if index < len(self.responses):
                return self.responses[index]

        return {
            "answer": f"answer for {question}",
            "retrieved_chunks": [{"source": source, "text": "chunk"}],
            "context_sufficient": True,
            "context_metrics": {"distance_gate_passed": True},
        }


def _base_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "query": QUERY,
        "chat_history": CHAT_HISTORY,
        "subtasks": SUBTASKS,
        "workflow_path": ["planner"],
    }
    state.update(overrides)
    return state


def test_worker_calls_ask_with_trace_per_subtask() -> None:
    rag = FakeRAG()
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state())

    assert len(rag.calls) == 2
    assert len(result["sub_answers"]) == 2
    assert result["sub_answers"][0]["answer"] == "answer for What method does paper1 use?"


def test_target_source_passed_to_ask_with_trace() -> None:
    rag = FakeRAG()
    worker = build_answer_worker_node(rag=rag)
    worker(_base_state())

    assert rag.calls[0]["source"] == "Paper1.pdf"
    assert rag.calls[1]["source"] == "Paper2.pdf"


def test_chat_history_is_forwarded() -> None:
    rag = FakeRAG()
    worker = build_answer_worker_node(rag=rag)
    worker(_base_state())

    assert rag.calls[0]["chat_history"] == CHAT_HISTORY
    assert rag.calls[1]["chat_history"] == CHAT_HISTORY


def test_empty_subtasks_fallback_to_single_subtask() -> None:
    rag = FakeRAG()
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state(subtasks=[]))

    assert len(rag.calls) == 1
    assert rag.calls[0]["question"] == QUERY
    assert rag.calls[0]["source"] is None
    assert result["sub_answers"][0]["sub_id"] == "s1"
    assert result["sub_answers"][0]["sub_question"] == QUERY


def test_rag_none_does_not_raise_and_records_error() -> None:
    worker = build_answer_worker_node(rag=None)
    result = worker(_base_state())

    assert len(result["sub_answers"]) == 2
    for sub_answer in result["sub_answers"]:
        assert sub_answer["error"] == "RAG system is not available."
        assert sub_answer["context_sufficient"] is False
        assert sub_answer["answer"] == ""
    assert "error" not in result


def test_single_subtask_failure_does_not_block_others() -> None:
    rag = FakeRAG(fail_on={"s1"})
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state())

    assert len(result["sub_answers"]) == 2
    assert result["sub_answers"][0]["error"] != ""
    assert result["sub_answers"][0]["context_sufficient"] is False
    assert result["sub_answers"][1]["error"] == ""
    assert result["sub_answers"][1]["answer"] != ""


def test_workflow_path_includes_answer_worker() -> None:
    rag = FakeRAG()
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state(workflow_path=["planner"]))

    assert result["workflow_path"] == ["planner", "answer_worker"]


def test_worker_failure_does_not_write_state_error() -> None:
    rag = MagicMock()
    rag.ask_with_trace.side_effect = RuntimeError("boom")
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state(subtasks=[SUBTASKS[0]]))

    assert result["sub_answers"][0]["error"] == "boom"
    assert "error" not in result


def test_sub_answer_shape_matches_contract() -> None:
    rag = FakeRAG(responses=[{
        "answer": "ok",
        "retrieved_chunks": [{"source": "Paper1.pdf", "text": "t"}],
        "context_sufficient": True,
        "context_metrics": {"distance_gate_passed": True},
    }])
    worker = build_answer_worker_node(rag=rag)
    result = worker(_base_state(subtasks=[SUBTASKS[0]]))

    sub_answer = result["sub_answers"][0]
    assert sub_answer == {
        "sub_id": "s1",
        "target_source": "Paper1.pdf",
        "sub_question": "What method does paper1 use?",
        "answer": "ok",
        "retrieved_chunks": [{"source": "Paper1.pdf", "text": "t"}],
        "context_sufficient": True,
        "context_metrics": {"distance_gate_passed": True},
        "error": "",
    }


def main() -> int:
    tests = [
        test_worker_calls_ask_with_trace_per_subtask,
        test_target_source_passed_to_ask_with_trace,
        test_chat_history_is_forwarded,
        test_empty_subtasks_fallback_to_single_subtask,
        test_rag_none_does_not_raise_and_records_error,
        test_single_subtask_failure_does_not_block_others,
        test_workflow_path_includes_answer_worker,
        test_worker_failure_does_not_write_state_error,
        test_sub_answer_shape_matches_contract,
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
