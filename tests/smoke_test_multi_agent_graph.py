"""
Smoke test for multi-agent RAG graph routing in builder.py.

Run:
    python tests/smoke_test_multi_agent_graph.py
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

from app.graph.builder import build_agent_graph, route_after_choose_tool
from app.tools import TOOLS

QUERY = "What is the difference between paper1 and paper2?"
AVAILABLE_SOURCES = ["Paper1.pdf", "Paper2.pdf", "Paper3.pdf"]


class FakeRAG:
    def ask_with_trace(self, question, chat_history=None, source=None):
        return {
            "answer": f"answer for {source or 'all'}: {question}",
            "retrieved_chunks": [
                {"source": source or "unknown", "text": f"chunk for {question}"},
            ],
            "context_sufficient": True,
            "context_metrics": {"distance_gate_passed": True},
        }


def _completion(content: str) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )


def _choose_tool_response(tool: str, tool_input: str) -> str:
    return json.dumps({
        "tool": tool,
        "input": tool_input,
        "reason": f"route to {tool}",
    })


def _planner_response() -> str:
    return json.dumps({
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
    })


def _base_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "session_id": "test_multi_agent",
        "query": QUERY,
        "chat_history": [],
        "available_sources": AVAILABLE_SOURCES,
    }
    state.update(overrides)
    return state


def test_route_after_choose_tool_rag_goes_to_planner() -> None:
    assert route_after_choose_tool({"decision": {"tool": "rag"}}) == "planner"
    assert route_after_choose_tool({"decision": {"tool": "calculator"}}) == "execute_tool"


def test_graph_compiles() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())
    assert graph is not None


def _patch_llm_side_effect(side_effect: list[MagicMock]):
    return patch(
        "app.llm_utils.client.chat.completions.create",
        side_effect=side_effect,
    )


def test_rag_path_workflow_and_final_answer() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())

    with _patch_llm_side_effect([
        _completion(_choose_tool_response("rag", QUERY)),
        _completion(_planner_response()),
        _completion(
            "[Source: Paper1.pdf] Paper1 uses CNN. "
            "[Source: Paper2.pdf] Paper2 uses GAN."
        ),
    ]):
        result = graph.invoke(_base_state())

    assert result["workflow_path"] == [
        "choose_tool",
        "planner",
        "answer_worker",
        "synthesizer",
        "generate_answer",
    ]
    assert result["final_answer"]
    assert "Paper1" in result["final_answer"] or "CNN" in result["final_answer"]
    assert "tool_result" not in result
    assert result.get("error", "") in ("", None)
    assert len(result.get("sub_answers", [])) == 2
    assert result["context_metrics"].get("multi_agent") is True


def test_non_rag_calculator_path_unchanged() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())

    with _patch_llm_side_effect([
        _completion(_choose_tool_response("calculator", "2+3")),
    ]):
        result = graph.invoke(_base_state(query="What is 2+3?"))

    assert result["workflow_path"] == [
        "choose_tool",
        "execute_tool",
        "generate_answer",
    ]
    assert "planner" not in result["workflow_path"]
    assert result["final_answer"] == "5"
    assert result["tool_result"]["tool_name"] == "calculator"


def test_non_rag_llm_path_does_not_enter_planner() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())

    with _patch_llm_side_effect([
        _completion(_choose_tool_response("llm", "Say hi")),
        _completion("Hi there"),
    ]):
        result = graph.invoke(_base_state(query="Say hi"))

    assert result["workflow_path"] == [
        "choose_tool",
        "execute_tool",
        "generate_answer",
    ]
    assert "planner" not in result["workflow_path"]
    assert result["tool_result"]["tool_name"] == "llm"
    assert result["final_answer"] == "Hi there"


def test_rag_path_generate_answer_passthrough_without_tool_result() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())

    with _patch_llm_side_effect([
        _completion(_choose_tool_response("rag", QUERY)),
        _completion(_planner_response()),
        _completion("[Source: Paper1.pdf] A [Source: Paper2.pdf] B"),
    ]):
        result = graph.invoke(_base_state())

    assert "tool_result" not in result
    assert result["final_answer"]
    assert isinstance(result.get("retrieved_chunks"), list)


def test_multi_agent_nodes_do_not_write_state_error() -> None:
    graph = build_agent_graph(TOOLS, rag=FakeRAG())

    with _patch_llm_side_effect([
        _completion(_choose_tool_response("rag", QUERY)),
        _completion(_planner_response()),
        _completion("combined answer"),
    ]):
        result = graph.invoke(_base_state())

    assert not result.get("error")


def main() -> int:
    tests = [
        test_route_after_choose_tool_rag_goes_to_planner,
        test_graph_compiles,
        test_rag_path_workflow_and_final_answer,
        test_non_rag_calculator_path_unchanged,
        test_non_rag_llm_path_does_not_enter_planner,
        test_rag_path_generate_answer_passthrough_without_tool_result,
        test_multi_agent_nodes_do_not_write_state_error,
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
