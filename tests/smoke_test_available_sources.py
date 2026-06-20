"""
Smoke test for AgentWorkflow.available_sources injection.

Run:
    python tests/smoke_test_available_sources.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.graph.builder import build_agent_graph
from app.graph.workflow import AgentWorkflow
from app.tools import TOOLS


def _make_rag(chunks: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(chunks=chunks)


def test_get_available_sources_deduplicates_and_sorts() -> None:
    rag = _make_rag([
        {"text": "a", "source": "Paper2.pdf"},
        {"text": "b", "source": "Paper1.pdf"},
        {"text": "c", "source": "Paper2.pdf"},
        {"text": "d", "source": "Paper3.pdf"},
    ])
    workflow = AgentWorkflow(TOOLS, rag=rag)

    assert workflow._get_available_sources() == [
        "Paper1.pdf",
        "Paper2.pdf",
        "Paper3.pdf",
    ]


def test_get_available_sources_when_rag_is_none() -> None:
    workflow = AgentWorkflow(TOOLS, rag=None)
    assert workflow._get_available_sources() == []


def test_get_available_sources_when_chunks_empty() -> None:
    rag = _make_rag([])
    workflow = AgentWorkflow(TOOLS, rag=rag)
    assert workflow._get_available_sources() == []


def test_invoke_injects_available_sources_without_breaking_graph() -> None:
    rag = _make_rag([
        {"text": "a", "source": "Paper2.pdf"},
        {"text": "b", "source": "Paper1.pdf"},
    ])

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "session_id": "test_001",
        "query": "What time is it now?",
        "chat_history": [],
        "available_sources": ["Paper1.pdf", "Paper2.pdf"],
        "final_answer": "mock",
        "workflow_path": ["choose_tool", "execute_tool", "generate_answer"],
    }

    workflow = AgentWorkflow(TOOLS, rag=rag)
    workflow.graph = mock_graph

    result = workflow.invoke(
        session_id="test_001",
        query="What time is it now?",
        chat_history=[],
    )

    mock_graph.invoke.assert_called_once()
    initial_state = mock_graph.invoke.call_args.args[0]
    assert initial_state["available_sources"] == ["Paper1.pdf", "Paper2.pdf"]
    assert result["final_answer"] == "mock"
    assert result["workflow_path"] == [
        "choose_tool",
        "execute_tool",
        "generate_answer",
    ]


def test_direct_graph_invoke_still_works_without_available_sources() -> None:
    mcp_client_module = MagicMock()
    mcp_client_module.MultiServerMCPClient = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "langchain_mcp_adapters": MagicMock(),
            "langchain_mcp_adapters.client": mcp_client_module,
        },
    ):
        graph = build_agent_graph(TOOLS, rag=None)

    state = {
        "session_id": "test_001",
        "query": "What time is it now?",
        "chat_history": [],
    }

    result = graph.invoke(state)

    assert "final_answer" in result
    assert result.get("workflow_path") == [
        "choose_tool",
        "execute_tool",
        "generate_answer",
    ]


def main() -> int:
    tests = [
        test_get_available_sources_deduplicates_and_sorts,
        test_get_available_sources_when_rag_is_none,
        test_get_available_sources_when_chunks_empty,
        test_invoke_injects_available_sources_without_breaking_graph,
        test_direct_graph_invoke_still_works_without_available_sources,
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
