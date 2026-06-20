"""
Smoke test for optional source-aware retrieval.

Run:
    python tests/smoke_test_source_filter.py
"""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.rag_system import RAGSystem
from app.vector_store.faiss_store import FaissVectorStore

REQUIRED_CHUNK_FIELDS = {"source", "text", "distance", "retrieval_rank"}

SYNTHETIC_CHUNKS = [
    {"text": "Paper1 method overview", "source": "Paper1.pdf"},
    {"text": "Paper1 experiment results", "source": "Paper1.pdf"},
    {"text": "Paper1 conclusion", "source": "Paper1.pdf"},
    {"text": "Paper2 architecture design", "source": "Paper2.pdf"},
    {"text": "Paper2 benchmark table", "source": "Paper2.pdf"},
    {"text": "Paper2 ablation study", "source": "Paper2.pdf"},
    {"text": "Paper3 unrelated topic A", "source": "Paper3.pdf"},
    {"text": "Paper3 unrelated topic B", "source": "Paper3.pdf"},
    {"text": "Paper3 unrelated topic C", "source": "Paper3.pdf"},
]

# Fixed 4-d vectors so FAISS ranking is deterministic without live embedding APIs.
CHUNK_VECTORS = {
  "Paper1 method overview": np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
  "Paper1 experiment results": np.array([0.95, 0.05, 0.0, 0.0], dtype="float32"),
  "Paper1 conclusion": np.array([0.9, 0.1, 0.0, 0.0], dtype="float32"),
  "Paper2 architecture design": np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
  "Paper2 benchmark table": np.array([0.05, 0.95, 0.0, 0.0], dtype="float32"),
  "Paper2 ablation study": np.array([0.1, 0.9, 0.0, 0.0], dtype="float32"),
  "Paper3 unrelated topic A": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
  "Paper3 unrelated topic B": np.array([0.0, 0.0, 0.95, 0.05], dtype="float32"),
  "Paper3 unrelated topic C": np.array([0.0, 0.0, 0.9, 0.1], dtype="float32"),
}

QUERY_VECTOR = np.array([0.2, 0.2, 0.9, 0.1], dtype="float32")


def _mock_get_embedding(text: str) -> np.ndarray:
    if text == "comparison query":
        return QUERY_VECTOR
    return CHUNK_VECTORS[text]


def _assert_chunk_shape(chunks: list[dict]) -> None:
    assert chunks, "expected non-empty retrieval results"
    for chunk in chunks:
        assert REQUIRED_CHUNK_FIELDS.issubset(chunk.keys())
        assert chunk["source"]
        assert chunk["text"]
        assert chunk["retrieval_rank"] >= 1


def _assert_all_from_source(chunks: list[dict], source: str) -> None:
    _assert_chunk_shape(chunks)
    assert all(chunk["source"] == source for chunk in chunks), (
        f"expected only {source}, got sources={[c['source'] for c in chunks]}"
    )


def test_faiss_source_filter() -> None:
    store = FaissVectorStore()

    with patch("app.vector_store.faiss_store.get_embedding", side_effect=_mock_get_embedding):
        store.build(SYNTHETIC_CHUNKS)

        unfiltered = store.search("comparison query", k=3)
        _assert_chunk_shape(unfiltered)
        assert len(unfiltered) == 3

        paper1 = store.search("comparison query", k=3, source="Paper1.pdf")
        _assert_all_from_source(paper1, "Paper1.pdf")
        assert len(paper1) == 3

        paper2 = store.search("comparison query", k=3, source="Paper2.pdf")
        _assert_all_from_source(paper2, "Paper2.pdf")
        assert len(paper2) == 3

        # Global top-3 without filter is dominated by Paper3 for this query vector.
        assert any(chunk["source"] == "Paper3.pdf" for chunk in unfiltered)


def test_rag_system_retrieve_source_filter() -> None:
    store = FaissVectorStore()

    with patch("app.vector_store.faiss_store.get_embedding", side_effect=_mock_get_embedding):
        store.build(SYNTHETIC_CHUNKS)
        rag = RAGSystem(SYNTHETIC_CHUNKS, top_k=3, vector_store=store)

        unfiltered = rag.retrieve("comparison query", k=3)
        _assert_chunk_shape(unfiltered)

        paper1 = rag.retrieve("comparison query", k=3, source="Paper1.pdf")
        _assert_all_from_source(paper1, "Paper1.pdf")

        paper2 = rag.retrieve("comparison query", k=3, source="Paper2.pdf")
        _assert_all_from_source(paper2, "Paper2.pdf")


def test_ask_with_trace_backward_compatible_signature() -> None:
    sig = inspect.signature(RAGSystem.ask_with_trace)
    assert "source" in sig.parameters
    assert sig.parameters["source"].default is None


def test_ask_endpoint_still_available() -> None:
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
        "final_answer": "mock answer",
        "retrieved_chunks": [],
        "decision": {"tool_name": "rag"},
        "tool_result": {"tool_name": "rag", "tool_input": "hello"},
        "fallback_used": False,
        "context_sufficient": True,
        "context_metrics": {},
        "error": "",
        "retry_count": 0,
        "workflow_path": ["choose_tool", "execute_tool", "generate_answer"],
    }

    with patch.object(main_module, "_init_rag_and_workflow", return_value=(3, 9)):
        with patch.object(main_module, "workflow", mock_workflow):
            with TestClient(main_module.app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "session_id": "smoke-source-filter",
                        "question": "What is paper1 about?",
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "mock answer"
    assert payload["question"] == "What is paper1 about?"
    assert "chunks" in payload
    assert "agent_trace" in payload
    mock_workflow.invoke.assert_called_once()


def main() -> int:
    tests = [
        test_faiss_source_filter,
        test_rag_system_retrieve_source_filter,
        test_ask_with_trace_backward_compatible_signature,
        test_ask_endpoint_still_available,
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
