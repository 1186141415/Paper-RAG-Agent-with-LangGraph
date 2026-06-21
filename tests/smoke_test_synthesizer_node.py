"""
Smoke test for synthesizer_node.

Run:
    python tests/smoke_test_synthesizer_node.py
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

from app.graph.agents import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    _aggregate_context_sufficiency,
    _deterministic_synthesis_fallback,
    _merge_retrieved_chunks,
    build_synthesizer_node,
)

QUERY = "What is the difference between paper1 and paper2?"


def _sub_answer(
    sub_id: str,
    target_source: str | None,
    *,
    answer: str = "local answer",
    context_sufficient: bool = True,
    error: str = "",
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "sub_id": sub_id,
        "target_source": target_source,
        "sub_question": f"question for {sub_id}",
        "answer": answer,
        "retrieved_chunks": chunks or [
            {"source": target_source, "text": f"chunk-{sub_id}"},
        ],
        "context_sufficient": context_sufficient,
        "context_metrics": {"distance_gate_passed": context_sufficient},
        "error": error,
    }


def _base_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "query": QUERY,
        "question_type": "COMPARISON",
        "workflow_path": ["planner", "answer_worker"],
    }
    state.update(overrides)
    return state


def test_empty_sub_answers_returns_insufficient_without_error() -> None:
    synthesizer = build_synthesizer_node()
    result = synthesizer(_base_state(sub_answers=[]))

    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["context_sufficient"] is False
    assert result["retrieved_chunks"] == []
    assert result["context_metrics"]["synthesizer_mode"] == "no_sub_answers"
    assert result["context_metrics"]["multi_agent"] is True
    assert "error" not in result


def test_single_sufficient_sub_answer_generates_final_answer() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [_sub_answer("s1", "Paper1.pdf", answer="Paper1 uses CNN.")]

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[Source: Paper1.pdf]\nPaper1 uses CNN."))]
        )
        result = synthesizer(_base_state(
            question_type="BROAD",
            sub_answers=sub_answers,
        ))

    assert "Paper1 uses CNN" in result["final_answer"]
    assert result["context_sufficient"] is True
    assert result["context_metrics"]["synthesizer_mode"] == "llm_single"


def test_comparison_all_sufficient_sets_context_sufficient_true() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [
        _sub_answer("s1", "Paper1.pdf", answer="Paper1 uses CNN."),
        _sub_answer("s2", "Paper2.pdf", answer="Paper2 uses GAN."),
    ]

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=(
                    "[Source: Paper1.pdf] Paper1 uses CNN. "
                    "[Source: Paper2.pdf] Paper2 uses GAN."
                )
            ))]
        )
        result = synthesizer(_base_state(sub_answers=sub_answers))

    assert result["context_sufficient"] is True
    assert result["context_metrics"].get("partial_answer") is not True
    assert result["context_metrics"]["synthesizer_mode"] == "llm_comparison"


def test_comparison_partial_insufficient_sets_partial_answer() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [
        _sub_answer("s1", "Paper1.pdf", answer="Paper1 uses CNN.", context_sufficient=True),
        _sub_answer("s2", "Paper2.pdf", answer="", context_sufficient=False, error="gate failed"),
    ]

    result = synthesizer(_base_state(sub_answers=sub_answers))

    assert result["context_sufficient"] is False
    assert result["context_metrics"]["partial_answer"] is True
    assert result["context_metrics"]["synthesizer_mode"] == "deterministic_partial"
    assert "Paper2.pdf" in result["final_answer"]
    assert "部分论文证据不足" in result["final_answer"]


def test_all_insufficient_sets_context_sufficient_false() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [
        _sub_answer("s1", "Paper1.pdf", answer="", context_sufficient=False),
        _sub_answer("s2", "Paper2.pdf", answer="", context_sufficient=False),
    ]

    result = synthesizer(_base_state(sub_answers=sub_answers))

    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result["context_sufficient"] is False
    assert result["context_metrics"]["synthesizer_mode"] == "insufficient_evidence"


def test_merge_retrieved_chunks_deduplicates_by_source_and_text() -> None:
    sub_answers = [
        {
            "retrieved_chunks": [
                {"source": "Paper1.pdf", "text": "same", "distance": 1.0},
                {"source": "Paper2.pdf", "text": "unique", "distance": 1.1},
            ]
        },
        {
            "retrieved_chunks": [
                {"source": "Paper1.pdf", "text": "same", "distance": 1.2},
                {"source": "Paper2.pdf", "text": "another", "distance": 1.3},
            ]
        },
    ]

    merged = _merge_retrieved_chunks(sub_answers)

    assert len(merged) == 3
    assert merged[0] == {"source": "Paper1.pdf", "text": "same", "distance": 1.0}
    assert merged[1] == {"source": "Paper2.pdf", "text": "unique", "distance": 1.1}
    assert merged[2] == {"source": "Paper2.pdf", "text": "another", "distance": 1.3}


def test_context_metrics_include_multi_agent_fields() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [_sub_answer("s1", "Paper1.pdf")]

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[Source: Paper1.pdf]\nok"))]
        )
        result = synthesizer(_base_state(
            question_type="SPECIFIC",
            sub_answers=sub_answers,
        ))

    metrics = result["context_metrics"]
    assert metrics["multi_agent"] is True
    assert metrics["question_type"] == "SPECIFIC"
    assert metrics["sub_metrics"] == [{
        "sub_id": "s1",
        "target_source": "Paper1.pdf",
        "context_sufficient": True,
        "error": "",
    }]


def test_workflow_path_appends_synthesizer() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [_sub_answer("s1", "Paper1.pdf")]

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[Source: Paper1.pdf]\nok"))]
        )
        result = synthesizer(_base_state(sub_answers=sub_answers))

    assert result["workflow_path"] == ["planner", "answer_worker", "synthesizer"]


def test_llm_failure_uses_deterministic_fallback_without_raising() -> None:
    synthesizer = build_synthesizer_node()
    sub_answers = [
        _sub_answer("s1", "Paper1.pdf", answer="Paper1 uses CNN."),
        _sub_answer("s2", "Paper2.pdf", answer="Paper2 uses GAN."),
    ]

    with patch("app.graph.agents.client.chat.completions.create") as mock_create:
        mock_create.side_effect = RuntimeError("llm unavailable")
        result = synthesizer(_base_state(sub_answers=sub_answers))

    assert result["context_metrics"]["synthesizer_mode"] == "deterministic_fallback"
    assert "[Source: Paper1.pdf]" in result["final_answer"]
    assert "[Source: Paper2.pdf]" in result["final_answer"]
    assert "error" not in result


def test_aggregate_context_sufficiency_rules() -> None:
    comparison_all = [
        _sub_answer("s1", "Paper1.pdf", context_sufficient=True),
        _sub_answer("s2", "Paper2.pdf", context_sufficient=True),
    ]
    comparison_partial = [
        _sub_answer("s1", "Paper1.pdf", context_sufficient=True),
        _sub_answer("s2", "Paper2.pdf", context_sufficient=False),
    ]

    assert _aggregate_context_sufficiency("COMPARISON", comparison_all) == (True, False)
    assert _aggregate_context_sufficiency("COMPARISON", comparison_partial) == (False, True)
    assert _aggregate_context_sufficiency("BROAD", [_sub_answer("s1", "Paper1.pdf")]) == (True, False)


def test_deterministic_fallback_includes_source_labels() -> None:
    text = _deterministic_synthesis_fallback(
        QUERY,
        "COMPARISON",
        [
            _sub_answer("s1", "Paper1.pdf", answer="A"),
            _sub_answer("s2", "Paper2.pdf", answer="B"),
        ],
    )

    assert "[Source: Paper1.pdf]" in text
    assert "[Source: Paper2.pdf]" in text


def main() -> int:
    tests = [
        test_empty_sub_answers_returns_insufficient_without_error,
        test_single_sufficient_sub_answer_generates_final_answer,
        test_comparison_all_sufficient_sets_context_sufficient_true,
        test_comparison_partial_insufficient_sets_partial_answer,
        test_all_insufficient_sets_context_sufficient_false,
        test_merge_retrieved_chunks_deduplicates_by_source_and_text,
        test_context_metrics_include_multi_agent_fields,
        test_workflow_path_appends_synthesizer,
        test_llm_failure_uses_deterministic_fallback_without_raising,
        test_aggregate_context_sufficiency_rules,
        test_deterministic_fallback_includes_source_labels,
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
