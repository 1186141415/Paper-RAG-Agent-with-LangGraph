from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = ROOT_DIR / "eval" / "baseline_comparison_questions.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "eval" / "results"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "baseline_single_rag.json"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "baseline_single_rag_summary.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    comparison_questions = [
        q for q in questions
        if str(q.get("type", "")).upper() == "COMPARISON"
    ]

    if len(comparison_questions) < 5:
        raise ValueError(
            f"{path} must contain at least 5 COMPARISON questions; "
            f"found {len(comparison_questions)}."
        )

    return comparison_questions


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except error.URLError as e:
        raise RuntimeError(f"Failed to connect to {url}: {e.reason}") from e


def source_distribution(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(chunk.get("source") or "unknown") for chunk in chunks)
    total = sum(counts.values())

    percentages = {
        source: round(count / total, 4) if total else 0.0
        for source, count in sorted(counts.items())
    }

    return {
        "total_chunks": total,
        "counts": dict(sorted(counts.items())),
        "percentages": percentages,
        "unique_sources": sorted(counts),
    }


def expected_sources(question: dict[str, Any]) -> list[str]:
    raw = str(question.get("expected_source", ""))
    parts = [part.strip() for part in raw.replace(",", "/").split("/")]
    return [part for part in parts if part]


def has_expected_source_coverage(
    distribution: dict[str, Any],
    question: dict[str, Any],
) -> bool:
    retrieved = set(distribution.get("counts", {}))
    expected = set(expected_sources(question))
    return bool(expected) and expected.issubset(retrieved)


def infer_failure_mode(record: dict[str, Any]) -> str:
    if record.get("request_error"):
        return "api_request_error"

    context_sufficient = record.get("context_sufficient")
    trace = record.get("agent_trace", {})
    metrics = record.get("context_metrics", {})
    distribution = record.get("retrieved_source_distribution", {})
    counts = distribution.get("counts", {})
    unique_sources = [source for source, count in counts.items() if count > 0]

    if trace.get("tool_used") and trace.get("tool_used") != "rag":
        return "not_routed_to_rag"

    if context_sufficient is True and not record.get("expected_source_coverage"):
        return "answered_without_all_compared_sources"

    if context_sufficient is False and len(unique_sources) < 2:
        return "comparison_blocked_single_source_retrieval"

    if context_sufficient is False and not metrics.get("llm_relevance_check"):
        return "comparison_blocked_by_typed_relevance_gate"

    if context_sufficient is False and not metrics.get("distance_gate_passed", True):
        return "comparison_blocked_by_distance_gate"

    if context_sufficient is False:
        return "comparison_blocked_insufficient_context"

    return "comparison_answered"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions)
    endpoint = args.base_url.rstrip("/") + "/ask"
    run_id = args.run_id or f"baseline-single-rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        session_id = f"{run_id}-{question['id']}"
        payload = {
            "session_id": session_id,
            "question": question["question"],
        }

        started = time.perf_counter()
        print(f"[{index}/{len(questions)}] POST /ask {question['id']}: {question['question']}")

        try:
            response = post_json(endpoint, payload, timeout=args.timeout)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            chunks = response.get("chunks") or []
            agent_trace = response.get("agent_trace") or {}
            context_metrics = agent_trace.get("context_metrics") or {}
            distribution = source_distribution(chunks)

            record = {
                "id": question.get("id"),
                "type": question.get("type"),
                "expected_keywords": question.get("expected_keywords", []),
                "expected_source": question.get("expected_source", ""),
                "expected_sources": expected_sources(question),
                "expected_source_coverage": has_expected_source_coverage(distribution, question),
                "session_id": response.get("session_id", session_id),
                "question": response.get("question", question["question"]),
                "answer": response.get("answer", ""),
                "chunks": chunks,
                "agent_trace": agent_trace,
                "context_sufficient": agent_trace.get("context_sufficient"),
                "context_metrics": context_metrics,
                "retrieved_source_distribution": distribution,
                "latency_ms": elapsed_ms,
                "request_error": "",
            }

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            if not args.continue_on_error:
                raise

            distribution = source_distribution([])
            record = {
                "id": question.get("id"),
                "type": question.get("type"),
                "expected_keywords": question.get("expected_keywords", []),
                "expected_source": question.get("expected_source", ""),
                "expected_sources": expected_sources(question),
                "expected_source_coverage": False,
                "session_id": session_id,
                "question": question["question"],
                "answer": "",
                "chunks": [],
                "agent_trace": {},
                "context_sufficient": None,
                "context_metrics": {},
                "retrieved_source_distribution": distribution,
                "latency_ms": elapsed_ms,
                "request_error": str(exc),
            }

        record["failure_mode"] = infer_failure_mode(record)
        results.append(record)

    payload = build_payload(args, endpoint, run_id, results)
    return payload


def build_payload(
    args: argparse.Namespace,
    endpoint: str,
    run_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_modes = Counter(record["failure_mode"] for record in results)
    source_coverage_count = sum(1 for record in results if record["expected_source_coverage"])
    context_sufficient_count = sum(1 for record in results if record["context_sufficient"] is True)
    routed_to_rag_count = sum(
        1
        for record in results
        if record.get("agent_trace", {}).get("tool_used") == "rag"
    )
    llm_gate_pass_count = sum(
        1
        for record in results
        if record.get("context_metrics", {}).get("llm_relevance_check") is True
    )
    distance_gate_pass_count = sum(
        1
        for record in results
        if record.get("context_metrics", {}).get("distance_gate_passed") is True
    )

    return {
        "run_name": "baseline_single_rag",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.base_url,
        "endpoint": endpoint,
        "question_file": str(args.questions),
        "results_file": str(args.output_json),
        "summary_file": str(args.output_summary),
        "notes": [
            "This baseline intentionally uses the existing FastAPI /ask endpoint.",
            "Only COMPARISON questions are included so future Planner -> Answer Worker -> Synthesizer changes can be compared against the current single-link RAG behavior.",
            "Each question uses a unique session_id to avoid chat-history contamination.",
        ],
        "aggregate": {
            "total_questions": len(results),
            "context_sufficient_count": context_sufficient_count,
            "context_sufficient_rate": round(context_sufficient_count / len(results), 4) if results else 0.0,
            "expected_source_coverage_count": source_coverage_count,
            "expected_source_coverage_rate": round(source_coverage_count / len(results), 4) if results else 0.0,
            "routed_to_rag_count": routed_to_rag_count,
            "llm_gate_pass_count": llm_gate_pass_count,
            "distance_gate_pass_count": distance_gate_pass_count,
            "failure_modes": dict(sorted(failure_modes.items())),
        },
        "results": results,
    }


def format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def render_source_counts(distribution: dict[str, Any]) -> str:
    counts = distribution.get("counts", {})
    if not counts:
        return "none"
    return ", ".join(f"{source}: {count}" for source, count in counts.items())


def build_summary_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    aggregate = payload["aggregate"]
    failure_modes = aggregate.get("failure_modes", {})
    total = aggregate["total_questions"]
    no_context = total - aggregate["context_sufficient_count"]
    paper3_records = [
        record for record in results
        if "Paper3.pdf" in record.get("retrieved_source_distribution", {}).get("counts", {})
    ]
    both_source_records = [
        record for record in results
        if record.get("expected_source_coverage")
    ]

    lines = [
        "# Baseline: Single-Link RAG on COMPARISON Questions",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Created at: `{payload['created_at']}`",
        f"- Endpoint: `{payload['endpoint']}`",
        f"- Question file: `{payload['question_file']}`",
        f"- Result JSON: `{payload['results_file']}`",
        "",
        "## Scope",
        "",
        "This run preserves the current single-link RAG behavior before the planned Planner -> Answer Worker -> Synthesizer upgrade. It uses only the existing FastAPI `/ask` endpoint and focuses on the five COMPARISON questions q008/q009/q010/q019/q020.",
        "",
        "## Aggregate Results",
        "",
        f"- Total COMPARISON questions: {aggregate['total_questions']}",
        f"- Routed to `rag`: {aggregate['routed_to_rag_count']}/{aggregate['total_questions']}",
        f"- Distance gate passed: {aggregate['distance_gate_pass_count']}/{aggregate['total_questions']}",
        f"- LLM relevance gate passed: {aggregate['llm_gate_pass_count']}/{aggregate['total_questions']}",
        f"- Final `context_sufficient = true`: {aggregate['context_sufficient_count']}/{aggregate['total_questions']}",
        f"- Retrieved both expected sources: {aggregate['expected_source_coverage_count']}/{aggregate['total_questions']}",
        "",
        "## Per-Question Snapshot",
        "",
        "| ID | Context sufficient | Expected sources covered | LLM type | LLM gate | Sources | Failure mode |",
        "|---|---:|---:|---|---:|---|---|",
    ]

    for record in results:
        metrics = record.get("context_metrics", {})
        lines.append(
            "| {id} | {suff} | {coverage} | {llm_type} | {llm_gate} | {sources} | `{mode}` |".format(
                id=record["id"],
                suff=format_bool(record.get("context_sufficient")),
                coverage=format_bool(record.get("expected_source_coverage")),
                llm_type=metrics.get("llm_question_type", "n/a"),
                llm_gate=format_bool(metrics.get("llm_relevance_check")),
                sources=render_source_counts(record.get("retrieved_source_distribution", {})),
                mode=record.get("failure_mode", "n/a"),
            )
        )

    lines.extend([
        "",
        "## Main Failure Modes",
        "",
    ])

    if failure_modes:
        for mode, count in failure_modes.items():
            lines.append(f"- `{mode}`: {count}")
    else:
        lines.append("- No failure modes recorded.")

    lines.extend([
        "",
        "### Interpretation",
        "",
        f"- All {total} COMPARISON questions stayed on the current single-link workflow: `choose_tool -> execute_tool -> generate_answer`.",
        f"- {aggregate['distance_gate_pass_count']}/{total} passed the distance gate, but {aggregate['llm_gate_pass_count']}/{total} passed the typed LLM relevance gate. This means vector distance looks good even when the retrieved evidence is not sufficient for cross-paper comparison.",
        f"- {aggregate['expected_source_coverage_count']}/{total} retrieved chunks from both expected sources (`Paper1.pdf` and `Paper2.pdf`). The other {total - aggregate['expected_source_coverage_count']} questions missed at least one expected paper.",
        f"- {len(paper3_records)}/{total} questions retrieved `Paper3.pdf` chunks even though the selected baseline questions compare Paper1 and Paper2. This is the clearest retrieval-disambiguation failure in the current setup.",
        f"- {len(both_source_records)}/{total} questions did retrieve both expected papers, but still failed the COMPARISON gate. In those cases, the mixed single retrieval/rerank pass surfaced some cross-paper chunks, but not enough comparable evidence for a supported answer.",
        f"- {no_context}/{total} questions returned the evidence-insufficient refusal instead of a fabricated comparison. That is good safety behavior, but it leaves COMPARISON usability at zero for this baseline.",
        "",
        "The main pattern to watch is whether a COMPARISON question retrieves balanced, comparable evidence from every paper being compared. If the source distribution is dominated by one paper or polluted by unrelated papers, the typed relevance gate correctly rejects the answer. The planned multi-agent path should improve this by forcing per-paper retrieval before synthesis.",
        "",
        "## What This Baseline Enables for A/B",
        "",
        "Future multi-agent runs can be compared against this file on the same questions and endpoint-level fields: final answer, retrieved chunk source coverage, `context_sufficient`, typed relevance-gate outcome, rerank trace, and `agent_trace.workflow`.",
        "",
        "The strongest A/B signals will be: higher Paper1/Paper2 source coverage, fewer unrelated Paper3 chunks, more COMPARISON questions passing the relevance gate, supported final answers instead of blanket refusals, and a trace that shows explicit planning / per-paper worker answers / synthesis rather than one mixed retrieval call.",
    ])

    request_errors = [record for record in results if record.get("request_error")]
    if request_errors:
        lines.extend([
            "",
            "## Request Errors",
            "",
        ])
        for record in request_errors:
            lines.append(f"- `{record['id']}`: {record['request_error']}")

    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any], output_json: Path, output_summary: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    output_summary.write_text(build_summary_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-link RAG COMPARISON baseline through FastAPI /ask."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--timeout", type=int, default=180, help="Per-question HTTP timeout in seconds.")
    parser.add_argument("--run-id", default="", help="Optional stable run id for session ids.")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record API errors and continue instead of failing fast.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    payload = run_eval(args)
    write_outputs(payload, args.output_json, args.output_summary)

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_summary}")

    errors = [record for record in payload["results"] if record.get("request_error")]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
