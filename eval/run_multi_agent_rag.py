from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_baseline_single_rag import (
    expected_sources,
    has_expected_source_coverage,
    load_questions,
    post_json,
    source_distribution,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = ROOT_DIR / "eval" / "baseline_comparison_questions.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "eval" / "results"
DEFAULT_JSON_PATH = DEFAULT_RESULTS_DIR / "multi_agent_rag.json"
DEFAULT_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "multi_agent_rag_summary.md"
DEFAULT_BASELINE_JSON_PATH = DEFAULT_RESULTS_DIR / "baseline_single_rag.json"
DEFAULT_BASELINE_SUMMARY_PATH = DEFAULT_RESULTS_DIR / "baseline_single_rag_summary.md"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MULTI_AGENT_STEPS = ["planner", "answer_worker", "synthesizer"]


def workflow_contains_multi_agent(workflow: list[str]) -> bool:
    return all(step in workflow for step in MULTI_AGENT_STEPS)


def collect_nested_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                found.append(current_value)
            found.extend(collect_nested_values(current_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(collect_nested_values(item, key))
    return found


def relevance_gate_rejected(record: dict[str, Any]) -> bool:
    metrics = record.get("context_metrics", {})
    values = collect_nested_values(metrics, "llm_relevance_check")
    return any(value is False for value in values)


def infer_outcome(record: dict[str, Any]) -> str:
    if record.get("request_error"):
        return "api_request_error"
    if record.get("tool_used") != "rag":
        return "not_routed_to_rag"
    if not record.get("workflow_contains_multi_agent"):
        return "rag_did_not_enter_multi_agent_path"
    if record.get("multi_agent_metric") is not True:
        return "missing_multi_agent_metric"
    if record.get("context_sufficient") is True and record.get("expected_source_coverage"):
        return "multi_agent_answered_with_expected_sources"
    if record.get("context_sufficient") is True:
        return "multi_agent_answered_without_all_expected_sources"
    if relevance_gate_rejected(record):
        return "multi_agent_relevance_gate_rejected"
    return "multi_agent_insufficient_context"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions)
    endpoint = args.base_url.rstrip("/") + "/ask"
    run_id = args.run_id or f"multi-agent-rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
            workflow = agent_trace.get("workflow") or []
            plan = agent_trace.get("plan") or {}
            sub_answers = agent_trace.get("sub_answers") or []
            context_metrics = agent_trace.get("context_metrics") or {}
            distribution = source_distribution(chunks)

            record = {
                "id": question.get("id"),
                "type": question.get("type"),
                "expected_keywords": question.get("expected_keywords", []),
                "expected_source": question.get("expected_source", ""),
                "expected_sources": expected_sources(question),
                "session_id": response.get("session_id", session_id),
                "question": response.get("question", question["question"]),
                "answer": response.get("answer", ""),
                "chunks": chunks,
                "agent_trace": agent_trace,
                "workflow": workflow,
                "plan": plan,
                "sub_answers": sub_answers,
                "context_sufficient": agent_trace.get("context_sufficient"),
                "context_metrics": context_metrics,
                "retrieved_source_distribution": distribution,
                "expected_source_coverage": has_expected_source_coverage(distribution, question),
                "workflow_contains_planner": "planner" in workflow,
                "workflow_contains_answer_worker": "answer_worker" in workflow,
                "workflow_contains_synthesizer": "synthesizer" in workflow,
                "workflow_contains_multi_agent": workflow_contains_multi_agent(workflow),
                "multi_agent_metric": context_metrics.get("multi_agent"),
                "tool_used": agent_trace.get("tool_used"),
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
                "session_id": session_id,
                "question": question["question"],
                "answer": "",
                "chunks": [],
                "agent_trace": {},
                "workflow": [],
                "plan": {},
                "sub_answers": [],
                "context_sufficient": None,
                "context_metrics": {},
                "retrieved_source_distribution": distribution,
                "expected_source_coverage": False,
                "workflow_contains_planner": False,
                "workflow_contains_answer_worker": False,
                "workflow_contains_synthesizer": False,
                "workflow_contains_multi_agent": False,
                "multi_agent_metric": None,
                "tool_used": "",
                "latency_ms": elapsed_ms,
                "request_error": str(exc),
            }

        record["relevance_gate_rejected"] = relevance_gate_rejected(record)
        record["outcome"] = infer_outcome(record)
        results.append(record)

    return build_payload(args, endpoint, run_id, results)


def build_payload(
    args: argparse.Namespace,
    endpoint: str,
    run_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    outcome_counts = Counter(record["outcome"] for record in results)
    paper3_pollution_count = sum(
        1
        for record in results
        if "Paper3.pdf" in record.get("retrieved_source_distribution", {}).get("counts", {})
    )
    context_sufficient_count = sum(1 for record in results if record.get("context_sufficient") is True)
    expected_source_coverage_count = sum(1 for record in results if record.get("expected_source_coverage"))
    workflow_multi_agent_count = sum(1 for record in results if record.get("workflow_contains_multi_agent"))
    multi_agent_metric_count = sum(1 for record in results if record.get("multi_agent_metric") is True)
    relevance_gate_rejected_count = sum(1 for record in results if record.get("relevance_gate_rejected"))

    return {
        "run_name": "multi_agent_rag",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_base_url": args.base_url,
        "endpoint": endpoint,
        "question_file": str(args.questions),
        "results_file": str(args.output_json),
        "summary_file": str(args.output_summary),
        "baseline_json_file": str(args.baseline_json),
        "baseline_summary_file": str(args.baseline_summary),
        "notes": [
            "This run evaluates the current Planner -> Answer Worker -> Synthesizer RAG path through the existing FastAPI /ask endpoint.",
            "It intentionally reuses eval/baseline_comparison_questions.json and writes separate multi_agent_rag outputs.",
            "It does not overwrite baseline_single_rag.json.",
        ],
        "aggregate": {
            "total_questions": total,
            "context_sufficient_count": context_sufficient_count,
            "context_sufficient_rate": round(context_sufficient_count / total, 4) if total else 0.0,
            "expected_source_coverage_count": expected_source_coverage_count,
            "expected_source_coverage_rate": round(expected_source_coverage_count / total, 4) if total else 0.0,
            "paper3_pollution_count": paper3_pollution_count,
            "paper3_pollution_rate": round(paper3_pollution_count / total, 4) if total else 0.0,
            "workflow_multi_agent_count": workflow_multi_agent_count,
            "multi_agent_metric_count": multi_agent_metric_count,
            "relevance_gate_rejected_count": relevance_gate_rejected_count,
            "outcomes": dict(sorted(outcome_counts.items())),
        },
        "baseline_comparison": build_baseline_comparison(args.baseline_json, results),
        "results": results,
    }


def build_baseline_comparison(
    baseline_json_path: Path,
    multi_results: list[dict[str, Any]],
) -> dict[str, Any]:
    multi_total = len(multi_results)
    multi_paper3 = sum(
        1
        for record in multi_results
        if "Paper3.pdf" in record.get("retrieved_source_distribution", {}).get("counts", {})
    )
    multi_expected = sum(1 for record in multi_results if record.get("expected_source_coverage"))
    multi_sufficient = sum(1 for record in multi_results if record.get("context_sufficient") is True)
    multi_rejected = sum(1 for record in multi_results if record.get("relevance_gate_rejected"))
    multi_workflow = sum(1 for record in multi_results if record.get("workflow_contains_multi_agent"))

    comparison = {
        "baseline_available": False,
        "baseline": {},
        "multi_agent": {
            "total_questions": multi_total,
            "paper3_pollution_count": multi_paper3,
            "expected_source_coverage_count": multi_expected,
            "context_sufficient_count": multi_sufficient,
            "relevance_gate_rejected_count": multi_rejected,
            "workflow_multi_agent_count": multi_workflow,
        },
        "delta": {},
    }

    if not baseline_json_path.exists():
        return comparison

    baseline = json.loads(baseline_json_path.read_text(encoding="utf-8"))
    baseline_results = baseline.get("results", [])
    baseline_total = len(baseline_results)
    baseline_paper3 = sum(
        1
        for record in baseline_results
        if "Paper3.pdf" in record.get("retrieved_source_distribution", {}).get("counts", {})
    )
    baseline_expected = sum(1 for record in baseline_results if record.get("expected_source_coverage"))
    baseline_sufficient = sum(1 for record in baseline_results if record.get("context_sufficient") is True)
    baseline_rejected = sum(
        1
        for record in baseline_results
        if record.get("context_metrics", {}).get("llm_relevance_check") is False
    )

    comparison["baseline_available"] = True
    comparison["baseline"] = {
        "total_questions": baseline_total,
        "paper3_pollution_count": baseline_paper3,
        "expected_source_coverage_count": baseline_expected,
        "context_sufficient_count": baseline_sufficient,
        "relevance_gate_rejected_count": baseline_rejected,
        "workflow_multi_agent_count": 0,
    }
    comparison["delta"] = {
        "paper3_pollution_count": multi_paper3 - baseline_paper3,
        "expected_source_coverage_count": multi_expected - baseline_expected,
        "context_sufficient_count": multi_sufficient - baseline_sufficient,
        "relevance_gate_rejected_count": multi_rejected - baseline_rejected,
        "workflow_multi_agent_count": multi_workflow,
    }

    return comparison


def render_bool(value: Any) -> str:
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
    aggregate = payload["aggregate"]
    comparison = payload["baseline_comparison"]
    results = payload["results"]
    total = aggregate["total_questions"]

    lines = [
        "# Multi-Agent RAG Eval on COMPARISON Questions",
        "",
        f"- Run id: `{payload['run_id']}`",
        f"- Created at: `{payload['created_at']}`",
        f"- Endpoint: `{payload['endpoint']}`",
        f"- Question file: `{payload['question_file']}`",
        f"- Result JSON: `{payload['results_file']}`",
        f"- Baseline summary: `{payload['baseline_summary_file']}`",
        "",
        "## Scope",
        "",
        "This run reuses the baseline COMPARISON question set and exercises the current FastAPI `/ask` endpoint after the Planner -> Answer Worker -> Synthesizer path is enabled.",
        "",
        "## Aggregate Results",
        "",
        f"- Total COMPARISON questions: {total}",
        f"- Workflow entered multi-agent path: {aggregate['workflow_multi_agent_count']}/{total}",
        f"- `context_metrics.multi_agent == true`: {aggregate['multi_agent_metric_count']}/{total}",
        f"- Final `context_sufficient = true`: {aggregate['context_sufficient_count']}/{total}",
        f"- Retrieved both expected sources: {aggregate['expected_source_coverage_count']}/{total}",
        f"- Retrieved `Paper3.pdf` despite Paper1/Paper2 comparison: {aggregate['paper3_pollution_count']}/{total}",
        f"- Any nested relevance gate rejection: {aggregate['relevance_gate_rejected_count']}/{total}",
        "",
        "## Baseline Comparison",
        "",
    ]

    if comparison.get("baseline_available"):
        baseline = comparison["baseline"]
        multi = comparison["multi_agent"]
        delta = comparison["delta"]
        lines.extend([
            "| Metric | Baseline single-link | Multi-agent | Delta |",
            "|---|---:|---:|---:|",
            f"| Paper3 pollution count | {baseline['paper3_pollution_count']} | {multi['paper3_pollution_count']} | {delta['paper3_pollution_count']:+d} |",
            f"| Expected source coverage count | {baseline['expected_source_coverage_count']} | {multi['expected_source_coverage_count']} | {delta['expected_source_coverage_count']:+d} |",
            f"| Context sufficient count | {baseline['context_sufficient_count']} | {multi['context_sufficient_count']} | {delta['context_sufficient_count']:+d} |",
            f"| Relevance gate rejected count | {baseline['relevance_gate_rejected_count']} | {multi['relevance_gate_rejected_count']} | {delta['relevance_gate_rejected_count']:+d} |",
            f"| Multi-agent workflow count | {baseline['workflow_multi_agent_count']} | {multi['workflow_multi_agent_count']} | {delta['workflow_multi_agent_count']:+d} |",
        ])
    else:
        lines.append("- Baseline JSON was not found; comparison table could not be generated.")

    lines.extend([
        "",
        "## Per-Question Snapshot",
        "",
        "| ID | Sufficient | Expected sources | Multi-agent workflow | Multi-agent metric | Sources | Outcome |",
        "|---|---:|---:|---:|---:|---|---|",
    ])

    for record in results:
        lines.append(
            "| {id} | {sufficient} | {coverage} | {workflow} | {metric} | {sources} | `{outcome}` |".format(
                id=record["id"],
                sufficient=render_bool(record.get("context_sufficient")),
                coverage=render_bool(record.get("expected_source_coverage")),
                workflow=render_bool(record.get("workflow_contains_multi_agent")),
                metric=render_bool(record.get("multi_agent_metric")),
                sources=render_source_counts(record.get("retrieved_source_distribution", {})),
                outcome=record.get("outcome", "n/a"),
            )
        )

    lines.extend([
        "",
        "## Initial Read",
        "",
        "- The key success criterion for the v1 path is that every COMPARISON question visibly enters `planner -> answer_worker -> synthesizer` and exposes `plan` plus `sub_answers` in `agent_trace`.",
        "- Compare quality against the baseline by watching whether Paper3 pollution drops, expected Paper1/Paper2 coverage rises, and final answers become supported instead of being blocked by relevance gates.",
        "- If nested worker-level relevance gates still reject a sub-answer, the summary records that separately from the top-level multi-agent synthesis outcome.",
    ])

    request_errors = [record for record in results if record.get("request_error")]
    if request_errors:
        lines.extend(["", "## Request Errors", ""])
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
        description="Run the multi-agent RAG COMPARISON eval through FastAPI /ask."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON_PATH)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY_PATH)
    parser.add_argument("--timeout", type=int, default=300, help="Per-question HTTP timeout in seconds.")
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
