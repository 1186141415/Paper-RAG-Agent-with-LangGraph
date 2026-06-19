# Baseline: Single-Link RAG on COMPARISON Questions

- Run id: `baseline-single-rag-20260619`
- Created at: `2026-06-19T15:18:12.715247+00:00`
- Endpoint: `http://127.0.0.1:8000/ask`
- Question file: `D:\Paper-RAG-Agent-with-LangGraph\eval\baseline_comparison_questions.json`
- Result JSON: `D:\Paper-RAG-Agent-with-LangGraph\eval\results\baseline_single_rag.json`

## Scope

This run preserves the current single-link RAG behavior before the planned Planner -> Answer Worker -> Synthesizer upgrade. It uses only the existing FastAPI `/ask` endpoint and focuses on the five COMPARISON questions q008/q009/q010/q019/q020.

## Aggregate Results

- Total COMPARISON questions: 5
- Routed to `rag`: 5/5
- Distance gate passed: 5/5
- LLM relevance gate passed: 0/5
- Final `context_sufficient = true`: 0/5
- Retrieved both expected sources: 2/5

## Per-Question Snapshot

| ID | Context sufficient | Expected sources covered | LLM type | LLM gate | Sources | Failure mode |
|---|---:|---:|---|---:|---|---|
| q008 | no | yes | COMPARISON | no | Paper1.pdf: 7, Paper2.pdf: 3 | `comparison_blocked_by_typed_relevance_gate` |
| q009 | no | yes | COMPARISON | no | Paper1.pdf: 5, Paper2.pdf: 1, Paper3.pdf: 4 | `comparison_blocked_by_typed_relevance_gate` |
| q010 | no | no | COMPARISON | no | Paper1.pdf: 4, Paper3.pdf: 6 | `comparison_blocked_by_typed_relevance_gate` |
| q019 | no | no | COMPARISON | no | Paper1.pdf: 5, Paper3.pdf: 5 | `comparison_blocked_by_typed_relevance_gate` |
| q020 | no | no | COMPARISON | no | Paper1.pdf: 3, Paper3.pdf: 7 | `comparison_blocked_by_typed_relevance_gate` |

## Main Failure Modes

- `comparison_blocked_by_typed_relevance_gate`: 5

### Interpretation

- All 5 COMPARISON questions stayed on the current single-link workflow: `choose_tool -> execute_tool -> generate_answer`.
- 5/5 passed the distance gate, but 0/5 passed the typed LLM relevance gate. This means vector distance looks good even when the retrieved evidence is not sufficient for cross-paper comparison.
- 2/5 retrieved chunks from both expected sources (`Paper1.pdf` and `Paper2.pdf`). The other 3 questions missed at least one expected paper.
- 4/5 questions retrieved `Paper3.pdf` chunks even though the selected baseline questions compare Paper1 and Paper2. This is the clearest retrieval-disambiguation failure in the current setup.
- 2/5 questions did retrieve both expected papers, but still failed the COMPARISON gate. In those cases, the mixed single retrieval/rerank pass surfaced some cross-paper chunks, but not enough comparable evidence for a supported answer.
- 5/5 questions returned the evidence-insufficient refusal instead of a fabricated comparison. That is good safety behavior, but it leaves COMPARISON usability at zero for this baseline.

The main pattern to watch is whether a COMPARISON question retrieves balanced, comparable evidence from every paper being compared. If the source distribution is dominated by one paper or polluted by unrelated papers, the typed relevance gate correctly rejects the answer. The planned multi-agent path should improve this by forcing per-paper retrieval before synthesis.

## What This Baseline Enables for A/B

Future multi-agent runs can be compared against this file on the same questions and endpoint-level fields: final answer, retrieved chunk source coverage, `context_sufficient`, typed relevance-gate outcome, rerank trace, and `agent_trace.workflow`.

The strongest A/B signals will be: higher Paper1/Paper2 source coverage, fewer unrelated Paper3 chunks, more COMPARISON questions passing the relevance gate, supported final answers instead of blanket refusals, and a trace that shows explicit planning / per-paper worker answers / synthesis rather than one mixed retrieval call.
