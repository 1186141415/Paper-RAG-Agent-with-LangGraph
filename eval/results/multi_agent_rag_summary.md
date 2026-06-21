# Multi-Agent RAG Eval on COMPARISON Questions

- Run id: `multi-agent-rag-20260621`
- Created at: `2026-06-21T11:45:52.988227+00:00`
- Endpoint: `http://127.0.0.1:8000/ask`
- Question file: `D:\Paper-RAG-Agent-with-LangGraph\eval\baseline_comparison_questions.json`
- Result JSON: `D:\Paper-RAG-Agent-with-LangGraph\eval\results\multi_agent_rag.json`
- Baseline summary: `D:\Paper-RAG-Agent-with-LangGraph\eval\results\baseline_single_rag_summary.md`

## Scope

This run reuses the baseline COMPARISON question set and exercises the current FastAPI `/ask` endpoint after the Planner -> Answer Worker -> Synthesizer path is enabled.

## Aggregate Results

- Total COMPARISON questions: 5
- Workflow entered multi-agent path: 5/5
- `context_metrics.multi_agent == true`: 5/5
- Final `context_sufficient = true`: 3/5
- Retrieved both expected sources: 5/5
- Retrieved `Paper3.pdf` despite Paper1/Paper2 comparison: 0/5
- Any nested relevance gate rejection: 0/5

## Baseline Comparison

| Metric | Baseline single-link | Multi-agent | Delta |
|---|---:|---:|---:|
| Paper3 pollution count | 4 | 0 | -4 |
| Expected source coverage count | 2 | 5 | +3 |
| Context sufficient count | 0 | 3 | +3 |
| Relevance gate rejected count | 5 | 0 | -5 |
| Multi-agent workflow count | 0 | 5 | +5 |

## Per-Question Snapshot

| ID | Sufficient | Expected sources | Multi-agent workflow | Multi-agent metric | Sources | Outcome |
|---|---:|---:|---:|---:|---|---|
| q008 | yes | yes | yes | yes | Paper1.pdf: 10, Paper2.pdf: 10 | `multi_agent_answered_with_expected_sources` |
| q009 | no | yes | yes | yes | Paper1.pdf: 10, Paper2.pdf: 10 | `multi_agent_insufficient_context` |
| q010 | yes | yes | yes | yes | Paper1.pdf: 10, Paper2.pdf: 10 | `multi_agent_answered_with_expected_sources` |
| q019 | no | yes | yes | yes | Paper1.pdf: 10, Paper2.pdf: 10 | `multi_agent_insufficient_context` |
| q020 | yes | yes | yes | yes | Paper1.pdf: 10, Paper2.pdf: 10 | `multi_agent_answered_with_expected_sources` |

## Initial Read

- The key success criterion for the v1 path is that every COMPARISON question visibly enters `planner -> answer_worker -> synthesizer` and exposes `plan` plus `sub_answers` in `agent_trace`.
- Compare quality against the baseline by watching whether Paper3 pollution drops, expected Paper1/Paper2 coverage rises, and final answers become supported instead of being blocked by relevance gates.
- If nested worker-level relevance gates still reject a sub-answer, the summary records that separately from the top-level multi-agent synthesis outcome.
