import json
from typing import Any

from app.config import CHAT_MODEL
from app.graph.nodes import clean_json_text
from app.graph.state import AgentState
from app.llm_utils import client
from app.logger_config import setup_logger

logger = setup_logger()

MAX_SUBTASKS = 3

VALID_QUESTION_TYPES = {"BROAD", "SPECIFIC", "COMPARISON"}


def _fallback_plan(query: str) -> dict[str, Any]:
    return {
        "question_type": "UNKNOWN",
        "subtasks": [
            {
                "sub_id": "s1",
                "sub_question": query,
                "target_source": None,
            }
        ],
    }


def _normalize_question_type(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"

    normalized = value.strip().upper()
    if normalized in VALID_QUESTION_TYPES:
        return normalized

    return "UNKNOWN"


def _normalize_target_source(
    target_source: Any,
    available_sources: list[str],
) -> str | None:
    if target_source is None:
        return None

    if not isinstance(target_source, str):
        return None

    source = target_source.strip()
    if not source:
        return None

    if source in available_sources:
        return source

    return None


def _normalize_subtasks(
    raw_subtasks: Any,
    query: str,
    available_sources: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        return _fallback_plan(query)["subtasks"]

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(raw_subtasks):
        if not isinstance(item, dict):
            continue

        sub_id = item.get("sub_id")
        if not isinstance(sub_id, str) or not sub_id.strip():
            sub_id = f"s{index + 1}"
        else:
            sub_id = sub_id.strip()

        sub_question = item.get("sub_question")
        if not isinstance(sub_question, str) or not sub_question.strip():
            sub_question = query
        else:
            sub_question = sub_question.strip()

        target_source = _normalize_target_source(
            item.get("target_source"),
            available_sources,
        )

        normalized.append({
            "sub_id": sub_id,
            "sub_question": sub_question,
            "target_source": target_source,
        })

    if not normalized:
        return _fallback_plan(query)["subtasks"]

    return normalized[:MAX_SUBTASKS]


def _parse_planner_json(content: str) -> dict[str, Any]:
    cleaned = clean_json_text(content)
    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError("Planner output is not a JSON object")

    return data


def _build_planner_prompt(
    query: str,
    available_sources: list[str],
) -> str:
    sources_text = ", ".join(available_sources) if available_sources else "(none)"

    return f"""
You are a planning assistant for a multi-paper RAG system.

Your job is ONLY to classify the question and decompose it into retrieval subtasks.
Do NOT answer the user's question.

Available paper sources in the knowledge base:
{sources_text}

Rules:
1. Return JSON only. No markdown, no code fences, no explanation.
2. question_type must be one of: BROAD, SPECIFIC, COMPARISON
3. subtasks is a list of objects with:
   - sub_id: short id like "s1", "s2"
   - sub_question: a single-paper retrieval question suitable for one document
   - target_source: one value from available_sources, or null if unknown
4. For BROAD questions: usually one subtask with target_source null or the referenced paper
5. For SPECIFIC questions: one subtask focused on the concrete topic
6. For COMPARISON questions:
   - create one subtask per compared paper
   - rewrite the comparison into single-paper questions
   - do NOT copy the original comparison question verbatim into each subtask
   - assign target_source only from available_sources
7. target_source must be null if the paper cannot be determined from available_sources
8. Keep subtasks minimal and retrieval-friendly

Return exactly this JSON shape:
{{
  "question_type": "BROAD|SPECIFIC|COMPARISON",
  "subtasks": [
    {{
      "sub_id": "s1",
      "sub_question": "...",
      "target_source": "Paper1.pdf"
    }}
  ]
}}

User question:
{query}
""".strip()


def _plan_from_llm_content(
    content: str,
    query: str,
    available_sources: list[str],
) -> dict[str, Any]:
    raw = _parse_planner_json(content)
    question_type = _normalize_question_type(raw.get("question_type"))
    subtasks = _normalize_subtasks(raw.get("subtasks"), query, available_sources)

    return {
        "question_type": question_type,
        "subtasks": subtasks,
    }


def build_planner_node():
    def planner_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        available_sources = state.get("available_sources", []) or []
        workflow_path = state.get("workflow_path", []) + ["planner"]

        logger.info(
            "[planner_node] query=%r, available_sources=%s",
            query,
            available_sources,
        )

        prompt = _build_planner_prompt(query, available_sources)

        try:
            messages: list[dict[str, str]] = []
            if chat_history:
                messages.extend(chat_history)
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            plan = _plan_from_llm_content(content, query, available_sources)

            logger.info(
                "[planner_node] question_type=%s, subtasks=%s",
                plan["question_type"],
                len(plan["subtasks"]),
            )

            return {
                "question_type": plan["question_type"],
                "subtasks": plan["subtasks"],
                "workflow_path": workflow_path,
            }

        except Exception as exc:
            logger.warning("[planner_node] fallback plan used: %s", exc)
            fallback = _fallback_plan(query)

            return {
                "question_type": fallback["question_type"],
                "subtasks": fallback["subtasks"],
                "workflow_path": workflow_path,
            }

    return planner_node


def _resolve_worker_subtasks(
    subtasks: Any,
    query: str,
) -> list[dict[str, Any]]:
    if not isinstance(subtasks, list) or not subtasks:
        return _fallback_plan(query)["subtasks"]

    return subtasks


def _failed_sub_answer(
    sub_id: str,
    sub_question: str,
    target_source: str | None,
    error: str,
) -> dict[str, Any]:
    return {
        "sub_id": sub_id,
        "target_source": target_source,
        "sub_question": sub_question,
        "answer": "",
        "retrieved_chunks": [],
        "context_sufficient": False,
        "context_metrics": {},
        "error": error,
    }


def build_answer_worker_node(rag=None):
    def answer_worker_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        chat_history = state.get("chat_history", [])
        subtasks = _resolve_worker_subtasks(state.get("subtasks"), query)
        workflow_path = state.get("workflow_path", []) + ["answer_worker"]

        logger.info(
            "[answer_worker_node] subtasks=%s, rag_available=%s",
            len(subtasks),
            rag is not None,
        )

        sub_answers: list[dict[str, Any]] = []

        for subtask in subtasks:
            sub_id = str(subtask.get("sub_id", ""))
            sub_question = subtask.get("sub_question", query)
            target_source = subtask.get("target_source")

            if rag is None:
                sub_answers.append(_failed_sub_answer(
                    sub_id=sub_id,
                    sub_question=sub_question,
                    target_source=target_source,
                    error="RAG system is not available.",
                ))
                continue

            try:
                result = rag.ask_with_trace(
                    sub_question,
                    chat_history=chat_history,
                    source=target_source,
                )

                sub_answers.append({
                    "sub_id": sub_id,
                    "target_source": target_source,
                    "sub_question": sub_question,
                    "answer": result.get("answer", ""),
                    "retrieved_chunks": result.get("retrieved_chunks", []),
                    "context_sufficient": result.get("context_sufficient", False),
                    "context_metrics": result.get("context_metrics", {}),
                    "error": "",
                })
            except Exception as exc:
                logger.warning(
                    "[answer_worker_node] subtask %s failed: %s",
                    sub_id,
                    exc,
                )
                sub_answers.append(_failed_sub_answer(
                    sub_id=sub_id,
                    sub_question=sub_question,
                    target_source=target_source,
                    error=str(exc),
                ))

        return {
            "sub_answers": sub_answers,
            "workflow_path": workflow_path,
        }

    return answer_worker_node


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "当前检索到的论文片段不足以直接支持这个问题的可靠回答。"
    "系统不会基于不相关或证据不足的片段强行推断。"
    "建议换一种更具体的问题，或补充包含相关内容的论文。"
)


def _merge_retrieved_chunks(sub_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for sub_answer in sub_answers:
        for chunk in sub_answer.get("retrieved_chunks") or []:
            if not isinstance(chunk, dict):
                continue

            source = str(chunk.get("source") or "")
            text = str(chunk.get("text") or "")
            key = (source, text)

            if key in seen:
                continue

            seen.add(key)
            merged.append(chunk)

    return merged


def _build_sub_metrics(sub_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sub_id": sub_answer.get("sub_id", ""),
            "target_source": sub_answer.get("target_source"),
            "context_sufficient": bool(sub_answer.get("context_sufficient")),
            "error": sub_answer.get("error", "") or "",
        }
        for sub_answer in sub_answers
    ]


def _aggregate_context_sufficiency(
    question_type: str,
    sub_answers: list[dict[str, Any]],
) -> tuple[bool, bool]:
    if not sub_answers:
        return False, False

    flags = [bool(item.get("context_sufficient")) for item in sub_answers]
    any_sufficient = any(flags)
    all_sufficient = all(flags)

    if not any_sufficient:
        return False, False

    normalized_type = _normalize_question_type(question_type)
    if normalized_type == "COMPARISON":
        return all_sufficient, not all_sufficient

    if all_sufficient:
        return True, False

    return False, True


def _insufficient_sub_sources(sub_answers: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []

    for sub_answer in sub_answers:
        if sub_answer.get("context_sufficient"):
            continue

        label = sub_answer.get("target_source") or sub_answer.get("sub_id") or "unknown"
        labels.append(str(label))

    return labels


def _format_source_label(target_source: Any) -> str:
    if isinstance(target_source, str) and target_source.strip():
        return f"[Source: {target_source.strip()}]"
    return "[Source: unknown]"


def _deterministic_synthesis_fallback(
    query: str,
    question_type: str,
    sub_answers: list[dict[str, Any]],
    *,
    partial: bool = False,
    insufficient_sources: list[str] | None = None,
) -> str:
    parts: list[str] = []

    for sub_answer in sub_answers:
        label = _format_source_label(sub_answer.get("target_source"))
        answer = str(sub_answer.get("answer") or "").strip()

        if answer:
            parts.append(f"{label}\n{answer}")
            continue

        error = str(sub_answer.get("error") or "").strip()
        if error:
            parts.append(f"{label}\n（该子任务未返回有效答案：{error}）")

    if not parts:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    body = "\n\n".join(parts)
    normalized_type = _normalize_question_type(question_type)

    if normalized_type == "COMPARISON":
        header = f"针对问题：{query}\n\n"
        if partial:
            gaps = "、".join(insufficient_sources or [])
            footer = (
                f"\n\n注意：部分论文证据不足（{gaps}），"
                "以下仅为基于现有证据的部分回答。"
            )
            return header + body + footer

        return header + body

    if len(parts) == 1:
        return parts[0]

    return body


def _build_comparison_synthesis_prompt(
    query: str,
    sub_answers: list[dict[str, Any]],
) -> str:
    evidence_blocks: list[str] = []

    for sub_answer in sub_answers:
        label = _format_source_label(sub_answer.get("target_source"))
        evidence_blocks.append(
            f"{label}\n"
            f"Sub-question: {sub_answer.get('sub_question', '')}\n"
            f"Answer: {sub_answer.get('answer', '')}"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    return f"""
You are a synthesis assistant for a multi-paper RAG system.

Your job is to answer the user's comparison question using only the sub-answers below.
Do not invent facts that are not supported by the provided sub-answers.

Rules:
1. Answer in the same language as the user question when possible.
2. Explicitly cite sources using labels like [Source: Paper1.pdf].
3. Focus on differences and similarities requested by the user.
4. Do not answer with JSON or markdown fences.

User question:
{query}

Sub-answers:
{evidence_text}
""".strip()


def _build_light_synthesis_prompt(
    query: str,
    sub_answer: dict[str, Any],
) -> str:
    label = _format_source_label(sub_answer.get("target_source"))

    return f"""
You are a synthesis assistant for a single-paper RAG answer.

Rewrite the sub-answer clearly for the user question.
Keep the meaning unchanged. Do not invent unsupported facts.
Explicitly keep the source label {label} in the final answer.

User question:
{query}

Sub-answer:
{label}
{sub_answer.get("answer", "")}
""".strip()


def _synthesize_with_llm(
    query: str,
    question_type: str,
    sub_answers: list[dict[str, Any]],
    *,
    partial: bool = False,
    insufficient_sources: list[str] | None = None,
) -> str:
    normalized_type = _normalize_question_type(question_type)

    if normalized_type == "COMPARISON" and not partial:
        prompt = _build_comparison_synthesis_prompt(query, sub_answers)
    elif len(sub_answers) == 1:
        prompt = _build_light_synthesis_prompt(query, sub_answers[0])
    else:
        prompt = _build_comparison_synthesis_prompt(query, sub_answers)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()

    if content:
        return content

    return _deterministic_synthesis_fallback(
        query,
        question_type,
        sub_answers,
        partial=partial,
        insufficient_sources=insufficient_sources,
    )


def _build_synthesizer_result(
    *,
    query: str,
    question_type: str,
    sub_answers: list[dict[str, Any]],
    workflow_path: list[str],
    synthesizer_mode: str,
    context_sufficient: bool,
    partial_answer: bool = False,
    final_answer: str | None = None,
) -> dict[str, Any]:
    sub_metrics = _build_sub_metrics(sub_answers)
    retrieved_chunks = _merge_retrieved_chunks(sub_answers)

    if final_answer is None:
        final_answer = INSUFFICIENT_EVIDENCE_MESSAGE

    context_metrics: dict[str, Any] = {
        "multi_agent": True,
        "question_type": _normalize_question_type(question_type),
        "sub_metrics": sub_metrics,
        "synthesizer_mode": synthesizer_mode,
    }

    if partial_answer:
        context_metrics["partial_answer"] = True

    return {
        "final_answer": final_answer,
        "retrieved_chunks": retrieved_chunks,
        "context_sufficient": context_sufficient,
        "context_metrics": context_metrics,
        "workflow_path": workflow_path,
    }


def build_synthesizer_node():
    def synthesizer_node(state: AgentState) -> AgentState:
        query = state.get("query", "")
        question_type = state.get("question_type", "UNKNOWN") or "UNKNOWN"
        sub_answers = state.get("sub_answers")
        workflow_path = state.get("workflow_path", []) + ["synthesizer"]

        logger.info(
            "[synthesizer_node] question_type=%s, sub_answers=%s",
            question_type,
            len(sub_answers) if isinstance(sub_answers, list) else 0,
        )

        if not isinstance(sub_answers, list) or not sub_answers:
            return _build_synthesizer_result(
                query=query,
                question_type=question_type,
                sub_answers=[],
                workflow_path=workflow_path,
                synthesizer_mode="no_sub_answers",
                context_sufficient=False,
                final_answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            )

        context_sufficient, partial_answer = _aggregate_context_sufficiency(
            question_type,
            sub_answers,
        )
        insufficient_sources = _insufficient_sub_sources(sub_answers)
        normalized_type = _normalize_question_type(question_type)

        if not any(item.get("context_sufficient") for item in sub_answers):
            return _build_synthesizer_result(
                query=query,
                question_type=question_type,
                sub_answers=sub_answers,
                workflow_path=workflow_path,
                synthesizer_mode="insufficient_evidence",
                context_sufficient=False,
                final_answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            )

        try:
            if normalized_type == "COMPARISON" and context_sufficient:
                final_answer = _synthesize_with_llm(
                    query,
                    question_type,
                    sub_answers,
                )
                synthesizer_mode = "llm_comparison"
            elif len(sub_answers) == 1:
                final_answer = _synthesize_with_llm(
                    query,
                    question_type,
                    sub_answers,
                )
                synthesizer_mode = "llm_single"
            elif partial_answer:
                final_answer = _deterministic_synthesis_fallback(
                    query,
                    question_type,
                    sub_answers,
                    partial=True,
                    insufficient_sources=insufficient_sources,
                )
                synthesizer_mode = "deterministic_partial"
            else:
                final_answer = _synthesize_with_llm(
                    query,
                    question_type,
                    sub_answers,
                )
                synthesizer_mode = "llm_multi"

            return _build_synthesizer_result(
                query=query,
                question_type=question_type,
                sub_answers=sub_answers,
                workflow_path=workflow_path,
                synthesizer_mode=synthesizer_mode,
                context_sufficient=context_sufficient,
                partial_answer=partial_answer,
                final_answer=final_answer,
            )

        except Exception as exc:
            logger.warning("[synthesizer_node] fallback synthesis used: %s", exc)
            final_answer = _deterministic_synthesis_fallback(
                query,
                question_type,
                sub_answers,
                partial=partial_answer,
                insufficient_sources=insufficient_sources,
            )

            return _build_synthesizer_result(
                query=query,
                question_type=question_type,
                sub_answers=sub_answers,
                workflow_path=workflow_path,
                synthesizer_mode="deterministic_fallback",
                context_sufficient=context_sufficient,
                partial_answer=partial_answer,
                final_answer=final_answer,
            )

    return synthesizer_node
