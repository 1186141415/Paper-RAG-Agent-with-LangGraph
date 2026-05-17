from app.llm_utils import client, get_embedding
from app.vector_store.factory import create_vector_store

from app.config import (
    CHAT_MODEL
)

from app.logger_config import setup_logger

logger = setup_logger()
# 轻量级上下文充分性阈值。
# FAISS IndexFlatL2 为相似度更高的向量返回更小的距离。
# 这些阈值是经验性的，应使用较小的评估集进行调优。
MIN_CONTEXT_CHUNKS = 2
CONTEXT_TOP_N_FOR_AVG = 3
CONTEXT_MAX_BEST_DISTANCE = 2.2
CONTEXT_MAX_AVG_TOP_DISTANCE = 2.4

# 向大型语言模型（LLM）展示的重新排序后的数据块数量。
# 使用3个以上的帮助项，可以避免在广泛性和比较性问题中产生假阴性结果。
RELEVANCE_GATE_PREVIEW_CHUNKS = 5


class RAGSystem:
    def __init__(self, chunks, top_k=20, rerank_k=10, vector_store=None):
        self.chunks = chunks
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.vector_store = vector_store or create_vector_store()

    def build_index(self):
        self.vector_store.build(self.chunks)

    def retrieve(self, query, k=5):
        return self.vector_store.search(query, k)

    def assess_context_sufficiency(self, retrieved_chunks):
        """
        Lightweight context sufficiency check.

        Current rule:
        - enough chunks are retrieved
        - best FAISS distance is below threshold
        - average distance of top chunks is below threshold

        This is not a perfect factuality check. It is a lightweight guardrail
        to avoid treating FAISS top-k results as sufficient only because they exist.
        """
        num_chunks = len(retrieved_chunks)

        metrics = {
            "num_chunks": num_chunks,
            "min_required_chunks": MIN_CONTEXT_CHUNKS,
            "best_distance": None,
            "avg_top_distance": None,
            "max_best_distance": CONTEXT_MAX_BEST_DISTANCE,
            "max_avg_top_distance": CONTEXT_MAX_AVG_TOP_DISTANCE,
            "reason": "",
        }

        if num_chunks < MIN_CONTEXT_CHUNKS:
            metrics["reason"] = "Not enough retrieved chunks."
            logger.info(f"[context_sufficiency] {metrics}")
            return False, metrics

        distances = [
            c.get("distance")
            for c in retrieved_chunks
            if c.get("distance") is not None
        ]

        if not distances:
            metrics["reason"] = "No FAISS distance is available for retrieved chunks."
            logger.warning(f"[context_sufficiency] {metrics}")
            return False, metrics

        sorted_distances = sorted(float(d) for d in distances)
        top_distances = sorted_distances[:min(CONTEXT_TOP_N_FOR_AVG, len(sorted_distances))]

        best_distance = sorted_distances[0]
        avg_top_distance = sum(top_distances) / len(top_distances)

        metrics["best_distance"] = round(best_distance, 4)
        metrics["avg_top_distance"] = round(avg_top_distance, 4)

        context_sufficient = (
                best_distance <= CONTEXT_MAX_BEST_DISTANCE
                and avg_top_distance <= CONTEXT_MAX_AVG_TOP_DISTANCE
        )

        if context_sufficient:
            metrics["reason"] = "Retrieved chunks meet the lightweight distance-based sufficiency rule."
        else:
            metrics["reason"] = "Retrieved chunks exist, but FAISS distances are not strong enough."

        logger.info(f"[context_sufficiency] {metrics}")

        return context_sufficient, metrics

    def assess_context_relevance_with_llm(self, question, retrieved_chunks):
        """
        LLM-based relevance gate with question type awareness.

        FAISS distance only tells which chunks are nearest in vector space.
        This gate checks whether the retrieved passages are useful for the current question.

        Question types:
        - BROAD: overall paper content, main contribution, what the paper does
        - SPECIFIC: concrete method/topic/fact, whether a paper mentions something
        - COMPARISON: differences/comparison between papers or methods
        """
        if not retrieved_chunks:
            return False, {
                "llm_question_type": "UNKNOWN",
                "llm_gate_mode": "hard",
                "llm_relevance_check": False,
                "llm_relevance_verdict": "NO",
                "llm_relevance_reason": "No chunks retrieved.",
                "llm_relevance_error": "",
                "llm_soft_warning": "",
            }

        preview = ""
        preview_chunks = retrieved_chunks[:RELEVANCE_GATE_PREVIEW_CHUNKS]

        for i, c in enumerate(preview_chunks, start=1):
            source = c.get("source", "unknown")
            distance = c.get("distance")
            text = c.get("text", "")

            snippet = text[:350]

            preview += (
                f"[Chunk {i}]\n"
                f"Source: {source}\n"
                f"Distance: {distance}\n"
                f"Text: {snippet}\n\n"
            )

        prompt = f"""
                    You are a relevance judge for a RAG system.
                
                    Your job is NOT to answer the user's question.
                    Your job is to judge whether the retrieved passages are useful enough for answering it.
                
                    First classify the question type:
                
                    - BROAD:
                      The question asks about overall paper content, main contribution, summary, motivation,
                      what the paper does, or the general study direction.
                      Examples:
                      "What is the main contribution of this paper?"
                      "What does paper1 do in their study?"
                      "What is this paper about?"
                
                    - SPECIFIC:
                      The question asks about a concrete method, fact, topic, term, formula, dataset, metric,
                      or whether the paper mentions a specific subject.
                      Examples:
                      "What does paper1 say about reinforcement learning?"
                      "Does paper2 mention GAN?"
                      "What attention mechanism is used?"
                
                    - COMPARISON:
                      The question asks about differences, comparison, similarities, or contrast between
                      two or more papers, methods, or systems.
                      Examples:
                      "What is the difference between paper1 and paper2?"
                      "Compare the methods in these two papers."
                
                    Judging rules:
                
                    - For BROAD questions:
                      Reply YES if the passages are clearly from or about the referenced paper(s) and contain
                      abstract, method, contribution, experiment, or conclusion information.
                      Do NOT require the exact words "main contribution" to appear.
                
                    - For SPECIFIC questions:
                      Reply YES only if the passages directly mention or support the concrete topic asked.
                      Reply NO if the passages are about a different topic.
                
                    - For COMPARISON questions:
                      Reply YES only if the passages contain enough information about the compared items.
                      If the passages only cover one side of the comparison, reply NO.
                
                    Question:
                    {question}
                
                    Retrieved passages:
                    {preview}
                
                    Return exactly two lines:
                
                    TYPE: BROAD|SPECIFIC|COMPARISON
                    VERDICT: YES|NO - short reason
                  """

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            verdict = response.choices[0].message.content.strip()

            import re

            type_match = re.search(
                r"TYPE:\s*(BROAD|SPECIFIC|COMPARISON)",
                verdict,
                re.IGNORECASE
            )

            verdict_match = re.search(
                r"VERDICT:\s*(YES|NO)\s*[-:]\s*(.*)",
                verdict,
                re.IGNORECASE | re.DOTALL
            )

            if not type_match or not verdict_match:
                logger.warning(f"[relevance_gate] unexpected verdict format: {verdict}")

                return True, {
                    "llm_question_type": "UNKNOWN",
                    "llm_gate_mode": "distance_only_fallback",
                    "llm_relevance_check": True,
                    "llm_relevance_verdict": verdict[:200],
                    "llm_relevance_reason": "Unexpected judge output format. Falling back to distance-based result.",
                    "llm_relevance_error": "unexpected_verdict_format",
                    "llm_soft_warning": "LLM relevance judge returned unexpected format.",
                }

            question_type = type_match.group(1).upper()
            yes_or_no = verdict_match.group(1).upper()
            reason = verdict_match.group(2).strip()

            is_relevant = yes_or_no == "YES"

            logger.info(
                f"[relevance_gate] type={question_type}, "
                f"verdict={yes_or_no}, reason={reason[:120]}"
            )

            return is_relevant, {
                "llm_question_type": question_type,
                "llm_gate_mode": "typed_relevance_gate",
                "llm_relevance_check": is_relevant,
                "llm_relevance_verdict": f"{yes_or_no} - {reason}"[:200],
                "llm_relevance_reason": reason[:200],
                "llm_relevance_error": "",
                "llm_soft_warning": "",
            }

        except Exception as e:
            logger.warning(f"[relevance_gate] LLM relevance check failed: {e}")

            return True, {
                "llm_question_type": "UNKNOWN",
                "llm_gate_mode": "distance_only_fallback",
                "llm_relevance_check": True,
                "llm_relevance_verdict": "FALLBACK",
                "llm_relevance_reason": "LLM relevance check failed. Falling back to distance-based result.",
                "llm_relevance_error": str(e),
                "llm_soft_warning": "LLM relevance judge failed, so the system used distance gate only.",
            }

    def rerank(self, query, chunks, return_trace=False):
        prompt = f"""You are a ranking assistant.

                    Query:
                    {query}
                
                    Rank the following passages from most relevant to least relevant.
                
                    Passages:
                  """

        for i, c in enumerate(chunks):
            prompt += f"\n[{i}] {c}\n"

        prompt += (
            "\nReturn ONLY a Python list of indices in sorted order, like [2,0,1]. "
            "No explanation, no code fences."
        )

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw_output = response.choices[0].message.content.strip()

        import ast
        import re

        cleaned = raw_output

        # 1. Clean markdown code fences, such as:
        # ```python
        # [2, 0, 1]
        # ```
        if "```" in cleaned:
            match = re.search(r"```(?:python|json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)

        # 2. Extract first bracket list from prose, such as:
        # "I think the order is [2, 0, 1]."
        bracket_match = re.search(r"\[[\d,\s]+\]", cleaned)
        if bracket_match:
            cleaned = bracket_match.group(0)

        rerank_trace = {
            "rerank_used": False,
            "rerank_fallback": False,
            "rerank_error": "",
            "rerank_raw_output": raw_output[:300],
            "rerank_cleaned_output": cleaned[:300],
            "rerank_indices": [],
        }

        try:
            result = ast.literal_eval(cleaned)

            if not isinstance(result, list):
                raise ValueError(f"Rerank output is not a list: {type(result)}")

            if not all(isinstance(i, int) for i in result):
                raise ValueError(f"Rerank output contains non-integer indices: {result}")

            if not all(0 <= i < len(chunks) for i in result):
                raise ValueError(f"Rerank output contains out-of-range indices: {result}")

            # 去重，同时保留顺序
            seen = set()
            deduped = []
            for i in result:
                if i not in seen:
                    deduped.append(i)
                    seen.add(i)

            # 如果 LLM 没返回完整索引，把缺失的补到后面，避免 best_chunks 数量不稳定
            missing = [i for i in range(len(chunks)) if i not in seen]
            final_indices = deduped + missing

            rerank_trace.update({
                "rerank_used": True,
                "rerank_fallback": False,
                "rerank_error": "",
                "rerank_indices": final_indices,
            })

            logger.info(f"[rerank] success, parsed_indices={final_indices[:self.rerank_k]}")

            if return_trace:
                return final_indices, rerank_trace

            return final_indices

        except Exception as e:
            fallback_indices = list(range(len(chunks)))

            rerank_trace.update({
                "rerank_used": False,
                "rerank_fallback": True,
                "rerank_error": str(e),
                "rerank_indices": fallback_indices,
            })

            logger.warning(
                f"[rerank] FALLBACK to original order. "
                f"reason={e}, raw_output={raw_output[:300]}"
            )

            if return_trace:
                return fallback_indices, rerank_trace

            return fallback_indices

    def ask_with_trace(self, question, chat_history=None):
        if chat_history is None:
            chat_history = []

        retrieved = self.retrieve(question, k=self.top_k)

        # rerank（用 text）
        texts = [c["text"] for c in retrieved]
        sorted_indices, rerank_trace = self.rerank(
            question,
            texts,
            return_trace=True
        )
        best_chunks = [retrieved[i] for i in sorted_indices[:self.rerank_k]]

        retrieved_chunks = []
        for c in best_chunks:
            retrieved_chunks.append({
                "source": c["source"],
                "text": c["text"],
                "distance": c.get("distance"),
                "retrieval_rank": c.get("retrieval_rank"),
            })

        distance_sufficient, context_metrics = self.assess_context_sufficiency(retrieved_chunks)

        context_relevant, relevance_metrics = self.assess_context_relevance_with_llm(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        question_type = relevance_metrics.get("llm_question_type", "UNKNOWN")
        llm_relevance_error = relevance_metrics.get("llm_relevance_error", "")

        context_metrics.update({
            "distance_gate_passed": distance_sufficient,
            "llm_question_type": question_type,
            "llm_gate_mode": relevance_metrics.get("llm_gate_mode"),
            "llm_relevance_check": relevance_metrics.get("llm_relevance_check"),
            "llm_relevance_verdict": relevance_metrics.get("llm_relevance_verdict"),
            "llm_relevance_reason": relevance_metrics.get("llm_relevance_reason"),
            "llm_relevance_error": llm_relevance_error,
            "llm_soft_warning": relevance_metrics.get("llm_soft_warning", ""),
            "rerank_used": rerank_trace.get("rerank_used"),
            "rerank_fallback": rerank_trace.get("rerank_fallback"),
            "rerank_error": rerank_trace.get("rerank_error"),
            "rerank_indices": rerank_trace.get("rerank_indices", [])[:self.rerank_k],
            "rerank_raw_output": rerank_trace.get("rerank_raw_output", ""),
        })

        if not distance_sufficient:
            context_sufficient = False
            context_metrics["final_sufficiency_reason"] = (
                "Context failed the distance-based retrieval gate."
            )

        elif llm_relevance_error:
            # Judge failed or returned unexpected format.
            # Do not hard block the answer. Fall back to distance gate and expose warning.
            context_sufficient = True
            context_metrics["final_sufficiency_reason"] = (
                "LLM relevance judge was unavailable or malformed; using distance gate as fallback."
            )

        elif context_relevant:
            context_sufficient = True
            context_metrics["final_sufficiency_reason"] = (
                "Context passed both the distance gate and the typed LLM relevance gate."
            )

        elif question_type == "BROAD":
            # Broad questions such as "main contribution" can be wrongly rejected by a strict judge.
            # Keep this as a soft warning rather than a hard block.
            context_sufficient = True
            context_metrics["llm_soft_warning"] = (
                "LLM relevance gate flagged low relevance for a BROAD question, "
                "but the system allowed generation because the distance gate passed."
            )
            context_metrics["final_sufficiency_reason"] = (
                "Context passed the distance gate; BROAD-question relevance warning was treated as soft."
            )

        else:
            # SPECIFIC and COMPARISON questions still use the relevance gate as a hard block.
            context_sufficient = False
            context_metrics["final_sufficiency_reason"] = (
                f"Context passed the distance gate but failed the typed LLM relevance gate for {question_type} question."
            )

        if not context_sufficient:
            return {
                "answer": (
                    "当前检索到的论文片段不足以直接支持这个问题的可靠回答。"
                    "系统不会基于不相关或证据不足的片段强行推断。"
                    "建议换一种更具体的问题，或补充包含相关内容的论文。"
                ),
                "retrieved_chunks": retrieved_chunks,
                "context_sufficient": False,
                "context_metrics": context_metrics,
            }

        # 拼 context（加来源）
        context = ""
        for c in best_chunks:
            context += f"[Source: {c['source']}]\n{c['text']}\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer based on context and conversation history. "
                    "If the context is not enough, clearly say the evidence is insufficient."
                )
            }
        ]

        # 保留历史对话
        messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": f"{context}\n\nQuestion: {question}"
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context_sufficient": context_sufficient,
            "context_metrics": context_metrics,
        }
