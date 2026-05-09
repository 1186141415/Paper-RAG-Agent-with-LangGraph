import json
from typing import Any

from app.graph.state import AgentState
from app.llm_utils import client
from app.config import CHAT_MODEL
from app.logger_config import setup_logger


logger = setup_logger()


# 保守兜底规则 路由可能会返回不准，这里加一些关于用论文rag和网络搜索硬约束
def maybe_force_web_search(query: str, decision: dict) -> dict:
    q = query.lower()

    web_keywords = [
        "latest", "recent", "current", "today", "news",
        "web", "online", "internet", "search the web",
        "最新", "最近", "当前", "今天", "联网", "网上", "搜索一下"
    ]

    local_doc_keywords = [
        "paper1", "paper2", "this paper", "the paper",
        "pdf", "document", "论文", "文档"
    ]

    has_web_signal = any(k in q for k in web_keywords)
    looks_like_local_doc = any(k in q for k in local_doc_keywords)

    if has_web_signal and looks_like_local_doc:
        return {
            "tool": "web_search",
            "input": query,
            "reason": "The question contains web/current information signals and should use external search instead of only local documents."
        }

    return decision


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def normalize_decision(decision: dict, query: str, valid_tool_names: set[str]) -> dict:
    if not isinstance(decision, dict):
        return {
            "tool": "llm",
            "input": query,
            "reason": "Router output is not a valid JSON object, so the system falls back to the general LLM tool."
        }

    tool = str(decision.get("tool", "")).strip().lower()

    raw_input = decision.get("input", "")
    tool_input = "" if raw_input is None else str(raw_input).strip()

    raw_reason = decision.get("reason", "")
    reason = "" if raw_reason is None else str(raw_reason).strip()

    if not reason:
        reason = f"The router selected {tool or 'llm'} based on the question type."

    if tool not in valid_tool_names:
        return {
            "tool": "llm",
            "input": query,
            "reason": f"Router selected an invalid tool '{tool}', so the system falls back to the general LLM tool."
        }

    # 对 rag / llm / time / web_search，统一保留原始 query
    # 避免 router 自己把 input 改写成答案或过度改写问题
    if tool in {"rag", "llm", "time", "web_search"}:
        return {
            "tool": tool,
            "input": query,
            "reason": reason
        }

    # calculator 允许保留模型抽出来的数学表达式
    if tool == "calculator":
        if not tool_input:
            return {
                "tool": "calculator",
                "input": query,
                "reason": reason
            }

        return {
            "tool": "calculator",
            "input": tool_input,
            "reason": reason
        }

    return {
        "tool": "llm",
        "input": query,
        "reason": "The router result could not be normalized, so the system falls back to the general LLM tool."
    }


def build_choose_tool_node(tools: list[dict[str, Any]]):
    valid_tool_names = {t["name"] for t in tools}

    def choose_tool_node(state: AgentState) -> AgentState:
        query = state["query"]
        workflow_path = state.get("workflow_path", []) + ["choose_tool"]

        logger.info(f"[choose_tool_node] query: {query}")

        tool_desc = "\n".join([
            f"{t['name']}: {t['description']}" for t in tools
        ])

        prompt = f"""
                    You are a tool router.
            
                    Your job is ONLY to choose the best tool and prepare its input.
                    Do NOT answer the user's question.
                    Do NOT rewrite the user's question into an answer.
                    Return JSON only.
            
                    Available tools:
                    {tool_desc}
            
                    Tool selection guidance:
                    - Use rag for questions about the loaded local papers/documents, such as paper1, paper2, this paper, 
                      the PDF, or document-based comparison/analysis.
                    - Use calculator for clear math calculations.
                    - Use time for current time questions.
                    - Use web_search for questions that explicitly need web information, latest information, recent updates, 
                      current events, online search, or information likely not contained in the local PDFs.
                    - Use llm for general questions that do not need document retrieval, calculation, time, or web search.
            
                    Rules:
                    1. You must return exactly one JSON object.
                    2. JSON format:
                    {{"tool": "...", "input": "...", "reason": "..."}}
                    3. tool must be one of: rag, calculator, time, web_search, llm
                    4. reason must be one short sentence explaining why this tool is selected.
                    5. reason must not answer the user's question.
                    6. For rag, llm, time, and web_search:
                       - input should stay the same as the user's original question
                       - do not invent a new sentence
                    7. For calculator:
                       - input should be the math expression only if you can extract it
                    8. Do not include markdown, explanations, or code fences.
            
                    User question:
                    {query}
                  """

        content = ""
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content

            cleaned = clean_json_text(content)
            raw_decision = json.loads(cleaned)
            decision = normalize_decision(raw_decision, query, valid_tool_names)
            decision = maybe_force_web_search(query, decision)

            logger.info(f"[choose_tool_node] raw decision: {raw_decision}")
            logger.info(f"[choose_tool_node] normalized decision: {decision}")

            return {
                "decision": decision,
                "workflow_path": workflow_path,
            }

        except Exception as e:
            logger.exception("choose_tool_node failed")
            return {
                "decision": {
                    "tool": "llm",
                    "input": query,
                    "reason": "Router failed, so the system falls back to the general LLM tool."
                },
                "error": f"choose_tool_node failed: {str(e)}",
                "workflow_path": workflow_path
            }

    return choose_tool_node


def build_execute_tool_node(tools: list[dict[str, Any]], rag=None):
    def execute_tool_node(state: AgentState) -> AgentState:
        workflow_path = state.get("workflow_path", []) + ["execute_tool"]

        try:
            decision = state["decision"]
            chat_history = state.get("chat_history", [])

            tool_name = decision["tool"]
            tool_input = decision["input"]

            logger.info(f"[execute_tool_node] tool_name: {tool_name}, tool_input: {tool_input}")

            for t in tools:
                if t["name"] == tool_name:
                    if tool_name == "rag":
                        result = t["func"](tool_input, rag, chat_history=chat_history)
                    elif tool_name == "llm":
                        result = t["func"](tool_input, chat_history=chat_history)
                    else:
                        result = t["func"](tool_input)

                    logger.info(f"[execute_tool_node] tool_output: {result}")

                    return {
                        "tool_result": {
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "tool_output": result
                        },
                        "workflow_path": workflow_path
                    }

            logger.warning(f"[execute_tool_node] tool not found: {tool_name}")

            return {
                "tool_result": {
                    "tool_name": "none",
                    "tool_input": tool_input,
                    "tool_output": "No valid tool found."
                },
                "error": f"Tool not found: {tool_name}",
                "workflow_path": workflow_path
            }

        except Exception as e:
            logger.exception("execute_tool_node failed")
            return {
                "tool_result": {
                    "tool_name": "error",
                    "tool_input": "",
                    "tool_output": "Tool execution failed."
                },
                "error": f"execute_tool_node failed: {str(e)}",
                "workflow_path": workflow_path
            }

    return execute_tool_node


def route_after_execute(state: AgentState) -> str:
    if state.get("error") and not state.get("fallback_used"):
        logger.warning("[route_after_execute] error found, route to llm_fallback")
        return "llm_fallback"

    logger.info("[route_after_execute] no error, route to generate_answer")
    return "generate_answer"

def llm_fallback_node(state: AgentState) -> AgentState:
    workflow_path = state.get("workflow_path", []) + ["llm_fallback"]

    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    previous_error = state.get("error", "")
    retry_count = state.get("retry_count", 0) + 1

    logger.warning(f"[llm_fallback_node] fallback triggered. previous_error: {previous_error}")

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "The previous tool execution failed, so you should answer the user directly. "
                    "If the question requires document evidence or external data that is unavailable, "
                    "clearly state the limitation instead of making unsupported claims."
                )
            }
        ]

        if chat_history:
            messages.extend(chat_history)

        messages.append({
            "role": "user",
            "content": query
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )

        answer = response.choices[0].message.content

        return {
            "tool_result": {
                "tool_name": "llm_fallback",
                "tool_input": query,
                "tool_output": answer
            },
            "fallback_used": True,
            "retry_count": retry_count,
            "error": "",
            "workflow_path": workflow_path
        }

    except Exception as e:
        logger.exception("llm_fallback_node failed")
        return {
            "tool_result": {
                "tool_name": "llm_fallback_error",
                "tool_input": query,
                "tool_output": "LLM fallback failed."
            },
            "fallback_used": True,
            "retry_count": retry_count,
            "error": f"llm_fallback_node failed: {str(e)}",
            "workflow_path": workflow_path
        }

def generate_answer_node(state: AgentState) -> AgentState:
    workflow_path = state.get("workflow_path", []) + ["generate_answer"]

    if state.get("error"):
        logger.warning(f"[generate_answer_node] error found in state: {state['error']}")
        return {
            "final_answer": f"系统执行过程中出现问题：{state['error']}",
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "context_sufficient": state.get("context_sufficient"),
            "context_metrics": state.get("context_metrics", {}),
            "workflow_path": workflow_path
        }

    tool_result = state["tool_result"]
    logger.info(f"[generate_answer_node] final_answer: {tool_result['tool_output']}")

    output = tool_result["tool_output"]

    if isinstance(output, dict):
        return {
            "final_answer": output.get("answer", ""),
            "retrieved_chunks": output.get("retrieved_chunks", []),
            "context_sufficient": output.get("context_sufficient"),
            "context_metrics": output.get("context_metrics", {}),
            "workflow_path": workflow_path
        }

    return {
        "final_answer": str(output),
        "retrieved_chunks": [],
        "context_sufficient": state.get("context_sufficient"),
        "context_metrics": state.get("context_metrics", {}),
        "workflow_path": workflow_path
    }
