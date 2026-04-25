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
        return {"tool": "web_search", "input": query}

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
        return {"tool": "llm", "input": query}

    tool = str(decision.get("tool", "")).strip().lower()
    tool_input = str(decision.get("input", "")).strip()

    if tool not in valid_tool_names:
        return {"tool": "llm", "input": query}

    # 关键规则：
    # 对 rag / llm / time 这三类，默认都把原始 query 继续传下去，
    # 不允许 router 自己现编一个答案塞进 input。
    # 对 rag / llm / time / web_search，都统一保留原始 query
    if tool in {"rag", "llm", "time", "web_search"}:
        return {"tool": tool, "input": query}

    # calculator 允许保留模型抽出来的表达式
    if tool == "calculator":
        if not tool_input:
            return {"tool": "calculator", "input": query}
        return {"tool": "calculator", "input": tool_input}

    return {"tool": "llm", "input": query}


def build_choose_tool_node(tools: list[dict[str, Any]]):
    valid_tool_names = {t["name"] for t in tools}

    def choose_tool_node(state: AgentState) -> AgentState:
        query = state["query"]
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
                    - Use rag for questions about the loaded local papers/documents, such as paper1, paper2, this paper, the PDF, or document-based comparison/analysis.
                    - Use calculator for clear math calculations.
                    - Use time for current time questions.
                    - Use web_search for questions that explicitly need web information, latest information, recent updates, current events, online search, or information likely not contained in the local PDFs.
                    - Use llm for general questions that do not need document retrieval, calculation, time, or web search.
            
                    Rules:
                    1. You must return exactly one JSON object.
                    2. JSON format:
                    {{"tool": "...", "input": "..."}}
                    3. tool must be one of: rag, calculator, time, web_search, llm
                    4. For rag, llm, time, and web_search:
                       - input should stay the same as the user's original question
                       - do not invent a new sentence
                    5. For calculator:
                       - input should be the math expression only if you can extract it
                    6. Do not include markdown, explanations, or code fences.
            
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
                "decision": decision
            }

        except Exception as e:
            logger.exception("choose_tool_node failed")
            return {
                "decision": {"tool": "llm", "input": query},
                "error": f"choose_tool_node failed: {str(e)}"
            }

    return choose_tool_node


def build_execute_tool_node(tools: list[dict[str, Any]], rag=None):
    def execute_tool_node(state: AgentState) -> AgentState:
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
                        }
                    }

            logger.warning(f"[execute_tool_node] tool not found: {tool_name}")

            return {
                "tool_result": {
                    "tool_name": "none",
                    "tool_input": tool_input,
                    "tool_output": "No valid tool found."
                },
                "error": f"Tool not found: {tool_name}"
            }

        except Exception as e:
            logger.exception("execute_tool_node failed")
            return {
                "tool_result": {
                    "tool_name": "error",
                    "tool_input": "",
                    "tool_output": "Tool execution failed."
                },
                "error": f"execute_tool_node failed: {str(e)}"
            }

    return execute_tool_node


def generate_answer_node(state: AgentState) -> AgentState:
    if state.get("error"):
        logger.warning(f"[generate_answer_node] error found in state: {state['error']}")
        return {
            "final_answer": f"系统执行过程中出现问题：{state['error']}"
        }

    tool_result = state["tool_result"]
    logger.info(f"[generate_answer_node] final_answer: {tool_result['tool_output']}")

    output = tool_result["tool_output"]

    if isinstance(output, dict):
        return {
            "final_answer": output.get("answer", ""),
            "retrieved_chunks": output.get("retrieved_chunks", [])
        }

    return {
        "final_answer": str(output),
        "retrieved_chunks": []
    }

    # return {
    #     "final_answer": str(tool_result["tool_output"])
    # }
