import json
from typing import Any

from app.graph.state import AgentState
from app.llm_utils import client
from app.config import CHAT_MODEL
from app.logger_config import setup_logger

logger = setup_logger()


def build_choose_tool_node(tools: list[dict[str, Any]]):
    def choose_tool_node(state: AgentState) -> AgentState:
        query = state["query"]

        tool_desc = "\n".join([
            f"{t['name']}: {t['description']}" for t in tools
        ])

        prompt = f"""
                    You are an AI agent.
                    
                    Available tools:
                    {tool_desc}
                    
                    User question:
                    {query}
                    
                    Return JSON:
                    {{"tool": "...", "input": "..."}}
                  """

        content = ""
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            decision = json.loads(content)
        except Exception:
            logger.warning(f"Tool decision parse failed: {content}")
            decision = {"tool": "llm", "input": query}

        return {
            "decision": decision
        }

    return choose_tool_node


def build_execute_tool_node(tools: list[dict[str, Any]], rag=None):
    def execute_tool_node(state: AgentState) -> AgentState:
        decision = state["decision"]
        chat_history = state.get("chat_history", [])

        tool_name = decision["tool"]
        tool_input = decision["input"]

        for t in tools:
            if t["name"] == tool_name:
                if tool_name == "rag":
                    result = t["func"](tool_input, rag, chat_history=chat_history)
                elif tool_name == "llm":
                    result = t["func"](tool_input, chat_history=chat_history)
                else:
                    result = t["func"](tool_input)

                return {
                    "tool_result": {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_output": result
                    }
                }

        return {
            "tool_result": {
                "tool_name": "none",
                "tool_input": tool_input,
                "tool_output": "No valid tool found."
            }
        }

    return execute_tool_node


def generate_answer_node(state: AgentState) -> AgentState:
    tool_result = state["tool_result"]

    return {
        "final_answer": str(tool_result["tool_output"])
    }

    # query = state["query"]
    # tool_result = state["tool_result"]
    #
    # prompt = f"""
    #             You are an AI assistant.
    #
    #             The user asked:
    #             {query}
    #
    #             A tool was used:
    #             Tool name: {tool_result['tool_name']}
    #             Tool input: {tool_result['tool_input']}
    #             Tool output: {tool_result['tool_output']}
    #
    #             Now provide a final helpful answer to the user.
    #           """
    #
    # response = client.chat.completions.create(
    #     model=CHAT_MODEL,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    #
    # final_answer = response.choices[0].message.content
    #
    # return {
    #     "final_answer": final_answer
    # }