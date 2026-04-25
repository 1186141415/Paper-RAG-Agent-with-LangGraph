from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # 当前会话信息
    session_id: str

    # 当前用户问题
    query: str

    # 历史对话
    chat_history: list[dict[str, str]]

    # 路由决策结果
    # 例如: {"tool": "rag", "input": "what is ..."}
    decision: dict[str, Any]

    # 工具执行结果
    # 例如:
    # {
    #   "tool_name": "rag",
    #   "tool_input": "...",
    #   "tool_output": "..."
    # }
    tool_result: dict[str, Any]

    # 最终返回给用户的答案
    final_answer: str

    retrieved_chunks: list[dict[str, Any]]

    # 预留错误字段，后面做异常兜底会用到
    error: str
