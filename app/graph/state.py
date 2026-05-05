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
    # 例如:
    # {
    #   "tool": "rag",
    #   "input": "what is ...",
    #   "reason": "The question requires evidence from uploaded papers."
    # }
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

    # RAG 检索到的上下文是否足够支撑回答
    context_sufficient: bool

    # context sufficiency 的轻量判断指标
    # 例如 best_distance / avg_top_distance / threshold / reason
    context_metrics: dict[str, Any]

    # 是否使用过 LLM fallback
    fallback_used: bool

    # fallback / retry 次数
    retry_count: int

    # 实际执行过的工作流节点路径
    # 例如: ["choose_tool", "execute_tool", "llm_fallback", "generate_answer"]
    workflow_path: list[str]

    # 预留错误字段，后面做异常兜底会用到
    error: str
