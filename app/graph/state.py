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

    # 问题类型（BROAD / SPECIFIC / COMPARISON / UNKNOWN）
    # 写入：planner_node；读取：answer_worker_node、synthesizer_node
    question_type: str

    # 子任务列表，每项描述一次按 source 或子问题拆分的检索/回答任务
    # 写入：planner_node；读取：answer_worker_node
    subtasks: list[dict[str, Any]]

    # 各子任务对应的检索结果与局部答案
    # 写入：answer_worker_node；读取：synthesizer_node
    sub_answers: list[dict[str, Any]]

    # 当前知识库中真实存在的论文 source 列表（如 Paper1.pdf）
    # 写入：AgentWorkflow 或 planner_node；读取：planner_node、answer_worker_node
    available_sources: list[str]

    # 证据校验结果，v1 预留，供未来 Evidence Verifier / Citation Checker 使用
    # 写入/读取：v1 暂不使用
    verification: dict[str, Any]
