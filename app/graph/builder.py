from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.agents import (
    build_answer_worker_node,
    build_planner_node,
    build_synthesizer_node,
)
from app.graph.nodes import (
    build_choose_tool_node,
    build_execute_tool_node,
    generate_answer_node,
    llm_fallback_node,
    route_after_execute,
)


def route_after_choose_tool(state: AgentState) -> str:
    decision = state.get("decision", {})
    if decision.get("tool") == "rag":
        return "planner"
    return "execute_tool"


def build_agent_graph(tools, rag=None):
    graph_builder = StateGraph(AgentState) # 传入状态图的格式要求

    # 1. 注册节点
    graph_builder.add_node("choose_tool", build_choose_tool_node(tools)) # 传入外部依赖，利用工厂函数构建节点
    graph_builder.add_node("planner", build_planner_node())
    graph_builder.add_node("answer_worker", build_answer_worker_node(rag=rag))
    graph_builder.add_node("synthesizer", build_synthesizer_node())
    graph_builder.add_node("execute_tool", build_execute_tool_node(tools, rag=rag)) # 传入外部依赖，利用工厂函数构建节点

    graph_builder.add_node("llm_fallback", llm_fallback_node)

    graph_builder.add_node("generate_answer", generate_answer_node) # 这个节点不需要外部依赖，使用时直接传窗台图就行，所以没有参数



    # 2. 连接流程 相当于开始编排工作流
    graph_builder.add_edge(START, "choose_tool")
    graph_builder.add_conditional_edges(
        "choose_tool",
        route_after_choose_tool,
        {
            "planner": "planner",
            "execute_tool": "execute_tool",
        },
    )

    graph_builder.add_edge("planner", "answer_worker")
    graph_builder.add_edge("answer_worker", "synthesizer")
    graph_builder.add_edge("synthesizer", "generate_answer")

    graph_builder.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {
            "llm_fallback": "llm_fallback",
            "generate_answer": "generate_answer",
        }
    )

    #graph_builder.add_edge("execute_tool", "generate_answer")
    graph_builder.add_edge("llm_fallback", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    # 3. 编译 graph
    return graph_builder.compile()