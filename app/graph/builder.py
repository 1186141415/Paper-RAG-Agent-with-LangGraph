from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes import (
    build_choose_tool_node,
    build_execute_tool_node,
    generate_answer_node,
)


def build_agent_graph(tools, rag=None):
    graph_builder = StateGraph(AgentState)

    # 1. 注册节点
    graph_builder.add_node("choose_tool", build_choose_tool_node(tools))
    graph_builder.add_node("execute_tool", build_execute_tool_node(tools, rag=rag))
    graph_builder.add_node("generate_answer", generate_answer_node)

    # 2. 连接流程
    graph_builder.add_edge(START, "choose_tool")
    graph_builder.add_edge("choose_tool", "execute_tool")
    graph_builder.add_edge("execute_tool", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    # 3. 编译 graph
    return graph_builder.compile()