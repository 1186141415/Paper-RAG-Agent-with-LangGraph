from app.graph.nodes import build_choose_tool_node
from app.tools import TOOLS

state = {
    "query": "What time is it now?",
    "chat_history": []
}

choose_tool_node = build_choose_tool_node(TOOLS)

result = choose_tool_node(state)
print(result)