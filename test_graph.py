from app.graph.builder import build_agent_graph
from app.tools import TOOLS

graph = build_agent_graph(TOOLS, rag=None)

state = {
    "session_id": "test_001",
    "query": "What time is it now?",
    "chat_history": []
}

result = graph.invoke(state)

print("=== graph result ===")
print(result)
print("=== final answer ===")
print(result["final_answer"])