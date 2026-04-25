from app.graph.builder import build_agent_graph


class AgentWorkflow:
    def __init__(self, tools, rag=None): #传入一些构建编排层的外部依赖
        self.tools = tools
        self.rag = rag
        self.graph = build_agent_graph(tools, rag=rag) # 建立图

    def invoke(self, session_id: str, query: str, chat_history=None):
        if chat_history is None:
            chat_history = []

        state = {
            "session_id": session_id,
            "query": query,
            "chat_history": chat_history,
        }

        result = self.graph.invoke(state)  #返回结果仍是状态图
        return result