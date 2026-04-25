# tools.py

from app.rag_system import RAGSystem
from app.mcp_tools import web_search_tool
from datetime import datetime
from app.llm_utils import client

from app.config import (
    CHAT_MODEL
)

#
# def rag_tool(query, rag: RAGSystem, chat_history=None):
#     return rag.ask(query, chat_history=chat_history)

def rag_tool(query, rag: RAGSystem, chat_history=None):
    return rag.ask_with_trace(query, chat_history=chat_history)


def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


def time_tool(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def llm_tool(query, chat_history=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )

    return response.choices[0].message.content


TOOLS = [
    {
        "name": "rag",
        "description": "Use for paper/document questions",
        "func": rag_tool
    },
    {
        "name": "calculator",
        "description": "Use for math calculations",
        "func": calculator_tool
    },
    {
        "name": "time",
        "description": "Get current time",
        "func": time_tool
    },
    {
        "name": "web_search",
        "description": "Use for external web search when local documents are not enough or when real-time web information is needed",
        "func": web_search_tool
    },
    {
        "name": "llm",
        "description": "Use for general questions",
        "func": llm_tool
    }
]