# tools.py

from app.rag_system import RAGSystem
from app.mcp_tools import web_search_tool
from datetime import datetime
from app.llm_utils import client

from app.config import (
    CHAT_MODEL
)


def rag_tool(query, rag: RAGSystem, chat_history=None):
    return rag.ask_with_trace(query, chat_history=chat_history)


def calculator_tool(expression):
    """
    Safe calculator tool for basic arithmetic.

    Supported:
    - numbers
    - +, -, *, /
    - parentheses
    - decimal points
    - spaces

    This intentionally avoids raw eval on unrestricted user input.
    """
    import ast
    import operator as op

    allowed_operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)
            if operator_type not in allowed_operators:
                raise ValueError("Unsupported operator.")
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_operators[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)
            if operator_type not in allowed_operators:
                raise ValueError("Unsupported unary operator.")
            operand = eval_node(node.operand)
            return allowed_operators[operator_type](operand)

        raise ValueError("Invalid expression.")

    try:
        tree = ast.parse(str(expression), mode="eval")
        result = eval_node(tree)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return str(result)

    except ZeroDivisionError:
        return "Invalid expression: division by zero."

    except Exception:
        return "Invalid expression. Only basic arithmetic is supported."


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
