import requests
from django.conf import settings


def ask_ai(session_id: str, question: str, chat_history: list[dict] | None = None) -> dict:
    payload = {
        "session_id": session_id,
        "question": question,
    }
    # 可选：把 Django 侧持久化的会话历史随请求传给 FastAPI，作为 LLM 上下文，
    # 统一会话来源，避免与 FastAPI 内存历史不同步。
    if chat_history is not None:
        payload["chat_history"] = chat_history

    resp = requests.post(
        f"{settings.FASTAPI_BASE_URL}/ask",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
