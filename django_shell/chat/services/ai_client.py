import requests
from django.conf import settings


def ask_ai(session_id: str, question: str) -> dict:
    resp = requests.post(
        f"{settings.FASTAPI_BASE_URL}/ask",
        json={
            "session_id": session_id,
            "question": question,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()