import uuid

from django.utils import timezone

from chat.models import ChatMessage, ChatSession
from chat.services.ai_client import ask_ai


def create_session(title: str = "") -> ChatSession:
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    return ChatSession.objects.create(
        session_id=session_id,
        title=title[:200] if title else "",
    )


def serialize_session(session: ChatSession, include_messages: bool = False) -> dict:
    data = {
        "session_id": session.session_id,
        "title": session.title or session.session_id,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "message_count": session.messages.count(),
    }
    if include_messages:
        data["messages"] = [
            serialize_message(m) for m in session.messages.all().order_by("created_at")
        ]
    return data


def serialize_message(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "metadata": message.metadata or {},
        "created_at": message.created_at.isoformat(),
    }


def get_chat_history(session: ChatSession, limit: int = 6) -> list[dict]:
    recent = list(session.messages.all().order_by("created_at"))[-limit:]
    return [{"role": m.role, "content": m.content} for m in recent]


def ask_and_persist(session_id: str, question: str) -> dict:
    session, _ = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={"title": question[:60]},
    )
    if not session.title:
        session.title = question[:60]
        session.save(update_fields=["title", "updated_at"])

    chat_history = get_chat_history(session, limit=6)

    result = ask_ai(
        session_id=session_id,
        question=question,
        chat_history=chat_history,
    )

    ChatMessage.objects.create(session=session, role="user", content=question)
    ChatMessage.objects.create(
        session=session,
        role="assistant",
        content=result.get("answer", ""),
        metadata={
            "agent_trace": result.get("agent_trace", {}),
            "chunks": result.get("chunks", []),
        },
    )
    session.updated_at = timezone.now()
    session.save(update_fields=["updated_at"])

    return {
        "session_id": session_id,
        "question": question,
        "answer": result.get("answer", ""),
        "chunks": result.get("chunks", []),
        "agent_trace": result.get("agent_trace", {}),
    }
