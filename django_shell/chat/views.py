from django.shortcuts import render, get_object_or_404, redirect
from .services.ai_client import ask_ai
from .models import ChatSession, ChatMessage


def chat_home(request):
    error = None
    chunks = []
    agent_trace = None
    current_session_id = request.GET.get("session_id", "demo-session").strip()

    if request.method == "POST":
        current_session_id = request.POST.get("session_id", "").strip()
        question = request.POST.get("question", "").strip()

        if current_session_id and question:
            try:
                result = ask_ai(session_id=current_session_id, question=question)
                chunks = result.get("chunks", [])
                agent_trace = result.get("agent_trace")

                session_obj, created = ChatSession.objects.get_or_create(
                    session_id=current_session_id,
                    defaults={"title": question[:60]}
                )

                if not session_obj.title:
                    session_obj.title = question[:60]
                    session_obj.save()

                ChatMessage.objects.create(
                    session=session_obj,
                    role="user",
                    content=question
                )

                ChatMessage.objects.create(
                    session=session_obj,
                    role="assistant",
                    content=result.get("answer", "")
                )

            except Exception as e:
                error = str(e)
        else:
            error = "session_id 和 question 不能为空"

    session_obj = ChatSession.objects.filter(session_id=current_session_id).first()
    messages = []
    if session_obj:
        messages = session_obj.messages.all().order_by("created_at")

    recent_sessions = ChatSession.objects.all().order_by("-updated_at")[:5]

    return render(
        request,
        "chat/chat_home.html",
        {
            "error": error,
            "current_session_id": current_session_id,
            "messages": messages,
            "recent_sessions": recent_sessions,
            "chunks": chunks,
            "agent_trace": agent_trace,
        }
    )


def session_list(request):
    sessions = ChatSession.objects.all().order_by("-updated_at")
    return render(
        request,
        "chat/session_list.html",
        {"sessions": sessions}
    )


def session_detail(request, session_id):
    session_obj = get_object_or_404(ChatSession, session_id=session_id)
    messages = session_obj.messages.all().order_by("created_at")

    return render(
        request,
        "chat/session_detail.html",
        {
            "session_obj": session_obj,
            "messages": messages,
        }
    )
