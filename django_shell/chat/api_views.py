import json

import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from chat.models import ChatSession
from chat.services.chat_service import (
    ask_and_persist,
    create_session,
    serialize_message,
    serialize_session,
)


def _json_body(request) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


@require_GET
def health(request):
    fastapi_status = "unknown"
    try:
        resp = requests.get(f"{settings.FASTAPI_BASE_URL}/docs", timeout=3)
        fastapi_status = "online" if resp.status_code == 200 else "degraded"
    except requests.RequestException:
        fastapi_status = "offline"

    return JsonResponse(
        {
            "status": "ok",
            "django": "online",
            "fastapi": fastapi_status,
            "fastapi_base_url": settings.FASTAPI_BASE_URL,
        }
    )


@require_GET
def session_list_api(request):
    sessions = ChatSession.objects.all().order_by("-updated_at")
    return JsonResponse(
        {"sessions": [serialize_session(s) for s in sessions]},
        safe=False,
    )


@csrf_exempt
@require_http_methods(["GET", "POST"])
def session_collection(request):
    if request.method == "GET":
        return session_list_api(request)

    body = _json_body(request)
    title = (body.get("title") or "").strip()
    session = create_session(title=title)
    return JsonResponse(serialize_session(session), status=201)


@require_GET
def session_detail_api(request, session_id: str):
    try:
        session = ChatSession.objects.get(session_id=session_id)
    except ChatSession.DoesNotExist:
        return JsonResponse({"error": "session not found"}, status=404)
    return JsonResponse(serialize_session(session, include_messages=True))


@csrf_exempt
@require_http_methods(["POST"])
def ask_api(request):
    try:
        body = _json_body(request)
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON"}, status=400)

    session_id = (body.get("session_id") or "").strip()
    question = (body.get("question") or "").strip()
    if not session_id or not question:
        return JsonResponse(
            {"error": "session_id and question are required"},
            status=400,
        )

    try:
        result = ask_and_persist(session_id, question)
        return JsonResponse(result)
    except requests.RequestException as e:
        return JsonResponse(
            {"error": f"FastAPI request failed: {e}"},
            status=502,
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
