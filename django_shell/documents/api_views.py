import json
import os

import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from documents.views import DATA_DIR


def _list_pdf_files() -> list[dict]:
    os.makedirs(DATA_DIR, exist_ok=True)
    files = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(DATA_DIR, name)
        files.append(
            {
                "name": name,
                "size_bytes": os.path.getsize(path),
            }
        )
    return files


@require_GET
def document_list_api(request):
    return JsonResponse({"files": _list_pdf_files()})


@csrf_exempt
@require_http_methods(["POST"])
def document_upload_api(request):
    file = request.FILES.get("paper_file")
    if not file:
        return JsonResponse({"error": "No file selected"}, status=400)
    if not file.name.lower().endswith(".pdf"):
        return JsonResponse({"error": "Only PDF files are supported"}, status=400)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, file.name)
        with open(save_path, "wb+") as f:
            for chunk in file.chunks():
                f.write(chunk)

        reload_result = None
        error = None
        try:
            response = requests.post(
                f"{settings.FASTAPI_BASE_URL}/reload_kb",
                timeout=(5, 180),
            )
            if response.status_code == 200:
                reload_result = response.json()
            else:
                error = f"reload_kb failed: HTTP {response.status_code}"
        except requests.exceptions.ReadTimeout:
            error = "reload_kb timed out; check FastAPI logs"
        except requests.exceptions.ConnectionError:
            error = f"cannot connect to FastAPI at {settings.FASTAPI_BASE_URL}"
        except Exception as e:
            error = str(e)

        payload = {
            "message": f"File uploaded: {file.name}",
            "reload_result": reload_result,
            "files": _list_pdf_files(),
        }
        if error:
            payload["error"] = error
            return JsonResponse(payload, status=502)
        return JsonResponse(payload)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
