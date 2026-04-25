import os
from django.shortcuts import render
from django.conf import settings


DATA_DIR = os.path.abspath(os.path.join(settings.BASE_DIR.parent, "data"))


def upload_page(request):
    message = None
    error = None

    if request.method == "POST":
        file = request.FILES.get("paper_file")

        if file:
            try:
                save_path = os.path.join(DATA_DIR, file.name)

                with open(save_path, "wb+") as f:
                    for chunk in file.chunks():
                        f.write(chunk)

                message = f"File uploaded: {file.name}"

                import requests
                FASTAPI_URL = "http://127.0.0.1:8000"
                try:
                    requests.post(f"{FASTAPI_URL}/reload_kb", timeout=10)
                except Exception as e:
                    print("⚠️ reload_kb failed:", e)

            except Exception as e:
                error = str(e)
        else:
            error = "No file selected"

    files = []
    try:
        for f in os.listdir(DATA_DIR):
            if f.endswith(".pdf"):
                files.append(f)
    except Exception as e:
        print("list files error:", e)

    return render(
        request,
        "documents/upload.html",
        {
            "message": message,
            "error": error,
            "files": files
        }
    )