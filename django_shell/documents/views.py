import os
from urllib import response

from django.shortcuts import render
from django.conf import settings


DATA_DIR = os.path.abspath(os.path.join(settings.BASE_DIR.parent, "data"))


def upload_page(request):
    message = None
    error = None
    reload_result = None

    if request.method == "POST":
        file = request.FILES.get("paper_file")

        if file:
            try:
                if not file.name.lower().endswith(".pdf"):
                    error = "Only PDF files are supported."
                else:
                    os.makedirs(DATA_DIR, exist_ok=True)

                    save_path = os.path.join(DATA_DIR, file.name)

                    with open(save_path, "wb+") as f:
                        for chunk in file.chunks():
                            f.write(chunk)

                    message = f"File uploaded: {file.name}"

                    import requests
                    FASTAPI_URL = "http://127.0.0.1:8000"

                    try:
                        response = requests.post(
                            f"{FASTAPI_URL}/reload_kb",
                            timeout=(5, 180)
                        )

                        if response.status_code == 200:
                            reload_result = response.json()
                        else:
                            error = (
                                f"File uploaded, but reload_kb failed. "
                                f"Status code: {response.status_code}, "
                                f"Response: {response.text}"
                            )

                    except requests.exceptions.ReadTimeout:
                        error = (
                            "File uploaded, but knowledge base reload timed out. "
                            "FastAPI may still be rebuilding the knowledge base in the background. "
                            "Please check the FastAPI terminal logs."
                        )

                    except requests.exceptions.ConnectionError:
                        error = (
                            "File uploaded, but Django could not connect to FastAPI. "
                            "Please make sure FastAPI is running at http://127.0.0.1:8000."
                        )

                    except Exception as e:
                        error = f"File uploaded, but reload_kb request failed: {e}"

            except Exception as e:
                error = str(e)
        else:
            error = "No file selected"

    files = []
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        for f in os.listdir(DATA_DIR):
            if f.lower().endswith(".pdf"):
                files.append(f)

    except Exception as e:
        print("list files error:", e)

    return render(
        request,
        "documents/upload.html",
        {
            "message": message,
            "error": error,
            "reload_result": reload_result,
            "files": files
        }
    )