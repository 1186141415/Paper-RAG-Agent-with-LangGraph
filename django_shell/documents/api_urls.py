from django.urls import path

from documents import api_views

urlpatterns = [
    path("", api_views.document_list_api, name="api_documents"),
    path("upload/", api_views.document_upload_api, name="api_upload"),
]
