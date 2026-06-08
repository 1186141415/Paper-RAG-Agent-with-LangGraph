from django.urls import path

from chat import api_views

urlpatterns = [
    path("health/", api_views.health, name="api_health"),
    path("sessions/", api_views.session_collection, name="api_sessions"),
    path("sessions/<str:session_id>/", api_views.session_detail_api, name="api_session_detail"),
    path("chat/ask/", api_views.ask_api, name="api_ask"),
]
