from django.urls import path
from .views import ChatView, home   # ✅ add home here

urlpatterns = [
    path("", home, name="home"),    # UI
    path("chat/", ChatView.as_view(), name="chat"),  # API
]