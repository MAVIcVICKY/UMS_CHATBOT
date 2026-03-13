from django.urls import path, include, re_path
from django.contrib.staticfiles.views import serve
from django.conf import settings

# Static files FIRST so they aren't caught by the empty prefix
urlpatterns = []

if settings.DEBUG:
    urlpatterns += [
        re_path(r'^static/(?P<path>.*)$', serve, {'insecure': True}),
    ]

urlpatterns += [
    path("api/", include("chatbot.urls")),  # API
    path("", include("chatbot.urls")),      # UI at root (last - catches all)
]
