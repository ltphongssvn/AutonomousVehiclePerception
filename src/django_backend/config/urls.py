# AutonomousVehiclePerception/src/django_backend/config/urls.py
"""Root URL configuration for AV Perception Django project."""

from django.contrib import admin
from django.urls import include, path
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response


@api_view(["GET"])
@permission_classes([AllowAny])
def api_root(request):
    """API root — lists available endpoints."""
    return Response(
        {
            "fleet": request.build_absolute_uri("/api/fleet/"),
            "perception": request.build_absolute_uri("/api/perception/"),
            "admin": request.build_absolute_uri("/admin/"),
        }
    )


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api_root, name="api-root"),
    path("api/fleet/", include("fleet.urls")),
    path("api/perception/", include("perception.urls")),
    path("api-auth/", include("rest_framework.urls")),
]
