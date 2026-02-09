# AutonomousVehiclePerception/src/django_backend/perception/urls.py
"""Perception API URL routing."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from perception.views import DetectedObjectViewSet, DetectionResultViewSet, MLModelViewSet

router = DefaultRouter()
router.register(r"models", MLModelViewSet)
router.register(r"detections", DetectionResultViewSet)
router.register(r"objects", DetectedObjectViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
