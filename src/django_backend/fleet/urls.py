# AutonomousVehiclePerception/src/django_backend/fleet/urls.py
"""Fleet management API URL routing."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from fleet.views import DrivingSessionViewSet, SensorFrameViewSet, VehicleViewSet

router = DefaultRouter()
router.register(r"vehicles", VehicleViewSet)
router.register(r"sessions", DrivingSessionViewSet)
router.register(r"frames", SensorFrameViewSet)

urlpatterns = [
    path("", include(router.urls)),
]
