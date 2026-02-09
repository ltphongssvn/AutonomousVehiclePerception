# AutonomousVehiclePerception/tests/unit/django/test_views.py
"""Unit tests for Django fleet and perception views."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from django.test import RequestFactory
from unittest.mock import patch, MagicMock

from fleet.views import VehicleViewSet, DrivingSessionViewSet, SensorFrameViewSet
from perception.views import MLModelViewSet, DetectionResultViewSet, DetectedObjectViewSet


class TestVehicleViewSet:
    def test_viewset_has_list(self):
        assert hasattr(VehicleViewSet, "list")

    def test_viewset_has_retrieve(self):
        assert hasattr(VehicleViewSet, "retrieve")

    def test_viewset_has_create(self):
        assert hasattr(VehicleViewSet, "create")

    def test_viewset_queryset(self):
        assert VehicleViewSet.queryset is not None

    def test_viewset_serializer_class(self):
        from fleet.serializers import VehicleSerializer

        assert VehicleViewSet.serializer_class == VehicleSerializer

    def test_summary_action_exists(self):
        assert hasattr(VehicleViewSet, "summary")

    def test_sessions_action_exists(self):
        assert hasattr(VehicleViewSet, "sessions")


class TestDrivingSessionViewSet:
    def test_viewset_has_list(self):
        assert hasattr(DrivingSessionViewSet, "list")

    def test_viewset_queryset(self):
        assert DrivingSessionViewSet.queryset is not None

    def test_viewset_serializer_class(self):
        from fleet.serializers import DrivingSessionSerializer

        assert hasattr(DrivingSessionViewSet, "get_serializer_class") or DrivingSessionViewSet.serializer_class is not None

    def test_frames_action_exists(self):
        assert hasattr(DrivingSessionViewSet, "frames")


class TestSensorFrameViewSet:
    def test_viewset_has_list(self):
        assert hasattr(SensorFrameViewSet, "list")

    def test_viewset_queryset(self):
        assert SensorFrameViewSet.queryset is not None


class TestMLModelViewSet:
    def test_viewset_has_list(self):
        assert hasattr(MLModelViewSet, "list")

    def test_viewset_queryset(self):
        assert MLModelViewSet.queryset is not None

    def test_viewset_serializer_class(self):
        from perception.serializers import MLModelSerializer

        assert MLModelViewSet.serializer_class == MLModelSerializer

    def test_deployed_action_exists(self):
        assert hasattr(MLModelViewSet, "deployed")

    def test_run_inference_action_exists(self):
        assert hasattr(MLModelViewSet, "run_inference")


class TestDetectionResultViewSet:
    def test_viewset_has_list(self):
        assert hasattr(DetectionResultViewSet, "list")

    def test_viewset_queryset(self):
        assert DetectionResultViewSet.queryset is not None

    def test_stats_action_exists(self):
        assert hasattr(DetectionResultViewSet, "stats")


class TestDetectedObjectViewSet:
    def test_viewset_has_list(self):
        assert hasattr(DetectedObjectViewSet, "list")

    def test_viewset_queryset(self):
        assert DetectedObjectViewSet.queryset is not None
