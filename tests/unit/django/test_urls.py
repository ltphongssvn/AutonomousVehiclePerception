# AutonomousVehiclePerception/tests/unit/django/test_urls.py
"""Unit tests for Django URL configuration."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from django.urls import reverse, resolve
from config.urls import urlpatterns


class TestRootURLs:
    def test_urlpatterns_not_empty(self):
        assert len(urlpatterns) > 0

    def test_admin_url_exists(self):
        paths = [str(p.pattern) for p in urlpatterns]
        assert any("admin" in p for p in paths)

    def test_api_url_exists(self):
        paths = [str(p.pattern) for p in urlpatterns]
        assert any("api" in p for p in paths)


class TestFleetURLs:
    def test_vehicle_list_url(self):
        url = reverse("vehicle-list")
        assert "/api/fleet/vehicles" in url

    def test_session_list_url(self):
        url = reverse("drivingsession-list")
        assert "/api/fleet/sessions" in url

    def test_frame_list_url(self):
        url = reverse("sensorframe-list")
        assert "/api/fleet/frames" in url


class TestPerceptionURLs:
    def test_model_list_url(self):
        url = reverse("mlmodel-list")
        assert "/api/perception/models" in url

    def test_detection_list_url(self):
        url = reverse("detectionresult-list")
        assert "/api/perception/detections" in url

    def test_object_list_url(self):
        url = reverse("detectedobject-list")
        assert "/api/perception/objects" in url
