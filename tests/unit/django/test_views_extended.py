# AutonomousVehiclePerception/tests/unit/django/test_views_extended.py
"""Extended view tests covering action methods and request handling."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from unittest.mock import patch, MagicMock
from django.test import RequestFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from rest_framework.test import APIRequestFactory

from fleet.views import VehicleViewSet, DrivingSessionViewSet, SensorFrameViewSet
from perception.views import MLModelViewSet, DetectionResultViewSet


class TestVehicleViewSetActions:
    def test_summary_method(self):
        factory = RequestFactory()
        request = factory.get("/api/fleet/vehicles/summary/")
        view = VehicleViewSet.as_view({"get": "summary"})
        response = view(request)
        assert response.status_code in (200, 401, 403)

    def test_sessions_action_method(self):
        assert hasattr(VehicleViewSet, "sessions")


class TestDrivingSessionViewSetActions:
    def test_get_serializer_class_list(self):
        view = DrivingSessionViewSet()
        view.action = "list"
        assert view.get_serializer_class() is not None

    def test_get_serializer_class_retrieve(self):
        view = DrivingSessionViewSet()
        view.action = "retrieve"
        assert view.get_serializer_class() is not None


class TestMLModelViewSetActions:
    def test_deployed_action(self):
        factory = RequestFactory()
        request = factory.get("/api/perception/models/deployed/")
        view = MLModelViewSet.as_view({"get": "deployed"})
        response = view(request)
        assert response.status_code in (200, 401, 403)

    def _make_inference_request(self, mock_model_type="cnn_2d", file_content=None):
        factory = APIRequestFactory()
        data = {}
        if file_content is not None:
            data["file"] = SimpleUploadedFile("test.png", file_content, content_type="image/png")
        request = factory.post("/api/perception/models/1/run_inference/", data, format="multipart")
        request.user = MagicMock()
        request.user.is_authenticated = True
        return request

    @patch("perception.views.MLModelViewSet.get_object")
    def test_run_inference_no_file(self, mock_get_obj):
        mock_get_obj.return_value = MagicMock(model_type="cnn_2d")
        request = self._make_inference_request()
        view = MLModelViewSet.as_view({"post": "run_inference"})
        response = view(request, pk=1)
        assert response.status_code == 400
        assert "No file" in str(response.data)

    @patch("perception.views.MLModelViewSet.get_object")
    def test_run_inference_unknown_model_type(self, mock_get_obj):
        mock_get_obj.return_value = MagicMock(model_type="unknown_type")
        request = self._make_inference_request(file_content=b"fake")
        view = MLModelViewSet.as_view({"post": "run_inference"})
        response = view(request, pk=1)
        assert response.status_code == 400
        assert "Unknown" in str(response.data)

    @patch("perception.views.requests.post")
    @patch("perception.views.MLModelViewSet.get_object")
    def test_run_inference_success(self, mock_get_obj, mock_post):
        mock_get_obj.return_value = MagicMock(model_type="cnn_2d")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"model": "cnn_2d", "predictions": [1]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        request = self._make_inference_request(file_content=b"fake image")
        view = MLModelViewSet.as_view({"post": "run_inference"})
        response = view(request, pk=1)
        assert response.status_code == 200

    @patch("perception.views.requests.post")
    @patch("perception.views.MLModelViewSet.get_object")
    def test_run_inference_connection_error(self, mock_get_obj, mock_post):
        import requests as req

        mock_get_obj.return_value = MagicMock(model_type="cnn_2d")
        mock_post.side_effect = req.ConnectionError("refused")

        request = self._make_inference_request(file_content=b"fake")
        view = MLModelViewSet.as_view({"post": "run_inference"})
        response = view(request, pk=1)
        assert response.status_code == 503

    @patch("perception.views.requests.post")
    @patch("perception.views.MLModelViewSet.get_object")
    def test_run_inference_timeout(self, mock_get_obj, mock_post):
        import requests as req

        mock_get_obj.return_value = MagicMock(model_type="cnn_2d")
        mock_post.side_effect = req.Timeout("timeout")

        request = self._make_inference_request(file_content=b"fake")
        view = MLModelViewSet.as_view({"post": "run_inference"})
        response = view(request, pk=1)
        assert response.status_code == 504


class TestDetectionResultViewSetActions:
    def test_get_serializer_class_list(self):
        view = DetectionResultViewSet()
        view.action = "list"
        assert view.get_serializer_class() is not None

    def test_get_serializer_class_retrieve(self):
        view = DetectionResultViewSet()
        view.action = "retrieve"
        assert view.get_serializer_class() is not None

    def test_stats_action(self):
        factory = RequestFactory()
        request = factory.get("/api/perception/detections/stats/")
        view = DetectionResultViewSet.as_view({"get": "stats"})
        response = view(request)
        assert response.status_code in (200, 401, 403)
