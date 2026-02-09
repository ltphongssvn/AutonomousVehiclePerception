# AutonomousVehiclePerception/tests/unit/django/test_tasks_extended.py
"""Extended tests for Celery tasks covering main logic."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from unittest.mock import patch, MagicMock
from fleet.tasks import process_driving_session, health_check_model_service


class TestProcessDrivingSessionLogic:
    @patch("fleet.models.DrivingSession.objects")
    def test_session_not_found(self, mock_qs):
        from fleet.models import DrivingSession

        mock_qs.get.side_effect = DrivingSession.DoesNotExist
        result = process_driving_session(session_id=99999)
        assert result == {"error": "Session 99999 not found"}

    @patch("perception.models.MLModel.objects")
    @patch("fleet.models.DrivingSession.objects")
    def test_no_deployed_model(self, mock_session_qs, mock_model_qs):
        mock_session_qs.get.return_value = MagicMock()
        mock_model_qs.filter.return_value.first.return_value = None
        result = process_driving_session(session_id=1)
        assert "error" in result

    @patch("perception.models.DetectionResult.objects")
    @patch("perception.models.MLModel.objects")
    @patch("fleet.models.DrivingSession.objects")
    def test_session_processed_successfully(self, mock_session_qs, mock_model_qs, mock_det_qs):
        mock_session = MagicMock()
        mock_frame = MagicMock()
        mock_frame.frame_number = 1
        mock_frames_qs = MagicMock()
        mock_frames_qs.__iter__ = MagicMock(return_value=iter([mock_frame]))
        mock_frames_qs.count.return_value = 1
        mock_session.frames.all.return_value = mock_frames_qs
        mock_session_qs.get.return_value = mock_session
        mock_model_qs.filter.return_value.first.return_value = MagicMock()
        mock_det_qs.create.return_value = MagicMock()

        result = process_driving_session(session_id=1)
        assert result["frames_processed"] == 1
        mock_session.save.assert_called_once()

    @patch("perception.models.DetectionResult.objects")
    @patch("perception.models.MLModel.objects")
    @patch("fleet.models.DrivingSession.objects")
    def test_frame_error_continues(self, mock_session_qs, mock_model_qs, mock_det_qs):
        mock_session = MagicMock()
        mock_frame = MagicMock()
        mock_frame.frame_number = 1
        mock_frames_qs = MagicMock()
        mock_frames_qs.__iter__ = MagicMock(return_value=iter([mock_frame]))
        mock_frames_qs.count.return_value = 1
        mock_session.frames.all.return_value = mock_frames_qs
        mock_session_qs.get.return_value = mock_session
        mock_model_qs.filter.return_value.first.return_value = MagicMock()
        mock_det_qs.create.side_effect = Exception("db error")

        result = process_driving_session(session_id=1)
        assert result["frames_processed"] == 0


class TestHealthCheckModelServiceLogic:
    @patch("fleet.tasks.requests.get")
    def test_healthy_service(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_resp
        result = health_check_model_service()
        assert result["status"] == "healthy"

    @patch("fleet.tasks.requests.get", side_effect=Exception("timeout"))
    def test_unhealthy_service(self, mock_get):
        result = health_check_model_service()
        assert result["status"] == "unhealthy"
