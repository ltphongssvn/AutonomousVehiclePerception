# AutonomousVehiclePerception/tests/unit/django/test_tasks.py
"""Unit tests for Celery tasks."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from unittest.mock import patch, MagicMock
from fleet.tasks import process_driving_session, health_check_model_service


class TestProcessDrivingSession:
    def test_function_exists(self):
        assert callable(process_driving_session)


class TestHealthCheckModelService:
    def test_function_exists(self):
        assert callable(health_check_model_service)

    @patch("fleet.tasks.requests.get", side_effect=Exception("connection refused"))
    def test_handles_connection_error(self, mock_get):
        result = health_check_model_service()
        assert result is not None
