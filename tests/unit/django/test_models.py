# AutonomousVehiclePerception/tests/unit/django/test_models.py
"""Unit tests for Django fleet and perception models and serializers."""

import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from fleet.models import Vehicle, DrivingSession
from fleet.serializers import VehicleSerializer, DrivingSessionSerializer
from perception.models import MLModel, DetectionResult, DetectedObject
from perception.serializers import MLModelSerializer, DetectedObjectSerializer


class TestVehicleModel:
    def test_status_choices(self):
        choices = dict(Vehicle.STATUS_CHOICES)
        assert "active" in choices
        assert "maintenance" in choices
        assert "offline" in choices
        assert "testing" in choices

    def test_str_representation(self):
        v = Vehicle(name="TestCar", vin="ABC123")
        assert "TestCar" in str(v)
        assert "ABC123" in str(v)

    def test_default_cameras(self):
        v = Vehicle(name="Test", vin="XYZ")
        assert v.num_cameras == 6

    def test_default_lidar(self):
        v = Vehicle(name="Test", vin="XYZ")
        assert v.has_lidar is True


class TestVehicleSerializer:
    def test_required_fields(self):
        s = VehicleSerializer()
        field_names = set(s.fields.keys())
        assert "name" in field_names
        assert "vin" in field_names
        assert "status" in field_names
        assert "id" in field_names

    def test_field_types(self):
        s = VehicleSerializer()
        from rest_framework import fields as drf_fields

        assert isinstance(s.fields["name"], drf_fields.CharField)
        assert isinstance(s.fields["num_cameras"], drf_fields.IntegerField)
        assert isinstance(s.fields["has_lidar"], drf_fields.BooleanField)


class TestDrivingSessionModel:
    def test_session_id_stored(self):
        ds = DrivingSession(session_id="SES-001")
        assert ds.session_id == "SES-001"

    def test_default_processed(self):
        ds = DrivingSession()
        assert ds.processed is False


class TestDrivingSessionSerializer:
    def test_required_fields(self):
        s = DrivingSessionSerializer()
        field_names = set(s.fields.keys())
        assert "session_id" in field_names
        assert "vehicle" in field_names or "vehicle_id" in field_names
        assert "processed" in field_names


class TestMLModel:
    def test_model_type_choices(self):
        choices = dict(MLModel.MODEL_TYPE_CHOICES)
        assert "cnn_2d" in choices
        assert "cnn_3d" in choices
        assert "fpn_resnet" in choices

    def test_status_choices(self):
        choices = dict(MLModel.STATUS_CHOICES)
        assert "training" in choices
        assert "deployed" in choices
        assert "archived" in choices

    def test_str_contains_name_version(self):
        m = MLModel(name="TestModel", version="1.0")
        result = str(m)
        assert "TestModel" in result
        assert "1.0" in result


class TestMLModelSerializer:
    def test_required_fields(self):
        s = MLModelSerializer()
        field_names = set(s.fields.keys())
        assert "name" in field_names
        assert "model_type" in field_names
        assert "version" in field_names
        assert "status" in field_names

    def test_field_types(self):
        s = MLModelSerializer()
        from rest_framework import fields as drf_fields

        assert isinstance(s.fields["name"], drf_fields.CharField)
        assert isinstance(s.fields["version"], drf_fields.CharField)


class TestDetectedObjectSerializer:
    def test_required_fields(self):
        s = DetectedObjectSerializer()
        field_names = set(s.fields.keys())
        assert "object_class" in field_names
        assert "confidence" in field_names
