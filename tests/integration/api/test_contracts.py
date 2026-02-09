# AutonomousVehiclePerception/tests/integration/api/test_contracts.py
"""API contract tests ensuring Django ↔ FastAPI schema alignment."""

import os
import sys

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src/django_backend"))
django.setup()

from src.fastapi_service.main import PredictionResponse, HealthResponse, ModelInfoResponse
from perception.serializers import DetectionResultSerializer, DetectedObjectSerializer
from fleet.serializers import SensorFrameSerializer


class TestPredictionResponseContract:
    """Verify FastAPI PredictionResponse contains all fields React frontend expects."""

    REQUIRED_FIELDS = {"model", "shape", "predictions", "inference_time_ms", "device"}

    def test_all_required_fields_present(self):
        resp = PredictionResponse(
            model="cnn_2d",
            shape=[1, 10, 30, 40],
            predictions=[0, 1, 2, 3],
            inference_time_ms=12.5,
            device="cpu",
        )
        response_fields = set(resp.model_dump().keys())
        assert self.REQUIRED_FIELDS.issubset(response_fields)

    def test_shape_is_list_of_ints(self):
        resp = PredictionResponse(model="fpn", shape=[1, 10, 120, 160], predictions=[], inference_time_ms=8.0, device="cuda")
        assert isinstance(resp.shape, list)
        assert all(isinstance(s, int) for s in resp.shape)

    def test_predictions_is_list(self):
        resp = PredictionResponse(
            model="cnn_3d", shape=[1, 5, 32, 32], predictions=[0, 1, 4, 2], inference_time_ms=15.0, device="cpu"
        )
        assert isinstance(resp.predictions, list)

    def test_inference_time_positive(self):
        resp = PredictionResponse(model="cnn_2d", shape=[1, 10, 30, 40], predictions=[], inference_time_ms=0.5, device="cpu")
        assert resp.inference_time_ms > 0


class TestHealthResponseContract:
    """Verify HealthResponse contains fields Gradio and React consume."""

    REQUIRED_FIELDS = {"status", "device", "models_loaded", "gpu_available"}

    def test_all_required_fields_present(self):
        resp = HealthResponse(status="healthy", device="cpu", models_loaded=["cnn_2d"], gpu_available=False)
        assert self.REQUIRED_FIELDS.issubset(set(resp.model_dump().keys()))

    def test_models_loaded_is_list(self):
        resp = HealthResponse(status="healthy", device="cpu", models_loaded=["cnn_2d", "fpn"], gpu_available=False)
        assert isinstance(resp.models_loaded, list)


class TestDjangoFastAPIAlignment:
    """Verify Django serializer fields align with what FastAPI produces."""

    def test_detection_result_has_inference_time(self):
        s = DetectionResultSerializer()
        assert "inference_time_ms" in s.fields

    def test_detected_object_has_confidence(self):
        s = DetectedObjectSerializer()
        fields = set(s.fields.keys())
        assert "object_class" in fields
        assert "confidence" in fields

    def test_detected_object_has_bbox_fields(self):
        s = DetectedObjectSerializer()
        fields = set(s.fields.keys())
        bbox_fields = {"bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"}
        assert bbox_fields.issubset(fields)

    def test_sensor_frame_has_paths(self):
        s = SensorFrameSerializer()
        fields = set(s.fields.keys())
        assert "camera_image_path" in fields
        assert "lidar_path" in fields
