# AutonomousVehiclePerception/tests/unit/fastapi/test_endpoints.py
"""Unit tests for FastAPI model inference service."""

import io
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.fastapi_service.main import app, PredictionResponse, HealthResponse, ModelInfoResponse


def _create_test_image():
    """Create a valid PNG image in bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestResponseSchemas:
    def test_prediction_response_schema(self):
        resp = PredictionResponse(
            model="cnn_2d",
            shape=[1, 10, 30, 40],
            predictions=[0, 1, 2],
            inference_time_ms=5.2,
            device="cpu",
        )
        assert resp.model == "cnn_2d"
        assert len(resp.shape) == 4
        assert isinstance(resp.predictions, list)
        assert resp.inference_time_ms > 0

    def test_health_response_schema(self):
        resp = HealthResponse(
            status="healthy",
            device="cpu",
            models_loaded=["cnn_2d", "cnn_3d"],
            gpu_available=False,
        )
        assert resp.status == "healthy"
        assert len(resp.models_loaded) == 2

    def test_model_info_response_schema(self):
        resp = ModelInfoResponse(
            name="cnn_2d",
            parameters=1000000,
            input_shape="(B, 3, 480, 640)",
            output_shape="(B, 10, 30, 40)",
        )
        assert resp.parameters == 1000000


class TestPreprocessing:
    def test_val_transforms_output_shape(self):
        from src.data.augmentations import get_val_transforms

        tf = get_val_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = tf(image=img, bboxes=[], class_labels=[])
        assert result["image"].shape == (3, 480, 640)
        assert result["image"].dtype == torch.float32

    def test_val_transforms_normalized(self):
        from src.data.augmentations import get_val_transforms

        tf = get_val_transforms(image_size=(480, 640))
        img = np.full((720, 1280, 3), 128, dtype=np.uint8)
        result = tf(image=img, bboxes=[], class_labels=[])
        # After ImageNet normalization, values should not be in 0-255 range
        assert result["image"].max() < 10.0
        assert result["image"].min() > -10.0


class TestEndpoints:
    """Tests using FastAPI TestClient require httpx."""

    def test_app_exists(self):
        assert app is not None
        assert app.title == "AV Perception Model Service"

    def test_app_routes(self):
        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/model-info" in routes
        assert "/predict/2d" in routes
        assert "/predict/3d" in routes
        assert "/predict/fpn" in routes
