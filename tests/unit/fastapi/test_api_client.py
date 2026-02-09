# AutonomousVehiclePerception/tests/unit/fastapi/test_api_client.py
"""FastAPI endpoint tests using TestClient."""

import io
import warnings

import numpy as np
import pytest
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi.testclient import TestClient
from src.fastapi_service.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def png_bytes():
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "device" in data
        assert "models_loaded" in data
        assert "gpu_available" in data

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_is_list(self, client):
        data = client.get("/model-info").json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_model_info_has_fields(self, client):
        data = client.get("/model-info").json()
        for model in data:
            assert "name" in model
            assert "parameters" in model
            assert "input_shape" in model
            assert "output_shape" in model


class TestPredict2DEndpoint:
    def test_predict_2d_returns_200(self, client, png_bytes):
        resp = client.post("/predict/2d", files={"file": ("test.png", png_bytes, "image/png")})
        assert resp.status_code == 200

    def test_predict_2d_response_schema(self, client, png_bytes):
        data = client.post("/predict/2d", files={"file": ("test.png", png_bytes, "image/png")}).json()
        assert "model" in data
        assert "shape" in data
        assert "predictions" in data
        assert "inference_time_ms" in data
        assert "device" in data

    def test_predict_2d_model_name(self, client, png_bytes):
        data = client.post("/predict/2d", files={"file": ("test.png", png_bytes, "image/png")}).json()
        assert data["model"] == "cnn_2d"

    def test_predict_2d_inference_time_positive(self, client, png_bytes):
        data = client.post("/predict/2d", files={"file": ("test.png", png_bytes, "image/png")}).json()
        assert data["inference_time_ms"] > 0


class TestPredictFPNEndpoint:
    def test_predict_fpn_returns_200(self, client, png_bytes):
        resp = client.post("/predict/fpn", files={"file": ("test.png", png_bytes, "image/png")})
        assert resp.status_code == 200

    def test_predict_fpn_model_name(self, client, png_bytes):
        data = client.post("/predict/fpn", files={"file": ("test.png", png_bytes, "image/png")}).json()
        assert data["model"] == "fpn"


class TestPredict3DEndpoint:
    def test_predict_3d_with_npy(self, client, tmp_path):
        points = np.random.randn(5000, 4).astype(np.float32)
        points[:, :3] *= 20
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), points)
        with open(npy_path, "rb") as f:
            resp = client.post("/predict/3d", files={"file": ("test.npy", f, "application/octet-stream")})
        assert resp.status_code == 200

    def test_predict_3d_response_schema(self, client, tmp_path):
        points = np.random.randn(5000, 4).astype(np.float32)
        points[:, :3] *= 20
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), points)
        with open(npy_path, "rb") as f:
            data = client.post("/predict/3d", files={"file": ("test.npy", f, "application/octet-stream")}).json()
        assert data["model"] == "cnn_3d"
        assert "shape" in data
