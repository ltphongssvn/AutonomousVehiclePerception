# AutonomousVehiclePerception/tests/unit/fastapi/test_gradio.py
"""Unit tests for Gradio demo UI."""

import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.fastapi_service.gradio_app import predict_image, check_health, build_gradio_app, API_URL


class TestPredictImage:
    def test_none_image_returns_message(self):
        result_img, result_text = predict_image(None, "2D CNN")
        assert result_img is None
        assert "upload" in result_text.lower()

    @patch("src.fastapi_service.gradio_app.requests.post")
    def test_successful_prediction(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "cnn_2d",
            "shape": [1, 10, 30, 40],
            "predictions": [0, 1, 2, 3, 0, 1, 2, 3, 0],
            "inference_time_ms": 12.5,
            "device": "cpu",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_img, result_text = predict_image(img, "2D CNN")
        assert "cnn_2d" in result_text
        assert "12.5" in result_text
        assert result_img is not None

    @patch("src.fastapi_service.gradio_app.requests.post")
    def test_fpn_endpoint_selected(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "fpn",
            "shape": [1, 10, 120, 160],
            "predictions": [],
            "inference_time_ms": 8.0,
            "device": "cpu",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predict_image(img, "FPN-ResNet50")
        call_url = mock_post.call_args[0][0] if mock_post.call_args[0] else str(mock_post.call_args)
        assert "fpn" in call_url

    @patch("src.fastapi_service.gradio_app.requests.post", side_effect=Exception("timeout"))
    def test_connection_error_handled(self, mock_post):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_img, result_text = predict_image(img, "2D CNN")
        assert "ERROR" in result_text

    @patch("src.fastapi_service.gradio_app.requests.post")
    def test_empty_predictions(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "cnn_2d",
            "shape": [1, 10, 30, 40],
            "predictions": [],
            "inference_time_ms": 5.0,
            "device": "cpu",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result_img, result_text = predict_image(img, "2D CNN")
        assert "cnn_2d" in result_text


class TestCheckHealth:
    @patch("src.fastapi_service.gradio_app.requests.get")
    def test_healthy_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "healthy",
            "device": "cpu",
            "gpu_available": False,
            "models_loaded": ["cnn_2d", "cnn_3d"],
        }
        mock_get.return_value = mock_resp

        result = check_health()
        assert "healthy" in result
        assert "cnn_2d" in result

    @patch("src.fastapi_service.gradio_app.requests.get", side_effect=Exception("connection refused"))
    def test_connection_error(self, mock_get):
        result = check_health()
        assert "ERROR" in result


class TestBuildApp:
    def test_build_function_exists(self):
        assert callable(build_gradio_app)

    def test_api_url_configured(self):
        assert API_URL == "http://localhost:8001"
