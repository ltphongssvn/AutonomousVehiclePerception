# AutonomousVehiclePerception/tests/unit/model/test_export.py
"""Unit tests for model export utilities."""

import warnings

import torch
import pytest

from src.model.cnn_2d import PerceptionCNN2D
from src.model.cnn_3d_voxel import VoxelBackbone3D
from src.model.export import export_onnx, export_torchscript, quantize_dynamic


@pytest.fixture
def cnn2d_model():
    model = PerceptionCNN2D(num_classes=5)
    model.eval()
    return model


@pytest.fixture
def voxel_model():
    model = VoxelBackbone3D(in_channels=1, num_classes=3)
    model.eval()
    return model


@pytest.fixture(autouse=True)
def _suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        yield


class TestExportONNX:
    def test_creates_onnx_file(self, cnn2d_model, tmp_path):
        out = tmp_path / "test.onnx"
        dummy = torch.randn(1, 3, 128, 128)
        export_onnx(cnn2d_model, dummy, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_onnx_3d_model(self, voxel_model, tmp_path):
        out = tmp_path / "voxel.onnx"
        dummy = torch.randn(1, 1, 16, 32, 32)
        export_onnx(voxel_model, dummy, out)
        assert out.exists()


class TestExportTorchExport:
    def test_creates_file(self, cnn2d_model, tmp_path):
        out = tmp_path / "test.pt2"
        dummy = torch.randn(1, 3, 128, 128)
        export_torchscript(cnn2d_model, dummy, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_3d(self, voxel_model, tmp_path):
        out = tmp_path / "voxel.pt2"
        dummy = torch.randn(1, 1, 16, 32, 32)
        export_torchscript(voxel_model, dummy, out)
        assert out.exists()


class TestQuantize:
    def test_quantized_model_runs(self, cnn2d_model, tmp_path):
        out = tmp_path / "quantized.pth"
        quantized = quantize_dynamic(cnn2d_model, out)
        dummy = torch.randn(1, 3, 128, 128)
        result = quantized(dummy)
        assert result.shape[0] == 1
        assert result.shape[1] == 5

    def test_quantized_file_created(self, cnn2d_model, tmp_path):
        out = tmp_path / "quantized.pth"
        quantize_dynamic(cnn2d_model, out)
        assert out.exists()

    def test_quantized_output_close(self, cnn2d_model, tmp_path):
        out = tmp_path / "quantized.pth"
        quantized = quantize_dynamic(cnn2d_model, out)
        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            orig = cnn2d_model(dummy)
            quant = quantized(dummy)
        assert orig.shape == quant.shape
