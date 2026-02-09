# AutonomousVehiclePerception/tests/test_models.py
"""Unit tests for PyTorch model definitions."""

import torch
import pytest

from src.model.cnn_2d import PerceptionCNN2D, ConvBlock
from src.model.cnn_3d_voxel import VoxelBackbone3D, ConvBlock3D
from src.model.fpn_resnet import FPNResNet50, FPNDetector


class TestConvBlock:
    def test_output_shape_no_pool(self):
        block = ConvBlock(3, 64)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 64, 64)

    def test_output_shape_max_pool(self):
        block = ConvBlock(3, 64, pool_type="max")
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)

    def test_output_shape_avg_pool(self):
        block = ConvBlock(3, 64, pool_type="avg")
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)


class TestPerceptionCNN2D:
    def test_output_shape(self):
        model = PerceptionCNN2D(num_classes=10)
        x = torch.randn(1, 3, 480, 640)
        out = model(x)
        assert out.shape == (1, 10, 30, 40)

    def test_batch_inference(self):
        model = PerceptionCNN2D(num_classes=5)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 5

    def test_gradient_flow(self):
        model = PerceptionCNN2D(num_classes=10)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestConvBlock3D:
    def test_output_shape_no_pool(self):
        block = ConvBlock3D(1, 32)
        x = torch.randn(1, 1, 20, 32, 32)
        out = block(x)
        assert out.shape == (1, 32, 20, 32, 32)

    def test_output_shape_max_pool(self):
        block = ConvBlock3D(1, 32, pool_type="max")
        x = torch.randn(1, 1, 20, 32, 32)
        out = block(x)
        assert out.shape == (1, 32, 10, 16, 16)


class TestVoxelBackbone3D:
    def test_output_shape(self):
        model = VoxelBackbone3D(in_channels=1, num_classes=5)
        x = torch.randn(1, 1, 40, 128, 128)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 5

    def test_gradient_flow(self):
        model = VoxelBackbone3D(in_channels=1, num_classes=5)
        x = torch.randn(1, 1, 16, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestFPNResNet50:
    def test_output_levels(self):
        fpn = FPNResNet50(out_channels=256, pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        out = fpn(x)
        assert "p2" in out
        assert "p3" in out
        assert "p4" in out
        assert "p5" in out

    def test_output_channels(self):
        fpn = FPNResNet50(out_channels=128, pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        out = fpn(x)
        for level in ["p2", "p3", "p4", "p5"]:
            assert out[level].shape[1] == 128


class TestFPNDetector:
    def test_output_shape(self):
        model = FPNDetector(num_classes=10, pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 10
