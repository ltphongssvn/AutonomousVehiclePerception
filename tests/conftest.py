# AutonomousVehiclePerception/tests/conftest.py
"""Shared fixtures for the AV Perception test suite."""

import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_image_tensor():
    """Standard camera image tensor (B, C, H, W)."""
    return torch.randn(1, 3, 480, 640)


@pytest.fixture
def dummy_batch_tensor():
    """Batch of camera images."""
    return torch.randn(4, 3, 480, 640)


@pytest.fixture
def dummy_voxel_tensor():
    """Standard voxel grid tensor (B, C, D, H, W)."""
    return torch.randn(1, 1, 20, 128, 128)


@pytest.fixture
def dummy_image_numpy():
    """Raw camera image as numpy array (H, W, C)."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def dummy_point_cloud():
    """LiDAR point cloud (N, 4) — x, y, z, intensity."""
    pts = np.random.randn(10000, 4).astype(np.float32)
    pts[:, :3] *= 20
    return pts


@pytest.fixture
def sample_kitti_label(tmp_path):
    """Write a sample KITTI label file and return its path."""
    label = "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
    label += "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n"
    f = tmp_path / "000001.txt"
    f.write_text(label)
    return str(f)


@pytest.fixture
def device():
    """Return available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
