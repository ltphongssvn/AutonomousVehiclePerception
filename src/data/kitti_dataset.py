# AutonomousVehiclePerception/src/data/kitti_dataset.py
"""KITTI dataset loader for 2D/3D object detection.

Loads KITTI format data:
- Camera images (left color)
- 3D bounding box labels (location, dimensions, rotation)
- Optional: LiDAR point clouds (.bin) for voxelization

Reference: http://www.cvlibs.net/datasets/kitti/eval_object.php
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# KITTI object classes
KITTI_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(KITTI_CLASSES)}


def parse_kitti_label(label_path):
    """Parse a KITTI label file into structured annotations.

    Each line format:
    type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y [score]

    Returns:
        list of dicts with keys: class, class_idx, bbox_2d, dimensions, location, rotation_y
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_class = parts[0]
            if obj_class == "DontCare":
                continue

            annotations.append(
                {
                    "class": obj_class,
                    "class_idx": CLASS_TO_IDX.get(obj_class, -1),
                    "truncated": float(parts[1]),
                    "occluded": int(parts[2]),
                    "bbox_2d": [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    "dimensions": [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                    "location": [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                    "rotation_y": float(parts[14]),
                }
            )
    return annotations


def load_lidar_points(bin_path):
    """Load KITTI LiDAR point cloud from .bin file.

    Returns:
        np.ndarray of shape (N, 4) — x, y, z, reflectance
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def voxelize_points(points, voxel_size=(0.2, 0.2, 0.2), point_range=(-40, -40, -3, 40, 40, 1)):
    """Convert point cloud to dense voxel grid.

    Args:
        points: (N, 4) point cloud [x, y, z, reflectance]
        voxel_size: (dx, dy, dz) voxel dimensions in meters
        point_range: (x_min, y_min, z_min, x_max, y_max, z_max)

    Returns:
        np.ndarray voxel grid of shape (D, H, W) with occupancy counts
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_range
    dx, dy, dz = voxel_size

    # Filter points within range
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    points = points[mask]

    # Compute voxel indices
    grid_size = (int((z_max - z_min) / dz), int((y_max - y_min) / dy), int((x_max - x_min) / dx))
    voxel_grid = np.zeros(grid_size, dtype=np.float32)

    ix = ((points[:, 0] - x_min) / dx).astype(np.int32)
    iy = ((points[:, 1] - y_min) / dy).astype(np.int32)
    iz = ((points[:, 2] - z_min) / dz).astype(np.int32)

    # Clip to valid range
    ix = np.clip(ix, 0, grid_size[2] - 1)
    iy = np.clip(iy, 0, grid_size[1] - 1)
    iz = np.clip(iz, 0, grid_size[0] - 1)

    # Fill occupancy grid
    np.add.at(voxel_grid, (iz, iy, ix), 1.0)

    return voxel_grid


class KITTIDataset(Dataset):
    """PyTorch Dataset for KITTI object detection.

    Expected directory structure:
    kitti_root/
    ├── training/
    │   ├── image_2/       # Left color camera images (.png)
    │   ├── label_2/       # Object labels (.txt)
    │   ├── velodyne/      # LiDAR point clouds (.bin)
    │   └── calib/         # Calibration files (.txt)
    └── testing/
        ├── image_2/
        ├── velodyne/
        └── calib/

    Args:
        root: Path to KITTI dataset root
        split: 'training' or 'testing'
        transform: Optional image transform (e.g., Albumentations)
        load_lidar: Whether to load and voxelize LiDAR data
    """

    def __init__(self, root, split="training", transform=None, load_lidar=False):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.load_lidar = load_lidar

        self.image_dir = self.root / split / "image_2"
        self.label_dir = self.root / split / "label_2"
        self.lidar_dir = self.root / split / "velodyne"

        # Collect sample IDs from image directory
        if self.image_dir.exists():
            self.sample_ids = sorted([f.stem for f in self.image_dir.glob("*.png")])
        else:
            self.sample_ids = []
            print(f"WARNING: Image directory not found: {self.image_dir}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Load image
        image_path = self.image_dir / f"{sample_id}.png"
        image = Image.open(image_path).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0

        # Load labels (training only)
        annotations = []
        if self.split == "training":
            label_path = self.label_dir / f"{sample_id}.txt"
            annotations = parse_kitti_label(str(label_path))

        # Apply augmentation
        if self.transform:
            bboxes = [ann["bbox_2d"] for ann in annotations]
            class_labels = [ann["class_idx"] for ann in annotations]
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Build target dict
        target = {
            "sample_id": sample_id,
            "annotations": annotations,
            "num_objects": len(annotations),
        }

        # Optional: load and voxelize LiDAR
        if self.load_lidar:
            lidar_path = self.lidar_dir / f"{sample_id}.bin"
            if lidar_path.exists():
                points = load_lidar_points(str(lidar_path))
                voxel_grid = voxelize_points(points)
                target["voxel_grid"] = torch.from_numpy(voxel_grid).unsqueeze(0)  # (1, D, H, W)

        return image, target


if __name__ == "__main__":
    print("KITTI Dataset Loader")
    print(f"Classes: {KITTI_CLASSES}")
    print(f"Class mapping: {CLASS_TO_IDX}")

    # Demo voxelization with random points
    fake_points = np.random.randn(10000, 4).astype(np.float32)
    fake_points[:, :3] *= 20  # Spread across range
    voxels = voxelize_points(fake_points)
    print(f"Voxel grid shape: {voxels.shape}")
    print(f"Occupied voxels: {(voxels > 0).sum()} / {voxels.size}")
