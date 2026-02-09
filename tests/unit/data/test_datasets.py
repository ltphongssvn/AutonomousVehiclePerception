# AutonomousVehiclePerception/tests/unit/data/test_datasets.py
"""Unit tests for dataset loaders and augmentations."""

import numpy as np
import torch

from src.data.kitti_dataset import (
    KITTI_CLASSES,
    CLASS_TO_IDX,
    parse_kitti_label,
    voxelize_points,
)
from src.data.nuscenes_dataset import NUSCENES_CLASSES, CAMERA_NAMES
from src.data.augmentations import get_train_transforms, get_val_transforms, get_lidar_augmentations


class TestKITTIConstants:
    def test_class_list(self):
        assert len(KITTI_CLASSES) == 9
        assert "Car" in KITTI_CLASSES
        assert "Pedestrian" in KITTI_CLASSES

    def test_class_to_idx_mapping(self):
        assert CLASS_TO_IDX["Car"] == 0
        assert len(CLASS_TO_IDX) == len(KITTI_CLASSES)


class TestVoxelization:
    def test_output_shape(self):
        points = np.random.randn(5000, 4).astype(np.float32)
        points[:, :3] *= 20
        voxels = voxelize_points(points)
        # Default: z=(-3,1)/0.2=20, y=(-40,40)/0.2=400, x=(-40,40)/0.2=400
        assert voxels.shape == (20, 400, 400)

    def test_custom_voxel_size(self):
        points = np.random.randn(1000, 4).astype(np.float32)
        points[:, :3] *= 10
        voxels = voxelize_points(points, voxel_size=(0.5, 0.5, 0.5))
        # z=(-3,1)/0.5=8, y=(-40,40)/0.5=160, x=(-40,40)/0.5=160
        assert voxels.shape == (8, 160, 160)

    def test_empty_points(self):
        points = np.zeros((0, 4), dtype=np.float32)
        voxels = voxelize_points(points)
        assert voxels.sum() == 0

    def test_all_out_of_range(self):
        points = np.full((100, 4), 100.0, dtype=np.float32)
        voxels = voxelize_points(points)
        assert voxels.sum() == 0

    def test_occupancy_non_negative(self):
        points = np.random.randn(5000, 4).astype(np.float32)
        points[:, :3] *= 20
        voxels = voxelize_points(points)
        assert (voxels >= 0).all()


class TestKITTILabelParsing:
    def test_nonexistent_file(self):
        result = parse_kitti_label("/nonexistent/path.txt")
        assert result == []

    def test_empty_annotations(self, tmp_path):
        label_file = tmp_path / "test.txt"
        label_file.write_text("")
        result = parse_kitti_label(str(label_file))
        assert result == []

    def test_dontcare_filtered(self, tmp_path):
        label_file = tmp_path / "test.txt"
        label_file.write_text("DontCare -1 -1 -10 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10\n")
        result = parse_kitti_label(str(label_file))
        assert result == []

    def test_valid_annotation(self, tmp_path):
        label_file = tmp_path / "test.txt"
        label_file.write_text("Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n")
        result = parse_kitti_label(str(label_file))
        assert len(result) == 1
        assert result[0]["class"] == "Car"
        assert result[0]["class_idx"] == 0
        assert len(result[0]["bbox_2d"]) == 4
        assert result[0]["bbox_2d"][0] < result[0]["bbox_2d"][2]  # x1 < x2
        assert result[0]["bbox_2d"][1] < result[0]["bbox_2d"][3]  # y1 < y2
        assert len(result[0]["dimensions"]) == 3
        assert len(result[0]["location"]) == 3


class TestNuScenesConstants:
    def test_class_list(self):
        assert len(NUSCENES_CLASSES) == 10
        assert "car" in NUSCENES_CLASSES
        assert "pedestrian" in NUSCENES_CLASSES

    def test_camera_names(self):
        assert len(CAMERA_NAMES) == 6
        assert "CAM_FRONT" in CAMERA_NAMES


class TestTrainTransforms:
    def test_output_shape(self):
        tf = get_train_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = tf(image=img, bboxes=[], class_labels=[])
        assert result["image"].shape == (3, 480, 640)

    def test_preserves_bbox_validity(self):
        tf = get_train_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        bboxes = [[200, 300, 500, 600]]
        labels = [0]
        result = tf(image=img, bboxes=bboxes, class_labels=labels)
        for bbox in result["bboxes"]:
            assert bbox[0] < bbox[2], "x_min must be < x_max"
            assert bbox[1] < bbox[3], "y_min must be < y_max"

    def test_label_count_preserved(self):
        tf = get_train_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        bboxes = [[100, 100, 400, 400], [500, 200, 800, 500]]
        labels = [0, 1]
        result = tf(image=img, bboxes=bboxes, class_labels=labels)
        assert len(result["bboxes"]) == len(result["class_labels"])

    def test_output_is_tensor(self):
        tf = get_train_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = tf(image=img, bboxes=[], class_labels=[])
        assert isinstance(result["image"], torch.Tensor)


class TestValTransforms:
    def test_output_shape(self):
        tf = get_val_transforms(image_size=(480, 640))
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = tf(image=img, bboxes=[], class_labels=[])
        assert result["image"].shape == (3, 480, 640)


class TestLidarAugmentations:
    def test_returns_all_augmentations(self):
        augs = get_lidar_augmentations()
        assert "flip_x" in augs
        assert "flip_y" in augs
        assert "noise" in augs
        assert "dropout" in augs

    def test_flip_x_shape_preserved(self):
        augs = get_lidar_augmentations()
        voxels = np.random.rand(20, 64, 64).astype(np.float32)
        result = augs["flip_x"](voxels, p=1.0)
        assert result.shape == voxels.shape

    def test_dropout_reduces_occupancy(self):
        augs = get_lidar_augmentations()
        voxels = np.ones((10, 32, 32), dtype=np.float32)
        result = augs["dropout"](voxels.copy(), drop_rate=0.5, p=1.0)
        assert result.sum() < voxels.sum()

    def test_noise_non_negative(self):
        augs = get_lidar_augmentations()
        voxels = np.ones((10, 32, 32), dtype=np.float32)
        result = augs["noise"](voxels.copy(), p=1.0)
        assert (result >= 0).all()
