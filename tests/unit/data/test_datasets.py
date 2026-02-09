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


class TestAugmentationsMainBlock:
    """Cover augmentations.py lines 92-147 (lidar aug helpers + main guard)."""

    def test_get_lidar_augmentations_keys(self):
        from src.data.augmentations import get_lidar_augmentations

        augs = get_lidar_augmentations()
        assert "flip_x" in augs
        assert "flip_y" in augs
        assert "noise" in augs
        assert "dropout" in augs

    def test_flip_y(self):
        from src.data.augmentations import get_lidar_augmentations

        augs = get_lidar_augmentations()
        voxel = np.random.rand(20, 128, 128).astype(np.float32)
        flipped = augs["flip_y"](voxel)
        assert flipped.shape == voxel.shape

    def test_noise_returns_array(self):
        from src.data.augmentations import get_lidar_augmentations

        augs = get_lidar_augmentations()
        voxel = np.ones((20, 128, 128), dtype=np.float32) * 0.5
        noisy = augs["noise"](voxel)
        assert noisy.shape == voxel.shape
        assert noisy.dtype == voxel.dtype

    def test_dropout_zeros_some(self):
        from src.data.augmentations import get_lidar_augmentations

        augs = get_lidar_augmentations()
        voxel = np.ones((20, 128, 128), dtype=np.float32)
        dropped = augs["dropout"](voxel)
        assert dropped.sum() <= voxel.sum()


class TestKITTIDatasetClass:
    """Cover kitti_dataset.py Dataset.__init__, __len__, __getitem__."""

    def test_dataset_init_nonexistent(self):
        from src.data.kitti_dataset import KITTIDataset

        ds = KITTIDataset(root="/nonexistent/path", split="training")
        assert str(ds.root) == "/nonexistent/path"

    def test_dataset_has_split(self):
        from src.data.kitti_dataset import KITTIDataset

        ds = KITTIDataset(root="/tmp", split="training")
        assert ds.split == "training"

    def test_dataset_len_empty(self):
        from src.data.kitti_dataset import KITTIDataset

        ds = KITTIDataset(root="/nonexistent", split="training")
        assert len(ds) == 0

    def test_voxelize_with_real_points(self):
        from src.data.kitti_dataset import voxelize_points

        pts = np.array([[0.0, 0.0, 0.0, 1.0], [5.0, 5.0, -1.0, 0.5], [10.0, -10.0, 0.5, 0.8]], dtype=np.float32)
        grid = voxelize_points(pts)
        assert grid.shape[0] == 20
        assert grid.sum() > 0

    def test_dataset_with_mock_files(self, tmp_path):
        from PIL import Image
        from src.data.kitti_dataset import KITTIDataset

        img_dir = tmp_path / "training" / "image_2"
        label_dir = tmp_path / "training" / "label_2"
        velodyne_dir = tmp_path / "training" / "velodyne"
        calib_dir = tmp_path / "training" / "calib"
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        velodyne_dir.mkdir(parents=True)
        calib_dir.mkdir(parents=True)

        # Create dummy image
        img = Image.fromarray(np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8))
        img.save(img_dir / "000000.png")

        # Create dummy label
        (label_dir / "000000.txt").write_text(
            "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
        )

        # Create dummy velodyne
        pts = np.random.randn(100, 4).astype(np.float32)
        pts.tofile(str(velodyne_dir / "000000.bin"))

        ds = KITTIDataset(root=str(tmp_path), split="training")
        assert len(ds) == 1


class TestNuScenesDatasetClass:
    """Cover nuscenes_dataset.py Dataset class."""

    def test_dataset_init(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/nonexistent", version="v1.0-mini")
        assert str(ds.root) == "/nonexistent"

    def test_dataset_has_version(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini")
        assert ds.version == "v1.0-mini"

    def test_dataset_len_empty(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/nonexistent", version="v1.0-mini")
        assert len(ds) == 0

    def test_class_list_complete(self):
        from src.data.nuscenes_dataset import NUSCENES_CLASSES

        assert len(NUSCENES_CLASSES) == 10
        assert "car" in NUSCENES_CLASSES
        assert "pedestrian" in NUSCENES_CLASSES
        assert "bicycle" in NUSCENES_CLASSES


# Append to tests/unit/data/test_datasets.py


class TestNuScenesDatasetMock:
    """Cover nuscenes_dataset.py Dataset __getitem__, transforms, lidar loading."""

    def test_dataset_camera_default(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini")
        assert ds.camera == "CAM_FRONT"

    def test_dataset_custom_camera(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini", camera="CAM_BACK")
        assert ds.camera == "CAM_BACK"

    def test_dataset_load_lidar_default(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini")
        assert ds.load_lidar is False

    def test_dataset_load_lidar_true(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini", load_lidar=True)
        assert ds.load_lidar is True

    def test_dataset_split(self):
        from src.data.nuscenes_dataset import NuScenesDataset

        ds = NuScenesDataset(root="/tmp", version="v1.0-mini", split="val")
        assert ds.split == "val"

    def test_dataset_with_mock_samples(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.nuscenes_dataset import NuScenesDataset

        cam_dir = tmp_path / "samples" / "CAM_FRONT"
        cam_dir.mkdir(parents=True)
        lidar_dir = tmp_path / "samples" / "LIDAR_TOP"
        lidar_dir.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8))
        img.save(cam_dir / "sample_0001.jpg")

        ds = NuScenesDataset(root=str(tmp_path), version="v1.0-mini")
        assert len(ds) == 1

    def test_dataset_getitem_with_mock(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.nuscenes_dataset import NuScenesDataset

        cam_dir = tmp_path / "samples" / "CAM_FRONT"
        cam_dir.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8))
        img.save(cam_dir / "sample_0001.jpg")

        ds = NuScenesDataset(root=str(tmp_path), version="v1.0-mini")
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 2

    def test_dataset_getitem_with_transform(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.nuscenes_dataset import NuScenesDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        cam_dir = tmp_path / "samples" / "CAM_FRONT"
        cam_dir.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8))
        img.save(cam_dir / "sample_0001.jpg")

        tf = A.Compose([A.Resize(480, 640), ToTensorV2()])
        ds = NuScenesDataset(root=str(tmp_path), version="v1.0-mini", transform=tf)
        sample = ds[0]
        assert isinstance(sample, tuple)

    def test_dataset_with_lidar(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.nuscenes_dataset import NuScenesDataset

        cam_dir = tmp_path / "samples" / "CAM_FRONT"
        cam_dir.mkdir(parents=True)
        lidar_dir = tmp_path / "samples" / "LIDAR_TOP"
        lidar_dir.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8))
        img.save(cam_dir / "sample_0001.jpg")

        pts = np.random.randn(1000, 5).astype(np.float32)
        pts.tofile(str(lidar_dir / "sample_0001.bin"))

        ds = NuScenesDataset(root=str(tmp_path), version="v1.0-mini", load_lidar=True)
        sample = ds[0]
        assert isinstance(sample, tuple)


class TestKITTIDatasetGetitem:
    """Cover kitti_dataset.py __getitem__ and transform paths."""

    def test_getitem_returns_sample(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.kitti_dataset import KITTIDataset

        img_dir = tmp_path / "training" / "image_2"
        label_dir = tmp_path / "training" / "label_2"
        velodyne_dir = tmp_path / "training" / "velodyne"
        for d in [img_dir, label_dir, velodyne_dir]:
            d.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8))
        img.save(img_dir / "000000.png")
        (label_dir / "000000.txt").write_text(
            "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
        )
        pts = np.random.randn(100, 4).astype(np.float32)
        pts.tofile(str(velodyne_dir / "000000.bin"))

        ds = KITTIDataset(root=str(tmp_path), split="training")
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert "annotations" in sample[1]

    def test_getitem_with_transform(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.kitti_dataset import KITTIDataset
        from src.data.augmentations import get_val_transforms

        img_dir = tmp_path / "training" / "image_2"
        label_dir = tmp_path / "training" / "label_2"
        for d in [img_dir, label_dir]:
            d.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8))
        img.save(img_dir / "000000.png")
        (label_dir / "000000.txt").write_text(
            "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
        )

        tf = get_val_transforms(image_size=(480, 640))
        ds = KITTIDataset(root=str(tmp_path), split="training", transform=tf)
        sample = ds[0]
        assert isinstance(sample, tuple)

    def test_getitem_with_lidar(self, tmp_path):
        from PIL import Image as PILImage
        from src.data.kitti_dataset import KITTIDataset

        img_dir = tmp_path / "training" / "image_2"
        label_dir = tmp_path / "training" / "label_2"
        velodyne_dir = tmp_path / "training" / "velodyne"
        for d in [img_dir, label_dir, velodyne_dir]:
            d.mkdir(parents=True)

        img = PILImage.fromarray(np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8))
        img.save(img_dir / "000000.png")
        (label_dir / "000000.txt").write_text(
            "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57\n"
        )
        pts = np.random.randn(500, 4).astype(np.float32)
        pts[:, :3] *= 20
        pts.tofile(str(velodyne_dir / "000000.bin"))

        ds = KITTIDataset(root=str(tmp_path), split="training", load_lidar=True)
        sample = ds[0]
        assert isinstance(sample, tuple)
        assert len(sample) >= 2
