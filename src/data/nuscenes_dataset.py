# AutonomousVehiclePerception/src/data/nuscenes_dataset.py
"""nuScenes dataset loader for multi-camera 3D object detection.

Loads nuScenes format data:
- Multi-camera images (6 surround-view cameras)
- 3D bounding box annotations with velocity
- LiDAR point clouds for voxelization

Reference: https://www.nuscenes.org/
Requires: pip install nuscenes-devkit
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# nuScenes detection classes
NUSCENES_CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(NUSCENES_CLASSES)}

# nuScenes camera names (6 surround-view cameras)
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesDataset(Dataset):
    """PyTorch Dataset for nuScenes multi-camera 3D detection.

    Supports two modes:
    1. Single-camera: returns one camera image per sample
    2. Multi-camera: returns all 6 surround-view images per sample

    Expected: nuScenes devkit installed and dataset downloaded.

    Args:
        root: Path to nuScenes dataset root (contains maps/, samples/, sweeps/, v1.0-*)
        version: Dataset version ('v1.0-mini', 'v1.0-trainval', 'v1.0-test')
        split: 'train' or 'val'
        camera: Single camera name or 'all' for multi-camera
        transform: Optional image transform
        load_lidar: Whether to load LiDAR point clouds
    """

    def __init__(self, root, version="v1.0-mini", split="train", camera="CAM_FRONT", transform=None, load_lidar=False):
        self.root = Path(root)
        self.version = version
        self.split = split
        self.camera = camera
        self.transform = transform
        self.load_lidar = load_lidar
        self.multi_camera = camera == "all"

        # Try to load nuScenes devkit
        self.nusc = None
        self.samples = []
        try:
            from nuscenes.nuscenes import NuScenes

            self.nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
            self.samples = self._get_split_samples()
            print(f"nuScenes {version} loaded: {len(self.samples)} samples ({split})")
        except ImportError:
            print("WARNING: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
            print("Falling back to file-based loading.")
            self._init_file_based()
        except Exception as e:
            print(f"WARNING: Could not load nuScenes database: {e}")
            self._init_file_based()

    def _get_split_samples(self):
        """Get sample tokens for train/val split."""
        from nuscenes.utils.splits import create_splits_scenes

        split_scenes = create_splits_scenes()
        split_key = "train" if self.split == "train" else "val"
        scene_names = split_scenes.get(split_key, [])

        sample_tokens = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_names:
                sample_token = scene["first_sample_token"]
                while sample_token:
                    sample_tokens.append(sample_token)
                    sample = self.nusc.get("sample", sample_token)
                    sample_token = sample["next"] if sample["next"] else None
        return sample_tokens

    def _init_file_based(self):
        """Fallback: load from file system without devkit."""
        samples_dir = self.root / "samples" / "CAM_FRONT"
        if samples_dir.exists():
            self.samples = sorted([f.stem for f in samples_dir.glob("*.jpg")])
            print(f"File-based loading: {len(self.samples)} images found")
        else:
            self.samples = []
            print(f"WARNING: No samples found at {samples_dir}")

    def _load_image(self, filepath):
        """Load and normalize an image."""
        image = Image.open(filepath).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0
        return image

    def _get_annotations(self, sample_token):
        """Get 3D bounding box annotations for a sample."""
        if self.nusc is None:
            return []

        sample = self.nusc.get("sample", sample_token)
        annotations = []

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            category = ann["category_name"].split(".")[-1]

            if category not in CLASS_TO_IDX:
                # Map subcategories to parent classes
                for cls in NUSCENES_CLASSES:
                    if cls in ann["category_name"]:
                        category = cls
                        break
                else:
                    continue

            annotations.append(
                {
                    "class": category,
                    "class_idx": CLASS_TO_IDX.get(category, -1),
                    "translation": ann["translation"],  # x, y, z in global frame
                    "size": ann["size"],  # width, length, height
                    "rotation": ann["rotation"],  # quaternion
                    "num_lidar_pts": ann["num_lidar_pts"],
                    "visibility": ann.get("visibility_token", ""),
                }
            )
        return annotations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]

        if self.nusc is not None:
            return self._getitem_devkit(sample_token)
        else:
            return self._getitem_file_based(sample_token)

    def _getitem_devkit(self, sample_token):
        """Load sample using nuScenes devkit."""
        sample = self.nusc.get("sample", sample_token)

        if self.multi_camera:
            # Load all 6 cameras
            images = {}
            for cam_name in CAMERA_NAMES:
                cam_data = self.nusc.get("sample_data", sample["data"][cam_name])
                img_path = self.root / cam_data["filename"]
                image = self._load_image(str(img_path))
                if self.transform:
                    image = self.transform(image=image)["image"]
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1)
                images[cam_name] = image

            annotations = self._get_annotations(sample_token)
            target = {
                "sample_token": sample_token,
                "annotations": annotations,
                "num_objects": len(annotations),
            }
            return images, target
        else:
            # Single camera
            cam_data = self.nusc.get("sample_data", sample["data"][self.camera])
            img_path = self.root / cam_data["filename"]
            image = self._load_image(str(img_path))

            if self.transform:
                image = self.transform(image=image)["image"]
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)

            annotations = self._get_annotations(sample_token)
            target = {
                "sample_token": sample_token,
                "annotations": annotations,
                "num_objects": len(annotations),
            }
            return image, target

    def _getitem_file_based(self, sample_id):
        """Fallback file-based loading."""
        img_path = self.root / "samples" / "CAM_FRONT" / f"{sample_id}.jpg"
        if img_path.exists():
            image = self._load_image(str(img_path))
        else:
            # Return placeholder
            image = np.zeros((900, 1600, 3), dtype=np.float32)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        target = {
            "sample_id": sample_id,
            "annotations": [],
            "num_objects": 0,
        }
        return image, target


if __name__ == "__main__":
    print("nuScenes Dataset Loader")
    print(f"Classes: {NUSCENES_CLASSES}")
    print(f"Cameras: {CAMERA_NAMES}")
    print(f"Class mapping: {CLASS_TO_IDX}")
