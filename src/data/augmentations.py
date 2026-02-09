# AutonomousVehiclePerception/src/data/augmentations.py
"""Albumentations-based augmentation pipelines for autonomous vehicle perception.

Provides:
- Training augmentations (heavy: flip, rotate, color jitter, blur, noise)
- Validation augmentations (resize + normalize only)
- Bounding box safe transforms (preserve object annotations)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=(480, 640)):
    """Training augmentation pipeline.

    Includes geometric and photometric augmentations that preserve
    bounding box annotations. Leverages locality and translation
    invariance properties of CNNs.

    Args:
        image_size: (height, width) output size
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
            # Photometric augmentations (simulate lighting/weather variations)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ],
                p=0.5,
            ),
            # Simulate weather/sensor noise
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            # Simulate occlusion
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2),
            # Normalize and convert to tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_val_transforms(image_size=(480, 640)):
    """Validation/inference augmentation pipeline.

    Minimal transforms: resize + normalize only.

    Args:
        image_size: (height, width) output size
    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_lidar_augmentations():
    """Augmentations for LiDAR voxel grids.

    Returns dict of augmentation functions applied to numpy voxel grids.
    These are applied manually since Albumentations doesn't support 3D volumes.
    """
    import numpy as np

    def random_flip_x(voxel_grid, p=0.5):
        """Flip voxel grid along x-axis (left-right)."""
        if np.random.random() < p:
            return np.flip(voxel_grid, axis=-1).copy()
        return voxel_grid

    def random_flip_y(voxel_grid, p=0.5):
        """Flip voxel grid along y-axis (front-back)."""
        if np.random.random() < p:
            return np.flip(voxel_grid, axis=-2).copy()
        return voxel_grid

    def random_noise(voxel_grid, std=0.1, p=0.3):
        """Add Gaussian noise to occupied voxels."""
        if np.random.random() < p:
            noise = np.random.normal(0, std, voxel_grid.shape).astype(np.float32)
            mask = voxel_grid > 0
            voxel_grid = voxel_grid + noise * mask
            return np.clip(voxel_grid, 0, None)
        return voxel_grid

    def random_dropout(voxel_grid, drop_rate=0.05, p=0.3):
        """Randomly drop occupied voxels to simulate LiDAR noise."""
        if np.random.random() < p:
            mask = np.random.random(voxel_grid.shape) > drop_rate
            return voxel_grid * mask.astype(np.float32)
        return voxel_grid

    return {
        "flip_x": random_flip_x,
        "flip_y": random_flip_y,
        "noise": random_noise,
        "dropout": random_dropout,
    }


if __name__ == "__main__":
    import numpy as np

    print("=== Image Augmentation Pipelines ===")
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    print(f"Train transforms: {len(train_tf.transforms)} stages")
    print(f"Val transforms:   {len(val_tf.transforms)} stages")

    # Test with dummy image and bbox
    dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    dummy_bboxes = [[100, 200, 300, 400]]  # pascal_voc format: [x_min, y_min, x_max, y_max]
    dummy_labels = [0]

    result = train_tf(image=dummy_image, bboxes=dummy_bboxes, class_labels=dummy_labels)
    print(f"Augmented image shape: {result['image'].shape}")
    print(f"Augmented bboxes: {result['bboxes']}")

    print("\n=== LiDAR Augmentations ===")
    lidar_augs = get_lidar_augmentations()
    dummy_voxels = np.random.rand(20, 64, 64).astype(np.float32)
    for name, aug_fn in lidar_augs.items():
        augmented = aug_fn(dummy_voxels.copy())
        print(f"{name}: shape={augmented.shape}, changed={not np.array_equal(augmented, dummy_voxels)}")
