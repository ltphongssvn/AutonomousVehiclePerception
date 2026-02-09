# AutonomousVehiclePerception/src/model/cnn_3d_voxel.py
"""3D CNN for voxelized LiDAR point cloud processing using nn.Conv3d.

Demonstrates:
- nn.Conv3d for volumetric feature extraction from LiDAR voxel grids
- 3D Max/Average pooling for spatial downsampling
- VoxelNet-style architecture for 3D object detection
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU -> Optional 3D Pooling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_type=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.pool = None
        if pool_type == "max":
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class VoxelBackbone3D(nn.Module):
    """3D CNN backbone for voxelized LiDAR point clouds.

    Processes a dense voxel grid (D x H x W) where each voxel contains
    point features (occupancy, intensity, etc).

    Architecture:
    - Progressive 3D convolution stages with pooling
    - Collapses depth dimension via strided convolution
    - Outputs 2D BEV (bird's eye view) feature map for detection head

    Input: (B, C, D, H, W) voxel grid
        - D: depth (z-axis bins)
        - H: height (y-axis bins)
        - W: width (x-axis bins)
        - C: per-voxel feature channels
    Output: (B, num_classes, H', W') BEV class probability map
    """

    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        # 3D feature extraction stages
        self.stage1 = nn.Sequential(
            ConvBlock3D(in_channels, 32),
            ConvBlock3D(32, 32),
        )
        self.stage2 = nn.Sequential(
            ConvBlock3D(32, 64, pool_type="max"),  # D/2, H/2, W/2
            ConvBlock3D(64, 64),
        )
        self.stage3 = nn.Sequential(
            ConvBlock3D(64, 128, pool_type="max"),  # D/4, H/4, W/4
            ConvBlock3D(128, 128),
        )

        # Collapse depth dimension with strided conv along z-axis
        # Converts 3D volume to 2D BEV (bird's eye view)
        self.depth_collapse = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # 2D BEV detection head (applied after squeezing depth dim)
        self.bev_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # 3D feature extraction
        x = self.stage1(x)  # (B, 32, D, H, W)
        x = self.stage2(x)  # (B, 64, D/2, H/2, W/2)
        x = self.stage3(x)  # (B, 128, D/4, H/4, W/4)

        # Collapse depth to BEV
        x = self.depth_collapse(x)  # (B, 128, D/8, H/4, W/4)
        x = x.mean(dim=2)  # Average over remaining depth → (B, 128, H/4, W/4)

        # BEV detection
        out = self.bev_head(x)  # (B, num_classes, H/4, W/4)
        return out


if __name__ == "__main__":
    # Sanity check with typical voxel grid dimensions
    # 40 depth bins x 512 height bins x 512 width bins, 1 channel (occupancy)
    model = VoxelBackbone3D(in_channels=1, num_classes=5)
    dummy_input = torch.randn(1, 1, 40, 128, 128)  # Smaller grid for testing
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
