# AutonomousVehiclePerception/src/model/cnn_2d.py
"""2D CNN backbone for camera-based object detection using nn.Conv2d.

Demonstrates:
- nn.Conv2d for spatial feature extraction
- Locality and translation invariance
- Max/Average pooling for downsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> Optional Pooling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_type=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pooling: demonstrates Max vs Average pooling for downsampling
        self.pool = None
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class PerceptionCNN2D(nn.Module):
    """Multi-class object detection backbone for camera images.

    Architecture:
    - Encoder: stack of ConvBlocks with progressive channel expansion and pooling
    - Feature Pyramid: multi-scale feature maps for detecting objects at different sizes
    - Classification head: predicts object class probabilities per spatial location

    Input: (B, 3, H, W) RGB camera image
    Output: (B, num_classes, H//16, W//16) class probability map
    """

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()

        # Encoder: progressive feature extraction with pooling
        # Each stage doubles channels and halves spatial resolution
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),  # H/2
            ConvBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, pool_type="max"),  # H/4
            ConvBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, pool_type="max"),  # H/8
            ConvBlock(256, 256),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, pool_type="max"),  # H/16
            ConvBlock(512, 512),
        )

        # Feature Pyramid Network (FPN) lateral connections
        # Fuses multi-scale features for better detection at various object sizes
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)

        # FPN smooth layers
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def _upsample_add(self, x, lateral):
        """Upsample x and add lateral connection for FPN."""
        _, _, h, w = lateral.shape
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) + lateral

    def forward(self, x):
        # Encoder forward pass — extracts multi-scale feature maps
        c1 = self.stage1(x)  # (B, 64, H/2, W/2)
        c2 = self.stage2(c1)  # (B, 128, H/4, W/4)
        c3 = self.stage3(c2)  # (B, 256, H/8, W/8)
        c4 = self.stage4(c3)  # (B, 512, H/16, W/16)

        # FPN top-down pathway — merges coarse and fine features
        p4 = self.lateral4(c4)  # (B, 256, H/16, W/16)
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))  # (B, 256, H/8, W/8)
        p2 = self.smooth2(self._upsample_add(p3, self.lateral2(c2)))  # (B, 256, H/4, W/4)

        # Classification on finest FPN level
        out = self.classifier(p4)  # (B, num_classes, H/16, W/16)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = PerceptionCNN2D(num_classes=10)
    dummy_input = torch.randn(1, 3, 480, 640)  # Single 640x480 camera frame
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
