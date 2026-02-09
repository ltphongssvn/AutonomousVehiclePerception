# AutonomousVehiclePerception/src/model/fpn_resnet.py
"""FPN-ResNet50 backbone using TorchVision pretrained weights.

Demonstrates:
- Transfer learning with pretrained ResNet50
- Feature Pyramid Network for multi-scale detection
- Integration with TorchVision's detection utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class FPNResNet50(nn.Module):
    """Feature Pyramid Network with ResNet50 backbone.

    Extracts multi-scale features from pretrained ResNet50 layers (C2-C5)
    and builds top-down FPN feature maps (P2-P5) for object detection.

    Input: (B, 3, H, W) RGB image
    Output: dict of FPN levels {"p2": ..., "p3": ..., "p4": ..., "p5": ...}
    """

    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()

        # Load pretrained ResNet50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Extract ResNet stages (C1-C5)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # C2: stride 4,  256 channels
        self.layer2 = backbone.layer2  # C3: stride 8,  512 channels
        self.layer3 = backbone.layer3  # C4: stride 16, 1024 channels
        self.layer4 = backbone.layer4  # C5: stride 32, 2048 channels

        # FPN lateral connections (1x1 conv to reduce channels)
        self.lateral5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(256, out_channels, kernel_size=1)

        # FPN smooth layers (3x3 conv to reduce aliasing from upsampling)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def _upsample_add(self, x, lateral):
        """Upsample x to match lateral size and add."""
        _, _, h, w = lateral.shape
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False) + lateral

    def forward(self, x):
        # Bottom-up (ResNet backbone)
        c1 = self.stem(x)
        c2 = self.layer1(c1)  # (B, 256, H/4, W/4)
        c3 = self.layer2(c2)  # (B, 512, H/8, W/8)
        c4 = self.layer3(c3)  # (B, 1024, H/16, W/16)
        c5 = self.layer4(c4)  # (B, 2048, H/32, W/32)

        # Top-down FPN with lateral connections
        p5 = self.lateral5(c5)
        p4 = self._upsample_add(p5, self.lateral4(c4))
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))

        # Smooth
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class FPNDetector(nn.Module):
    """Object detector using FPN-ResNet50 backbone.

    Adds a shared classification + regression head on top of FPN features.

    Input: (B, 3, H, W) RGB image
    Output: (B, num_classes, H/4, W/4) class probability map from finest FPN level
    """

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.fpn = FPNResNet50(out_channels=256, pretrained=pretrained)

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        fpn_features = self.fpn(x)
        # Use finest FPN level (P2) for dense prediction
        out = self.head(fpn_features["p2"])
        return out


if __name__ == "__main__":
    model = FPNDetector(num_classes=10, pretrained=False)
    dummy_input = torch.randn(1, 3, 480, 640)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")

    # Show FPN multi-scale outputs
    fpn_out = model.fpn(dummy_input)
    for level, feat in fpn_out.items():
        print(f"FPN {level}: {feat.shape}")
