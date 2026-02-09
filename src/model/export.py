# AutonomousVehiclePerception/src/model/export.py
"""Model export utilities for production deployment.

Supports:
- ONNX export for cross-platform interoperability
- TorchScript export for C++ runtime
- torch.compile optimization
- INT8 quantization via torch.ao
"""

import os
from pathlib import Path

import torch
import torch.ao.quantization as tq

from src.model.cnn_2d import PerceptionCNN2D
from src.model.cnn_3d_voxel import VoxelBackbone3D
from src.model.fpn_resnet import FPNDetector


def export_onnx(model, dummy_input, output_path, opset_version=17, dynamic_axes=None):
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (eval mode)
        dummy_input: Example input tensor
        output_path: Path for .onnx file
        opset_version: ONNX opset version
        dynamic_axes: Dict for dynamic batch size support
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model exported: {output_path} ({file_size_mb:.1f} MB)")
    return output_path


def export_torchscript(model, dummy_input, output_path):
    """Export model to TorchScript for C++ inference."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.trace(model, dummy_input)
    scripted.save(str(output_path))
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TorchScript model exported: {output_path} ({file_size_mb:.1f} MB)")
    return output_path


def optimize_with_compile(model, mode="reduce-overhead"):
    """Apply torch.compile for graph-mode optimization.

    Args:
        model: PyTorch model
        mode: Compile mode - 'default', 'reduce-overhead', or 'max-autotune'
    Returns:
        Compiled model
    """
    compiled = torch.compile(model, mode=mode)
    print(f"Model compiled with mode='{mode}'")
    return compiled


def quantize_dynamic(model, output_path=None):
    """Apply dynamic INT8 quantization for CPU inference.

    Reduces model size and speeds up inference on CPU.
    """
    model.eval()
    quantized = tq.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized.state_dict(), output_path)
        print(f"Quantized model saved: {output_path}")
    return quantized


if __name__ == "__main__":
    export_dir = Path("exports")

    # Export 2D CNN
    print("=== 2D CNN Export ===")
    cnn2d = PerceptionCNN2D(num_classes=10)
    dummy_2d = torch.randn(1, 3, 480, 640)
    export_onnx(cnn2d, dummy_2d, export_dir / "cnn_2d.onnx")
    export_torchscript(cnn2d, dummy_2d, export_dir / "cnn_2d.pt")

    # Export 3D Voxel CNN
    print("\n=== 3D Voxel CNN Export ===")
    cnn3d = VoxelBackbone3D(in_channels=1, num_classes=5)
    dummy_3d = torch.randn(1, 1, 40, 128, 128)
    export_onnx(cnn3d, dummy_3d, export_dir / "cnn_3d_voxel.onnx")
    export_torchscript(cnn3d, dummy_3d, export_dir / "cnn_3d_voxel.pt")

    # Export FPN-ResNet50
    print("\n=== FPN-ResNet50 Export ===")
    fpn = FPNDetector(num_classes=10, pretrained=False)
    export_onnx(fpn, dummy_2d, export_dir / "fpn_resnet.onnx")

    print("\nAll exports complete.")
