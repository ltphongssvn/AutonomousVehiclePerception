# AutonomousVehiclePerception/src/model/export.py
"""Model export utilities for deployment.

Provides:
- ONNX export for cross-platform deployment (TensorRT, ONNX Runtime)
- torch.export for modern PyTorch deployment
- torch.compile optimization for GPU inference
- INT8 dynamic quantization for reduced model size
"""

import warnings
from pathlib import Path

import torch


def export_onnx(model, dummy_input, output_path, opset_version=18):
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (must be in eval mode)
        dummy_input: Example input tensor for tracing
        output_path: Path to save .onnx file
        opset_version: ONNX opset version (default: 18)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    print(f"ONNX exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def export_torchscript(model, dummy_input, output_path):
    """Export PyTorch model using torch.export (modern API).

    Args:
        model: PyTorch model (must be in eval mode)
        dummy_input: Example input tensor for tracing
        output_path: Path to save .pt2 file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = torch.export.export(model, (dummy_input,), strict=False)
    torch.export.save(exported, str(output_path))
    print(f"torch.export saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def optimize_with_compile(model, mode="reduce-overhead"):
    """Apply torch.compile for GPU inference optimization.

    Args:
        model: PyTorch model
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
    Returns:
        Compiled model
    """
    compiled = torch.compile(model, mode=mode)
    print(f"torch.compile applied (mode={mode})")
    return compiled


def quantize_dynamic(model, output_path=None):
    """Apply INT8 dynamic quantization for CPU inference.

    Args:
        model: PyTorch model (must be in eval mode)
        output_path: Optional path to save quantized state dict
    Returns:
        Quantized model
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import torch.ao.quantization as tq

        quantized = tq.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized.state_dict(), output_path)
        print(f"Quantized model saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return quantized


if __name__ == "__main__":
    from src.model.cnn_2d import PerceptionCNN2D
    from src.model.cnn_3d_voxel import VoxelBackbone3D
    from src.model.fpn_resnet import FPNDetector

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    dummy_2d = torch.randn(1, 3, 480, 640)
    dummy_3d = torch.randn(1, 1, 20, 128, 128)

    cnn2d = PerceptionCNN2D(num_classes=10).eval()
    cnn3d = VoxelBackbone3D(in_channels=1, num_classes=5).eval()
    fpn = FPNDetector(num_classes=10, pretrained=False).eval()

    export_onnx(cnn2d, dummy_2d, export_dir / "cnn_2d.onnx")
    export_onnx(cnn3d, dummy_3d, export_dir / "cnn_3d_voxel.onnx")
    export_onnx(fpn, dummy_2d, export_dir / "fpn_resnet.onnx")

    export_torchscript(cnn2d, dummy_2d, export_dir / "cnn_2d.pt2")
    export_torchscript(cnn3d, dummy_3d, export_dir / "cnn_3d_voxel.pt2")

    quantize_dynamic(cnn2d, export_dir / "cnn_2d_int8.pth")
