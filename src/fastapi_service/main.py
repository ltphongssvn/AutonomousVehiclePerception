# AutonomousVehiclePerception/src/fastapi_service/main.py
"""FastAPI model inference microservice.

Endpoints:
- POST /predict/2d — 2D CNN inference on camera images
- POST /predict/3d — 3D CNN inference on voxelized LiDAR
- POST /predict/fpn — FPN-ResNet50 inference
- GET /health — Health check
- GET /model-info — Model metadata

Gradio demo UI available at /gradio
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.data.augmentations import get_val_transforms
from src.model.cnn_2d import PerceptionCNN2D
from src.model.cnn_3d_voxel import VoxelBackbone3D
from src.model.fpn_resnet import FPNDetector

# ---------------------------------------------------------------------------
# Global model registry — loaded once at startup
# ---------------------------------------------------------------------------
models = {}
device = None


def load_models():
    """Load all models into memory. Called once at startup."""
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2D CNN
    cnn2d = PerceptionCNN2D(num_classes=10)
    cnn2d.eval().to(device)
    models["cnn_2d"] = cnn2d

    # 3D Voxel CNN
    cnn3d = VoxelBackbone3D(in_channels=1, num_classes=5)
    cnn3d.eval().to(device)
    models["cnn_3d"] = cnn3d

    # FPN-ResNet50
    fpn = FPNDetector(num_classes=10, pretrained=False)
    fpn.eval().to(device)
    models["fpn"] = fpn

    # Load weights if available
    weights_dir = Path("weights")
    for name, model in models.items():
        weight_path = weights_dir / f"{name}.pth"
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            print(f"Loaded weights: {weight_path}")
        else:
            print(f"No weights found for {name}, using random initialization")

    print(f"Loaded {len(models)} models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    load_models()
    yield
    models.clear()
    print("Models unloaded")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AV Perception Model Service",
    description="Real-time object detection inference for autonomous vehicles",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validation transforms
val_transforms = get_val_transforms(image_size=(480, 640))


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    model: str
    shape: list
    predictions: list
    inference_time_ms: float
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: list
    gpu_available: bool


class ModelInfoResponse(BaseModel):
    name: str
    parameters: int
    input_shape: str
    output_shape: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=str(device),
        models_loaded=list(models.keys()),
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/model-info", response_model=list[ModelInfoResponse])
async def model_info():
    """Return metadata for all loaded models."""
    info = []
    for name, model in models.items():
        num_params = sum(p.numel() for p in model.parameters())
        if name == "cnn_2d":
            in_shape, out_shape = "(B, 3, 480, 640)", "(B, 10, 30, 40)"
        elif name == "cnn_3d":
            in_shape, out_shape = "(B, 1, 40, 128, 128)", "(B, 5, 32, 32)"
        else:
            in_shape, out_shape = "(B, 3, 480, 640)", "(B, 10, 120, 160)"
        info.append(ModelInfoResponse(name=name, parameters=num_params, input_shape=in_shape, output_shape=out_shape))
    return info


@app.post("/predict/2d", response_model=PredictionResponse)
async def predict_2d(file: UploadFile = File(...)):
    """Run 2D CNN inference on an uploaded camera image."""
    if "cnn_2d" not in models:
        raise HTTPException(status_code=503, detail="2D CNN model not loaded")

    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)

    transformed = val_transforms(image=image_np, bboxes=[], class_labels=[])
    tensor = transformed["image"].unsqueeze(0).to(device)

    # Inference
    start = time.perf_counter()
    with torch.no_grad():
        output = models["cnn_2d"](tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Get class predictions
    probs = torch.softmax(output, dim=1)
    pred_classes = probs.argmax(dim=1)

    return PredictionResponse(
        model="cnn_2d",
        shape=list(output.shape),
        predictions=pred_classes[0].cpu().flatten().tolist()[:100],
        inference_time_ms=round(elapsed_ms, 2),
        device=str(device),
    )


@app.post("/predict/3d", response_model=PredictionResponse)
async def predict_3d(file: UploadFile = File(...)):
    """Run 3D CNN inference on an uploaded LiDAR point cloud (.bin or .npy)."""
    if "cnn_3d" not in models:
        raise HTTPException(status_code=503, detail="3D CNN model not loaded")

    contents = await file.read()

    # Parse point cloud
    if file.filename and file.filename.endswith(".npy"):
        points = np.load(io.BytesIO(contents))
    else:
        points = np.frombuffer(contents, dtype=np.float32).reshape(-1, 4)

    # Voxelize
    from src.data.kitti_dataset import voxelize_points

    voxel_grid = voxelize_points(points)
    tensor = torch.from_numpy(voxel_grid).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)

    # Inference
    start = time.perf_counter()
    with torch.no_grad():
        output = models["cnn_3d"](tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    pred_classes = output.argmax(dim=1)

    return PredictionResponse(
        model="cnn_3d",
        shape=list(output.shape),
        predictions=pred_classes[0].cpu().flatten().tolist()[:100],
        inference_time_ms=round(elapsed_ms, 2),
        device=str(device),
    )


@app.post("/predict/fpn", response_model=PredictionResponse)
async def predict_fpn(file: UploadFile = File(...)):
    """Run FPN-ResNet50 inference on an uploaded camera image."""
    if "fpn" not in models:
        raise HTTPException(status_code=503, detail="FPN model not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)

    transformed = val_transforms(image=image_np, bboxes=[], class_labels=[])
    tensor = transformed["image"].unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output = models["fpn"](tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(output, dim=1)
    pred_classes = probs.argmax(dim=1)

    return PredictionResponse(
        model="fpn",
        shape=list(output.shape),
        predictions=pred_classes[0].cpu().flatten().tolist()[:100],
        inference_time_ms=round(elapsed_ms, 2),
        device=str(device),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.fastapi_service.main:app", host="0.0.0.0", port=8001, reload=True)
