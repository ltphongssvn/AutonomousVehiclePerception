AutonomousVehiclePerception/README.md
# Autonomous Vehicle Perception — CNN Pipeline

Real-time object detection and segmentation from multi-camera feeds for self-driving vehicles, built with PyTorch, Django, FastAPI, and NVIDIA Triton.

## Problem Statement

Autonomous vehicles require real-time perception systems that detect and classify objects (vehicles, pedestrians, cyclists, traffic signs) from multi-camera and LiDAR sensor feeds. This project implements an end-to-end deep learning pipeline from training through production deployment.

## Deep Learning Concepts

- **`nn.Conv2d/Conv3d`** — Spatial feature extraction from camera images and LiDAR voxel grids
- **Locality and translation invariance** — Detecting objects regardless of position in frame
- **Max/Average pooling** — Downsampling feature maps and building feature pyramids
- **FPN-ResNet50 / EfficientNet** — Backbone architectures for object detection
- **VoxelNet-style 3D CNNs** — Processing voxelized LiDAR point clouds

## Architecture
```
┌───────────────────────────────────────────────────────────┐
│                    Frontend (React)                       │
│         Three.js (3D point cloud visualization)           │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│              Django Backend (Port 8000)                   │
│   Fleet management, RBAC, audit logs, admin dashboards    │
│   Django REST Framework API                               │
│   Celery + Redis (async batch processing)                 │
│   PostgreSQL + TimescaleDB + MinIO/S3                     │
└──────────────────────┬────────────────────────────────────┘
                       │ REST/gRPC
┌──────────────────────▼─────────────────────────────────────┐
│           FastAPI Model Service (Port 8001)                │
│   Preprocessing + PyTorch CNN inference + Postprocessing   │
│   Gradio demo UI (Port 7860)                               │
└──────────────────────┬─────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────┐
│        NVIDIA Triton Inference Server (Port 8002)          │
│   TensorRT-optimized models, batched GPU inference         │
└────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Training
| Component | Technology |
|---|---|
| Notebook | JupyterLab on AWS SageMaker Studio |
| GPU | NVIDIA A100 |
| Framework | PyTorch + TorchVision |
| Backbones | FPN-ResNet50, EfficientNet |
| 3D Convolutions | `nn.Conv3d` on voxelized LiDAR (VoxelNet-style) |
| Data Pipeline | AWS S3 → PyTorch DataLoader, Albumentations augmentation |
| Experiment Tracking | TensorBoard, Weights & Biases |

### Backend & Serving
| Component | Technology |
|---|---|
| Backend | Django + Django REST Framework |
| Model Service | FastAPI microservice (async) |
| GPU Inference | NVIDIA Triton Inference Server (TensorRT) |
| Frontend | React + Three.js |
| Database | PostgreSQL (metadata) + TimescaleDB (telemetry) |
| Object Storage | MinIO / AWS S3 (raw sensor data) |
| Task Queue | Celery + Redis |
| Auth | Django + OAuth2 / Keycloak SSO |

### Production & Deployment
| Component | Technology |
|---|---|
| Containerization | Docker + Docker Compose |
| Orchestration | AWS EKS (Kubernetes) |
| IaC | Terraform |
| GitOps | ArgoCD |
| CI/CD | GitHub Actions → ECR → EKS |
| Monitoring | Prometheus + Grafana (infra), TensorBoard (training) |
| Edge/Mobile | ONNX export → TensorRT on NVIDIA Jetson AGX Orin |

### Performance & Optimization
| Concern | Implementation |
|---|---|
| Compilation | `torch.compile` (graph mode) |
| Quantization | INT8/INT4 via `torch.ao.quantization` |
| Interoperability | ONNX export, TorchExport |
| Mobile/Edge | ExecuTorch, TensorRT |
| Profiling | TensorBoard Profiler |

## Public Training Datasets

| Dataset | Size | Source |
|---|---|---|
| **KITTI** | 15K frames, 3D bounding boxes | Karlsruhe Institute of Technology |
| **nuScenes** | 1.4M 3D boxes, 40K frames, LiDAR | Motional |
| **Waymo Open Dataset** | 12M 3D labels, 1.2K driving segments | Waymo |
| **BDD100K** | 100K videos, 10 tasks | UC Berkeley |
| **COCO** | 330K images, 80 categories | Microsoft |

## Project Structure
```
AutonomousVehiclePerception/
├── README.md
├── SECURITY.md
├── .pre-commit-config.yaml
├── .secrets.baseline
├── .gitignore
├── docker-compose.yml
├── terraform/
├── .github/
│   └── workflows/
├── notebooks/                  # JupyterLab training notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_training.ipynb
│   ├── 03_3d_voxel_training.ipynb
│   └── 04_model_export.ipynb
├── src/
│   ├── django_backend/         # Django fleet management app
│   │   ├── manage.py
│   │   ├── config/
│   │   ├── fleet/
│   │   ├── perception/
│   │   └── templates/
│   ├── fastapi_service/        # FastAPI model inference microservice
│   │   ├── main.py
│   │   ├── inference/
│   │   └── preprocessing/
│   ├── model/                  # PyTorch model definitions
│   │   ├── cnn_2d.py
│   │   ├── cnn_3d_voxel.py
│   │   ├── fpn_resnet.py
│   │   └── export.py
│   ├── data/                   # Dataset loaders and augmentation
│   │   ├── kitti_dataset.py
│   │   ├── nuscenes_dataset.py
│   │   └── augmentations.py
│   └── frontend/               # React + Three.js frontend
│       ├── package.json
│       └── src/
├── triton/                     # Triton model repository
│   └── models/
├── deploy/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
└── tests/
```

## Quick Start
```bash
# Clone
git clone git@github.com:ltphongssvn/AutonomousVehiclePerception.git
cd AutonomousVehiclePerception

# Install pre-commit hooks
pip install pre-commit detect-secrets
pre-commit install

# Start services
docker-compose up -d

# Access
# Django admin:  http://localhost:8000/admin
# FastAPI docs:  http://localhost:8001/docs
# Gradio demo:   http://localhost:7860
# Triton:        http://localhost:8002
```

## Flow
```
JupyterLab (train CNN on GPU)
  → torch.compile for optimization
  → ONNX export
  → TensorRT optimization
  → Triton server on cloud (shadow-mode validation)
  → ExecuTorch/TensorRT on edge (real-time inference at 30+ FPS)
```

## Security

See [SECURITY.md](SECURITY.md) for secret management, pre-commit hooks, and team guidelines.

## License

MIT
