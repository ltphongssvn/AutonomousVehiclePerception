# AutonomousVehiclePerception/ENGINEERING_LOG.md
# Engineering Log & TDD Strategy

## TDD Strategy for AV Perception Platform

### Core Rule: RED → GREEN → REFACTOR
1. Write failing test for desired behavior
2. Implement simplest code to pass
3. Refactor without changing behavior

---

### Test Pyramid
```
tests/
├── unit/
│   ├── model/          # PyTorch model shape/gradient tests
│   ├── data/           # Dataset loader, augmentation tests
│   ├── fastapi/        # Endpoint, preprocessing tests
│   └── django/         # Model, serializer, view tests
├── integration/
│   ├── api/            # Django ↔ FastAPI contract tests
│   └── compose/        # Docker Compose wiring tests
└── e2e/
    └── ui/             # React upload → detection flow
```

### A) Model Layer (`src/model/`) — pytest + torch
- Tensor shape contracts (input → output)
- Gradient flow verification
- ONNX export produces valid file + passes OnnxRuntime check
- Golden tests: fixed inputs produce consistent summary stats

### B) Data Pipeline (`src/data/`) — pytest
- Dataset returns stable schema: image tensor + annotations dict
- Augmentations preserve bbox validity (x1<x2, y1<y2)
- Label count matches box count post-augmentation
- Voxelization output shape matches Conv3d expectations
- Corrupt input handling (missing files, empty annotations)

### C) FastAPI Service (`src/fastapi_service/`) — pytest + httpx
- `/health` returns 200 + schema
- `/predict/2d` returns 200 + PredictionResponse on valid image
- `/predict/3d` returns 200 on valid .bin/.npy
- Returns 422 on invalid payload
- Preprocessing produces correct tensor shape/dtype
- API response schema matches what Django proxy and React expect

### D) Django Backend (`src/django_backend/`) — pytest-django
- Model constraints (unique VIN, FK cascades)
- Serializer validation (required fields, read-only)
- ViewSet CRUD operations
- Fleet summary endpoint aggregation
- Celery task logic with mocked model service
- FastAPI proxy forwards correctly

### E) Triton (`triton_models/`) — nightly CI
- Config pbtxt matches ONNX model I/O dims
- Smoke test: send request, validate output shape

### F) React Frontend — Jest + React Testing Library
- Components render given props
- API error handling (connection refused, timeout)
- Upload → result display flow

### Cross-Service Contracts
- FastAPI PredictionResponse is versioned schema
- Django proxy returns same schema as direct FastAPI call
- React consumes all fields in PredictionResponse

### CI Strategy (GitHub Actions)
- **PR (fast):** lint (black, flake8) + unit tests (CPU)
- **PR (medium):** Docker build + compose integration
- **Nightly:** Triton + ONNX export + performance smoke

---

## Development Log

### 2025-02-08: Project Initialization
- Created GitHub repo: ltphongssvn/AutonomousVehiclePerception
- Security setup: pre-commit hooks, detect-secrets, SECURITY.md
- All hooks passing: secrets, model file blocking, data bloat, black, flake8

### 2025-02-08: PyTorch Models
- `src/model/cnn_2d.py` — 2D CNN with FPN, Conv2d, Max/Avg pooling
- `src/model/cnn_3d_voxel.py` — 3D VoxelNet-style backbone with Conv3d, BEV output
- `src/model/fpn_resnet.py` — FPN-ResNet50 with TorchVision pretrained weights
- `src/model/export.py` — ONNX, TorchScript, torch.compile, INT8 quantization
- Unit tests: 13/13 passing (`python -m pytest tests/test_models.py`)
- Fixed: `triton/` → `triton_models/` (Python namespace conflict with PyTorch triton)

### 2025-02-08: Data Pipeline
- `src/data/kitti_dataset.py` — KITTI loader with label parsing, LiDAR voxelization
- `src/data/nuscenes_dataset.py` — nuScenes multi-camera loader with devkit fallback
- `src/data/augmentations.py` — Albumentations (camera) + custom 3D (LiDAR) augmentations

### 2025-02-08: FastAPI Model Service
- `src/fastapi_service/main.py` — /predict/2d, /predict/3d, /predict/fpn, /health, /model-info
- `src/fastapi_service/gradio_app.py` — Interactive demo UI on port 7860

### 2025-02-08: Django Backend
- Apps: fleet (Vehicle, DrivingSession, SensorFrame), perception (MLModel, DetectionResult, DetectedObject)
- DRF serializers, viewsets, URL routing, admin registration
- Celery async tasks for batch session processing
- Settings: env-based config, PostgreSQL/SQLite, CORS, DRF pagination

### 2025-02-08: Infrastructure
- docker-compose.yml: Django, FastAPI, Celery, PostgreSQL, Redis, Prometheus, Grafana
- Dockerfiles: Django (python:3.11-slim + psycopg2), FastAPI (+ libgl1 for OpenCV)
- Kubernetes: namespace, Django/FastAPI/Triton deployments, ingress, secrets
- Terraform: EKS, RDS, ElastiCache, ECR, S3 on AWS
- GitHub Actions CI: lint, test, Docker build
- Triton model repository: cnn_2d, cnn_3d_voxel, fpn_resnet configs

### 2025-02-08: Frontend
- React + Three.js: App shell, FleetDashboard, PointCloudViewer, InferencePanel
- 3D point cloud with bounding boxes, orbit controls
- Model selection, image upload, result visualization

### 2025-02-08: Monitoring
- Prometheus scrape configs for Django, FastAPI, Triton
- Grafana datasource provisioning
- TensorBoard for training metrics (in notebooks)
