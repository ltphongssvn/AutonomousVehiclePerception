# AutonomousVehiclePerception/SECURITY.md
# Security Best Practices Implementation

## Secret Management Implementation

### 1. Pre-commit Secret Detection Setup

**Installation:**
```bash
pip install pre-commit detect-secrets
```

**Activation:**
```bash
detect-secrets scan > .secrets.baseline
pre-commit install
```

### 2. Environment Variables Required

Create `.env` file (never commit):
```bash
# AWS
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_here

# Database
POSTGRES_PASSWORD=your_password_here
REDIS_PASSWORD=your_password_here

# Django
DJANGO_SECRET_KEY=your_django_secret_here
DJANGO_DEBUG=False

# Weights & Biases
WANDB_API_KEY=your_key_here

# Triton / Model Serving
TRITON_SERVER_URL=localhost:8001
```

### 3. Pre-commit Workflow

Every commit now automatically:
1. Scans for secrets using detect-secrets
2. Blocks large model files (.pth, .onnx, .engine, .trt, .bin)
3. Warns on data/checkpoint directory commits
4. Checks Python formatting (black) and critical linting (flake8)

**Manual scan:**
```bash
pre-commit run --all-files
```

### 4. Team Guidelines

- Never commit `.env` files
- Use environment variables for all credentials
- Run `pre-commit install` after cloning
- Review `.secrets.baseline` changes carefully
- Rotate any accidentally exposed keys immediately
- Never commit model weights — use S3/GCS or DVC
- Keep training data off Git — use cloud storage + DVC

### 5. Blocked File Extensions

The pre-commit hook blocks these from being committed:
`.h5`, `.pkl`, `.pth`, `.onnx`, `.parquet`, `.pt`, `.engine`, `.trt`, `.bin`

## Verification
```bash
$ pre-commit run --all-files
Detect secrets...........................................................Passed
Block Large Model Files..................................................Passed
Check Data Bloat.........................................................Passed
```

All secrets removed and pre-commit protection active.
