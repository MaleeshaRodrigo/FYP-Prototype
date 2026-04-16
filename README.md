# HARE Platform

Hybrid Adversarially Robust Ensemble (HARE) platform for skin cancer decision support.

This repository contains:
- `ui/`: Vanilla ES2022 single-page frontend (no build step)
- `api/`: FastAPI backend with inference and research/admin endpoints
- `shared/`: Shared constants and mock fixtures
- `.github/workflows/`: Azure deployment pipelines

## Guides

- Local setup: `docs/LOCAL_SETUP_GUIDE.md`
- Deployment: `docs/DEPLOYMENT_GUIDE.md`

## Highlights

- Role-based portals: clinician, research, admin, and system admin
- Clinical safety behavior:
  - Borderline warning when `|confidence - theta| < 0.05`
  - Model version shown with each diagnosis
  - GradCAM required on MEL-positive results
- Mock-first frontend mode for offline/demo development
- Research tools: version history, tradeoff chart, PGD attack simulation

## Project Structure

```text
.
├── ui/
│   ├── app.js
│   ├── index.html
│   ├── style.css
│   ├── infrastructure/
│   ├── domain/
│   ├── application/
│   └── presentation/
├── api/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── core/
│   ├── domain/
│   ├── services/
│   └── routers/
├── shared/
│   ├── constants.json
│   └── mock-fixtures/
└── .github/workflows/
```

## Prerequisites

- Python 3.11+
- `pip`
- (Optional) Docker

## Environment Variables

Create a local environment file from the template:

```bash
copy .env.example .env
```

If you run the API from inside the `api/` directory instead of the repository root:

```bash
copy api\.env.example api\.env
```

Required for production-like API runs:
- `JWT_SECRET`: set to a strong random value

The API reads `.env` automatically from the repository root (or from `api/.env` if present).

## Model Checkpoint Placement

Place checkpoints in the repository-level `models/` folder.

Default active checkpoint:
- `stage2_v8.pth`

These values are controlled by:
- `MODEL_DIR`
- `ACTIVE_CHECKPOINT`

Example for root-based runs (`.env`):

```env
MODEL_DIR=./models
ACTIVE_CHECKPOINT=stage2_v8.pth
```

Example for API-folder runs (`api/.env`):

```env
MODEL_DIR=../models
ACTIVE_CHECKPOINT=stage2_v8.pth
```

## Run the API (FastAPI)

From repository root:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r api/requirements.txt
copy .env.example .env
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- Health: `GET /api/health`
- Docs: `GET /api/docs`

## Run the UI (No Build Step)

Serve from the repository root so both `ui` and `shared` are accessible:

```bash
python -m http.server 8080
```

Open:
- Mock mode (default): `http://localhost:8080/ui/`
- Production service mode (local API): `http://localhost:8080/ui/?env=production&apiBaseUrl=http://localhost:8000`

When `env=production` is set, the frontend uses the production API service implementation.
`apiBaseUrl` is optional in hosted deployments, but recommended for local development when UI and API run on different ports.

## Key Frontend Routes

- `#/login`
- `#/thesis`
- `#/clinical`
- `#/research`
- `#/admin`
- `#/system`

## Main API Routes

- `POST /api/predict`
- `GET /api/gradcam/{image_id}`
- `POST /api/attack/simulate`
- `GET /api/metrics/{version}`
- `GET /api/metrics/comparison/{baseline_version}/{candidate_version}`
- `GET /api/metrics/thesis/summary`
- `GET /api/metrics/thesis/sweep`
- `GET /api/metrics/thesis/trades-beta-sweep`
- `GET /api/metrics/thesis/export/json`
- `GET /api/metrics/thesis/export/csv`
- `GET /api/experiments`
- `GET /api/admin/usage`
- `GET /api/admin/audit`
- `GET /api/system/models`
- `PUT /api/system/parameters`

## CI/CD

- UI deployment: `.github/workflows/deploy-ui.yml`
- API deployment: `.github/workflows/deploy-api.yml`

Both workflows trigger on pushes to `main` scoped by path (`ui/**` and `api/**`).

## Notes

- The repository currently includes mock/demo behavior for model outputs and GradCAM.
- Replace service internals with real model loading/inference logic as checkpoints and runtime infrastructure are finalized.
