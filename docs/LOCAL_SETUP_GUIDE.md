# HARE Platform Local Setup Guide

This guide sets up the full project locally (API + UI) for development and testing.

## 1. Prerequisites

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer
- pip (bundled with Python)
- Git
- Optional: Docker Desktop (for container-based local runs)

## 2. Clone and Open the Project

```bash
git clone <your-repo-url>
cd Prototype
```

Open the folder in VS Code.

## 3. Create and Activate a Virtual Environment

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

## 4. Install Backend Dependencies

Install API dependencies from the backend requirements file:

```bash
pip install -r api/requirements.txt
```

## 5. Configure Environment Variables

From repository root:

```bash
copy .env.example .env
```

Or if you run the API from inside the api folder:

```bash
copy api\.env.example api\.env
```

Update the environment file values as needed.

Minimum required variables:

- DEBUG=true
- JWT_SECRET=<long-random-secret>
- CORS_ORIGINS=http://localhost:8080,http://localhost:3000
- MODEL_DIR=./models (or ../models if running from api folder)
- ACTIVE_CHECKPOINT=stage2_v8.pth

GA defaults (optional to tune):

- GA_THETA=0.3985
- GA_TAU=0.7671
- GA_ALPHA=0.5467

## 6. Place Model Checkpoint Files

Add model checkpoints to the repository models folder:

- models/stage2_v8.pth
- Any additional checkpoints used by System Admin model switching

If no real checkpoint is available, the application can still run in fallback/demo mode.

## 7. Start the API Server

From repository root (recommended):

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Verify backend health:

- API health: http://localhost:8000/api/health
- Swagger docs: http://localhost:8000/api/docs

## 8. Start the UI Server

Open a new terminal in repository root and run:

```bash
python -m http.server 8080
```

Open the UI:

- Mock mode: http://localhost:8080/ui/
- Production mode (local API): http://localhost:8080/ui/?env=production&apiBaseUrl=http://localhost:8000

## 9. Quick Functional Verification Checklist

1. Open Login and sign in to each role flow.
2. Go to Thesis Hub and confirm summary cards/charts load.
3. Open Clinical view and run prediction.
4. Trigger GradCAM and verify heatmap viewer renders.
5. Open Research view and run attack simulation with an image.
6. Open System view and test GA parameter load/save.

## 10. Common Local Issues and Fixes

### PowerShell execution policy blocks venv activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then re-run activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

### CORS error in browser

Check CORS_ORIGINS in .env and include UI origin:

- http://localhost:8080

Restart API after changing env values.

### Missing or invalid model checkpoint

- Verify ACTIVE_CHECKPOINT matches a file in MODEL_DIR.
- If unavailable, continue in fallback/demo mode for UI integration testing.

### Port already in use

Use alternate ports:

```bash
python -m uvicorn api.main:app --reload --port 8001
python -m http.server 8081
```

Then update apiBaseUrl in UI query string accordingly.

## 11. Optional Local Docker Run (API)

If you want a containerized backend locally:

```bash
docker build -t hare-api-local ./api
docker run --rm -p 8000:8000 --env-file .env hare-api-local
```

Note:

- Ensure env values are production-safe if DEBUG=false.
- Ensure model files are available to the container (bake into image or mount as volume).

## 12. Recommended Local Dev Workflow

1. Keep API running with --reload.
2. Serve UI with python -m http.server.
3. Work in production UI mode using apiBaseUrl to validate real API integration.
4. Use mock mode only when backend is intentionally offline.
