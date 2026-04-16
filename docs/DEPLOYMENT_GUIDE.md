# HARE Platform Deployment Guide

This guide covers practical deployment for the current repository using Azure Static Web Apps (UI) and Azure Container Apps (API), aligned with existing GitHub Actions workflows.

## 1. Deployment Architecture

- UI: Azure Static Web Apps
- API: Azure Container Apps
- Container Registry: Azure Container Registry (ACR)
- CI/CD: GitHub Actions workflows in .github/workflows

Workflows already in repo:

- .github/workflows/deploy-ui.yml
- .github/workflows/deploy-api.yml

## 2. Prerequisites

- Azure subscription
- Resource Group
- Azure Container Registry (ACR)
- Azure Container Apps environment and app (hare-api)
- Azure Static Web App resource
- GitHub repository secrets configured

## 3. Required GitHub Secrets

### For UI workflow

- AZURE_STATIC_WEB_APPS_API_TOKEN

### For API workflow

- ACR_LOGIN_SERVER
- ACR_USERNAME
- ACR_PASSWORD
- ACR_NAME
- AZURE_RESOURCE_GROUP

## 4. API Environment Configuration (Container App)

Set these environment variables in Azure Container App:

- DEBUG=false
- JWT_SECRET=<long-random-secret>
- JWT_EXPIRY_MINUTES=60
- CORS_ORIGINS=<your-ui-production-origin>
- MODEL_DIR=<container path to model files>
- ACTIVE_CHECKPOINT=stage2_v8.pth
- GA_THETA=0.3985
- GA_TAU=0.7671
- GA_ALPHA=0.5467
- BLOB_CONNECTION_STRING=<optional if pulling checkpoints from storage>

Important:

- With DEBUG=false, JWT_SECRET must be provided.
- Keep CORS_ORIGINS tightly scoped to deployed UI domains.

## 5. Deploy UI (Azure Static Web Apps)

Current trigger:

- Push to main with changes under ui/**

Workflow behavior:

- Checks out code
- Uploads app from /ui

If your API is on a separate domain, ensure frontend runtime uses correct apiBaseUrl query string or configure your app hosting/reverse proxy strategy accordingly.

## 6. Deploy API (Azure Container Apps)

Current trigger:

- Push to main with changes under api/**

Workflow behavior:

1. Login to ACR
2. Build image from ./api
3. Push image tagged with commit SHA
4. Deploy image to Container App hare-api

## 7. Pre-Deployment Validation Checklist

1. Local API health endpoint responds: /api/health
2. Thesis metrics endpoints return 200
3. Attack simulation accepts multipart image upload
4. GradCAM endpoint responds for valid image ids
5. UI production mode works against local API URL
6. No unresolved diagnostics in changed files

## 8. Post-Deployment Smoke Tests

Run these against deployed API:

- GET /api/health
- GET /api/docs
- GET /api/metrics/thesis/summary
- GET /api/metrics/thesis/sweep
- GET /api/metrics/thesis/trades-beta-sweep
- GET /api/metrics/thesis/export/json
- GET /api/metrics/thesis/export/csv

UI checks:

1. Open login page.
2. Enter each role path.
3. Validate Thesis Hub charts and exports.
4. Validate Clinical prediction + GradCAM display.
5. Validate Research attack workflow with uploaded image.

## 9. Rollback Strategy

API rollback options:

- Redeploy previous known-good image tag from ACR.
- Re-run deployment workflow on previous stable commit.

UI rollback options:

- Re-run UI workflow from a previous stable commit.
- Revert problematic UI commit and push to main.

## 10. Security Hardening Recommendations

1. Store all secrets only in GitHub/Azure secret stores.
2. Rotate JWT and registry credentials periodically.
3. Restrict CORS to exact production origins.
4. Disable debug mode in production.
5. Add auth in front of API docs if docs should not be public.
6. Enable Azure diagnostics and alerting for API failures.

## 11. Cost and Reliability Notes

- Container Apps can scale to zero depending on configuration; expect cold starts.
- Static Web Apps is cost-efficient for the no-build UI.
- Add health probes and minimum replicas if low-latency API startup is required.

## 12. Optional: One-Domain Production Pattern

To simplify browser CORS and runtime API URL handling, use one public domain with reverse proxy:

- Route / to static UI
- Route /api to container app backend

This reduces front-end runtime configuration complexity and avoids cross-origin issues.
