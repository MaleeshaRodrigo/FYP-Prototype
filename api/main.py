"""HARE Platform API — FastAPI entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.model_loader import load_model
from .routers import admin, attack, experiments, gradcam, metrics, predict, system

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(gradcam.router)
app.include_router(attack.router)
app.include_router(metrics.router)
app.include_router(experiments.router)
app.include_router(admin.router)
app.include_router(system.router)


@app.on_event("startup")
async def startup():
    """Loads the HARE model on application startup."""
    load_model()


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "model": settings.ACTIVE_CHECKPOINT,
    }
