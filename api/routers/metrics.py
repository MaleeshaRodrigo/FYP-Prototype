"""GET /api/metrics/{version} — Model performance metrics."""

from fastapi import APIRouter, Depends

from ..core.security import get_current_user
from ..services.metrics_service import metrics_service

router = APIRouter(prefix="/api", tags=["metrics"])


@router.get("/metrics/{version}")
async def get_metrics(
    version: str,
    _user: dict = Depends(get_current_user),
) -> dict:
    """Returns performance metrics for the specified model version."""
    return metrics_service.get_metrics(version)
