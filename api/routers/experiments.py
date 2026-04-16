"""GET /api/experiments — Experiment version history."""

from fastapi import APIRouter, Depends

from ..core.security import get_current_user
from ..services.experiment_service import experiment_service

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments")
async def get_experiments(
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """Returns the full AT experiment version history."""
    return experiment_service.get_version_history()
