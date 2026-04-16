"""GET /api/metrics/{version} — Model performance metrics."""

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

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


@router.get("/metrics/comparison/{baseline_version}/{candidate_version}")
async def get_metrics_comparison(
    baseline_version: str,
    candidate_version: str,
    _user: dict = Depends(get_current_user),
) -> dict:
    """Returns side-by-side metrics comparison and deltas."""
    return metrics_service.get_comparison(baseline_version, candidate_version)


@router.get("/metrics/thesis/summary")
async def get_thesis_summary(
    _user: dict = Depends(get_current_user),
) -> dict:
    """Returns canonical thesis result blocks for Stage 1/2 views."""
    return metrics_service.get_thesis_summary()


@router.get("/metrics/thesis/sweep")
async def get_thesis_sweep(
    _user: dict = Depends(get_current_user),
) -> dict:
    """Returns epsilon sweep values used in thesis robustness curve."""
    return metrics_service.get_robustness_sweep()


@router.get("/metrics/thesis/trades-beta-sweep")
async def get_trades_beta_sweep(
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """Returns TRADES beta sweep points for charting."""
    return metrics_service.get_trades_beta_sweep()


@router.get("/metrics/thesis/export/json")
async def export_thesis_json(
    _user: dict = Depends(get_current_user),
) -> dict:
    """Returns consolidated thesis export payload as JSON."""
    return metrics_service.get_thesis_export_json()


@router.get("/metrics/thesis/export/csv", response_class=PlainTextResponse)
async def export_thesis_csv(
    _user: dict = Depends(get_current_user),
) -> str:
    """Returns canonical thesis summary as CSV text."""
    return metrics_service.get_thesis_export_csv()
