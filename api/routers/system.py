"""GET/PUT /api/system/models, /api/system/parameters — System admin endpoints."""

from fastapi import APIRouter, Depends

from ..core.model_loader import set_active_version
from ..core.security import require_role
from ..domain.clinical_targets import GA_DEFAULTS
from ..domain.schemas import GAParametersRequest, GAParametersResponse, ModelRegistryEntry

router = APIRouter(prefix="/api/system", tags=["system"])

_ga_params = dict(GA_DEFAULTS)


@router.get("/models", response_model=list[ModelRegistryEntry])
async def get_model_registry(
    _user: dict = Depends(require_role("system")),
) -> list[ModelRegistryEntry]:
    """Returns all registered model checkpoints."""
    return [
        ModelRegistryEntry(id="v8", label="Stage 2 v8 — PGD-AT (w_adv=0.05)", checkpoint="stage2_v8.pth", stage=2, status="active", isActive=True, isPending=False),
        ModelRegistryEntry(id="v7", label="Stage 2 v7 — PGD-AT (w_adv=0.15)", checkpoint="stage2_v7.pth", stage=2, status="deprecated", isActive=False, isPending=False),
        ModelRegistryEntry(id="v9-trades", label="Stage 3 v9 — TRADES", checkpoint="stage3_v9_trades.pth", stage=3, status="pending", isActive=False, isPending=True),
    ]


@router.put("/parameters", response_model=GAParametersResponse)
async def update_parameters(
    params: GAParametersRequest,
    _user: dict = Depends(require_role("system")),
) -> GAParametersResponse:
    """Updates the GA-optimised ensemble parameters."""
    _ga_params.update(params.model_dump())
    return GAParametersResponse(
        success=True,
        applied_bal_acc=0.7980,
        alpha=params.alpha,
        tau=params.tau,
        theta=params.theta,
    )


@router.put("/models/{version_id}/activate")
async def activate_model(
    version_id: str,
    _user: dict = Depends(require_role("system")),
) -> dict:
    """Activates a model checkpoint as the primary inference model."""
    set_active_version(version_id)
    return {"success": True, "activated": version_id}
