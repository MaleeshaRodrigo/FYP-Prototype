"""GET/PUT /api/system/models, /api/system/parameters — System admin endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..core.config import settings
from ..core.model_loader import set_active_version
from ..core.runtime_state import get_active_version, get_ga_parameters, update_ga_parameters
from ..core.security import require_role
from ..domain.clinical_targets import GA_DEFAULTS
from ..domain.schemas import GAParametersRequest, GAParametersResponse, ModelRegistryEntry

router = APIRouter(prefix="/api/system", tags=["system"])


def _load_model_versions() -> list[dict]:
    constants_path = Path(__file__).resolve().parents[2] / "shared" / "constants.json"
    if constants_path.exists():
        try:
            payload = json.loads(constants_path.read_text(encoding="utf-8"))
            versions = payload.get("modelVersions", [])
            if isinstance(versions, list):
                return [v for v in versions if isinstance(v, dict)]
        except Exception:
            pass
    return [
        {"id": "stage1", "label": "Stage 1 — Clean Baseline", "checkpoint": "stage1_baseline.pth", "stage": 1},
        {"id": "v8", "label": "Stage 2 v8 — PGD-AT (w_adv=0.05)", "checkpoint": "stage2_v8.pth", "stage": 2},
        {"id": "v9-trades", "label": "Stage 3 v9 — TRADES", "checkpoint": "stage3_v9_trades.pth", "stage": 3},
    ]


_MODEL_VERSIONS = _load_model_versions()


def _to_response(ga: dict[str, float], applied_bal_acc: float = 0.7980) -> GAParametersResponse:
    return GAParametersResponse(
        success=True,
        applied_bal_acc=applied_bal_acc,
        alpha=float(ga["weight_cnn"]),
        tau=float(ga["temperature"]),
        theta=float(ga["threshold"]),
        weight_cnn=float(ga["weight_cnn"]),
        temperature=float(ga["temperature"]),
        threshold=float(ga["threshold"]),
    )


@router.get("/models", response_model=list[ModelRegistryEntry])
async def get_model_registry(
    _user: dict = Depends(require_role("system")),
) -> list[ModelRegistryEntry]:
    """Returns all registered model checkpoints."""
    active = get_active_version()
    entries: list[ModelRegistryEntry] = []
    for item in _MODEL_VERSIONS:
        version_id = str(item.get("id", ""))
        stage = int(item.get("stage", 0))
        is_active = version_id == active
        status = "active" if is_active else ("pending" if "trades" in version_id else "deprecated")
        entries.append(
            ModelRegistryEntry(
                id=version_id,
                label=str(item.get("label", version_id)),
                checkpoint=str(item.get("checkpoint", "")),
                stage=stage,
                status=status,
                isActive=is_active,
                isPending=status == "pending",
            )
        )
    return entries


@router.get("/parameters", response_model=GAParametersResponse)
async def get_parameters(
    _user: dict = Depends(require_role("system")),
) -> GAParametersResponse:
    """Returns currently active GA parameters."""
    ga = get_ga_parameters()
    return _to_response(ga)


@router.put("/parameters", response_model=GAParametersResponse)
async def update_parameters(
    params: GAParametersRequest,
    _user: dict = Depends(require_role("system")),
) -> GAParametersResponse:
    """Updates the GA-optimised ensemble parameters."""
    payload = {
        "weight_cnn": params.alpha,
        "temperature": params.tau,
        "threshold": params.theta,
    }
    ga = update_ga_parameters(payload)
    return _to_response(ga)


@router.put("/models/{version_id}/activate")
async def activate_model(
    version_id: str,
    _user: dict = Depends(require_role("system")),
) -> dict:
    """Activates a model checkpoint as the primary inference model."""
    selected = next((m for m in _MODEL_VERSIONS if str(m.get("id")) == version_id), None)
    if not selected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown model version '{version_id}'")

    checkpoint = str(selected.get("checkpoint", settings.ACTIVE_CHECKPOINT))
    checkpoint_path = settings.MODEL_DIR / checkpoint
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint not found for '{version_id}' at {checkpoint_path}",
        )

    set_active_version(version_id, checkpoint=checkpoint)
    return {"success": True, "activated": version_id, "checkpoint": checkpoint}
