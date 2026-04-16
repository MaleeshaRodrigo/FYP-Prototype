"""POST /api/attack/simulate — PGD adversarial attack simulation."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..core.security import get_current_user
from ..domain.schemas import AttackResponse
from ..services.attack_service import attack_service

router = APIRouter(prefix="/api", tags=["attack"])


@router.post("/attack/simulate", response_model=AttackResponse)
async def simulate_attack(
    image: UploadFile = File(...),
    epsilon: float = Form(default=0.01),
    pgd_steps: int = Form(default=10),
    pgd_alpha: float = Form(default=0.003),
    true_label: int | None = Form(default=None),
    _user: dict = Depends(get_current_user),
) -> AttackResponse:
    """Runs a real PGD adversarial attack against uploaded dermoscopic image."""
    if not (0 < epsilon <= 0.1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="epsilon must be in (0, 0.1]",
        )
    if not (1 <= pgd_steps <= 100):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pgd_steps must be in [1, 100]",
        )
    if not (0 < pgd_alpha <= 0.01):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pgd_alpha must be in (0, 0.01]",
        )
    if true_label is not None and true_label not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="true_label must be 0 or 1",
        )

    image_bytes = await image.read()
    return attack_service.simulate_attack(
        image_bytes=image_bytes,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        pgd_alpha=pgd_alpha,
        true_label=true_label,
    )
