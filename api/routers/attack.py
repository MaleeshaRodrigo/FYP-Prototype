"""POST /api/attack/simulate — PGD adversarial attack simulation."""

from fastapi import APIRouter, Depends

from ..core.security import get_current_user
from ..domain.schemas import AttackRequest, AttackResponse
from ..services.attack_service import attack_service

router = APIRouter(prefix="/api", tags=["attack"])


@router.post("/attack/simulate", response_model=AttackResponse)
async def simulate_attack(
    request: AttackRequest,
    _user: dict = Depends(get_current_user),
) -> AttackResponse:
    """Simulates a PGD adversarial attack against the HARE model."""
    return attack_service.simulate_attack(
        epsilon=request.epsilon,
        pgd_steps=request.pgd_steps,
        pgd_alpha=request.pgd_alpha,
    )
