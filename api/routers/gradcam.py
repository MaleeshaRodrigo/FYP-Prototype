"""GET /api/gradcam/{image_id} — GradCAM heatmap generation."""

from fastapi import APIRouter, Depends

from ..core.security import get_current_user
from ..domain.schemas import GradCAMResponse
from ..services.gradcam_service import gradcam_service

router = APIRouter(prefix="/api", tags=["gradcam"])


@router.get("/gradcam/{image_id}", response_model=GradCAMResponse)
async def get_gradcam(
    image_id: str,
    _user: dict = Depends(get_current_user),
) -> GradCAMResponse:
    """Generates a GradCAM heatmap for the given image."""
    return gradcam_service.generate_heatmap(image_id)
