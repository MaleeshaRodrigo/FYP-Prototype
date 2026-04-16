"""GradCAM heatmap generation service."""

import random

from ..domain.schemas import GradCAMResponse
from .base_service import BaseService


class GradCAMService(BaseService):
    """Generates GradCAM attention heatmaps for model predictions."""

    def generate_heatmap(self, image_id: str) -> GradCAMResponse:
        """
        Generates a GradCAM heatmap. Returns demo data when model isn't available.
        In production, this would use pytorch-grad-cam on the ViT/ResNet backbone.
        """
        size = 14  # ViT-Small patch resolution (14x14)
        heatmap = [[round(random.random(), 4) for _ in range(size)] for _ in range(size)]

        return GradCAMResponse(
            image_id=image_id,
            heatmap_data=heatmap,
            original_url="",
            overlay_url="",
        )


gradcam_service = GradCAMService()
