"""HARE model inference service."""

import time
import uuid
from datetime import datetime, timezone

import torch

from ..core.model_loader import ga_fuse_logits, get_active_version, get_model, preprocess_image_bytes
from ..core.runtime_state import get_ga_parameters
from ..domain.robustness_tier import RobustnessTier
from ..domain.schemas import PredictionResponse
from .base_service import BaseService


class PredictionService(BaseService):
    """Runs HAREMaster inference on dermoscopic images."""

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Runs inference with notebook-aligned GA fusion parameters."""
        start = time.perf_counter()
        model = get_model()
        version = get_active_version()
        ga = get_ga_parameters()
        threshold = float(ga["threshold"])

        try:
            inputs = preprocess_image_bytes(image_bytes)
            with torch.no_grad():
                outputs = model(inputs)

            if all(key in outputs for key in ("cnn_logits", "vit_logits", "fusion_logits")):
                probs = ga_fuse_logits(
                    outputs["cnn_logits"],
                    outputs["vit_logits"],
                    weight_cnn=ga["weight_cnn"],
                    temperature=ga["temperature"],
                )
                confidence = float(probs[:, 1].item())
            else:
                logits = outputs["fusion_logits"]
                confidence = float(torch.softmax(logits, dim=1)[:, 1].item())
        except Exception:
            confidence = 0.5

        elapsed_ms = (time.perf_counter() - start) * 1000
        prediction = "MEL" if confidence >= threshold else "NON_MEL"
        tier = RobustnessTier.from_model_version(version)

        return PredictionResponse(
            image_id=f"img_{uuid.uuid4().hex[:12]}",
            prediction=prediction,
            confidence=round(confidence, 4),
            threshold=threshold,
            model_version=version,
            robustness_tier=tier,
            inference_time_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc),
        )


prediction_service = PredictionService()
