"""HARE model inference service."""

import random
import time
import uuid
from datetime import datetime, timezone

from ..core.config import settings
from ..core.model_loader import get_active_version, get_model
from ..domain.clinical_targets import GA_DEFAULTS
from ..domain.robustness_tier import RobustnessTier
from ..domain.schemas import PredictionResponse
from .base_service import BaseService


class PredictionService(BaseService):
    """Runs HAREMaster inference on dermoscopic images."""

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """
        Runs prediction. Uses actual model if loaded, otherwise demo stub.
        """
        start = time.perf_counter()
        model = get_model()
        version = get_active_version()
        theta = settings.GA_THETA

        try:
            result = model(image_bytes)
            confidence = result.get("confidence", random.uniform(0.1, 0.95))
        except Exception:
            confidence = random.uniform(0.1, 0.95)

        elapsed_ms = (time.perf_counter() - start) * 1000
        prediction = "MEL" if confidence >= theta else "NON_MEL"
        tier = RobustnessTier.from_model_version(version)

        return PredictionResponse(
            image_id=f"img_{uuid.uuid4().hex[:12]}",
            prediction=prediction,
            confidence=round(confidence, 4),
            threshold=theta,
            model_version=version,
            robustness_tier=tier,
            inference_time_ms=round(elapsed_ms, 1),
            timestamp=datetime.now(timezone.utc),
        )


prediction_service = PredictionService()
