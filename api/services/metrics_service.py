"""Model performance metrics service."""

import json
from pathlib import Path

from .base_service import BaseService

FIXTURES_PATH = Path(__file__).parent.parent.parent / "shared" / "mock-fixtures"


class MetricsService(BaseService):
    """Provides model performance metrics from stored evaluation results."""

    def __init__(self):
        self._metrics = self._load_metrics()

    def _load_metrics(self) -> dict:
        metrics_file = FIXTURES_PATH / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
        return {}

    def get_metrics(self, version: str) -> dict:
        """Returns metrics for a given model version key."""
        key = version.replace("-", "_")
        if key in self._metrics:
            return self._metrics[key]

        clean_key = f"{key}_clean"
        if clean_key in self._metrics:
            return self._metrics[clean_key]

        return {"error": f"Metrics for version '{version}' not found"}


metrics_service = MetricsService()
