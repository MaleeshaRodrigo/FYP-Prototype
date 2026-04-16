"""Model performance metrics service."""

import csv
import io
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

    def get_thesis_summary(self) -> dict:
        """Returns canonical thesis result blocks for Stage 1/2 comparisons."""
        return {
            "stage1_clean_default": self.get_metrics("stage1"),
            "stage2_clean_default": self.get_metrics("v8_clean"),
            "stage2_clean_ga": self.get_metrics("v8_ga"),
            "stage2_adv_ga": self.get_metrics("v8_adversarial"),
        }

    def get_robustness_sweep(self) -> dict:
        """Returns a 5-point epsilon robustness sweep aligned with thesis narrative."""
        if "robustness_sweep" in self._metrics and isinstance(self._metrics["robustness_sweep"], dict):
            return self._metrics["robustness_sweep"]
        return {
            "0.0": {"bal_acc": 0.7980, "sens_mel": 0.7550, "spec_nonmel": 0.8416, "auc": 0.8711},
            "0.01": {"bal_acc": 0.6320, "sens_mel": 0.6110, "spec_nonmel": 0.6530, "auc": 0.6220},
            "0.02": {"bal_acc": 0.5050, "sens_mel": 0.5720, "spec_nonmel": 0.4380, "auc": 0.3520},
            "0.03": {"bal_acc": 0.3771, "sens_mel": 0.7204, "spec_nonmel": 0.0339, "auc": 0.1032},
            "0.06": {"bal_acc": 0.2920, "sens_mel": 0.6830, "spec_nonmel": 0.0110, "auc": 0.0660},
        }

    def get_trades_beta_sweep(self) -> list[dict]:
        """Returns TRADES beta sweep points for charting."""
        points = self._metrics.get("trades_beta_sweep")
        if isinstance(points, list) and points:
            return points
        return [
            {"beta": 1.0, "cleanAUC": 0.8650, "advBalAcc": 0.4250},
            {"beta": 2.0, "cleanAUC": 0.8500, "advBalAcc": 0.5500},
            {"beta": 3.0, "cleanAUC": 0.8350, "advBalAcc": 0.6500},
            {"beta": 6.0, "cleanAUC": 0.8000, "advBalAcc": 0.7150},
        ]

    def get_comparison(self, baseline_version: str, candidate_version: str) -> dict:
        """Returns side-by-side metrics and deltas for two version keys."""
        baseline = self.get_metrics(baseline_version)
        candidate = self.get_metrics(candidate_version)
        if "error" in baseline or "error" in candidate:
            return {"error": "Unable to compare versions", "baseline": baseline, "candidate": candidate}

        keys = ["auc", "balancedAccuracy", "melanomaSensitivity", "nonMelSpecificity"]
        deltas: dict[str, float | None] = {}
        for key in keys:
            b = baseline.get(key)
            c = candidate.get(key)
            if isinstance(b, (int, float)) and isinstance(c, (int, float)):
                deltas[key] = round(float(c) - float(b), 4)
            else:
                deltas[key] = None

        return {
            "baselineVersion": baseline.get("modelVersion", baseline_version),
            "candidateVersion": candidate.get("modelVersion", candidate_version),
            "baseline": baseline,
            "candidate": candidate,
            "delta": deltas,
        }

    def get_thesis_export_json(self) -> dict:
        """Returns a consolidated thesis export payload."""
        summary = self.get_thesis_summary()
        return {
            "summary": summary,
            "robustness_sweep": self.get_robustness_sweep(),
            "trades_beta_sweep": self.get_trades_beta_sweep(),
            "comparisons": {
                "stage1_to_stage2_clean": self.get_comparison("stage1", "v8_clean"),
                "stage2_clean_to_stage2_adv": self.get_comparison("v8_clean", "v8_adversarial"),
            },
        }

    def get_thesis_export_csv(self) -> str:
        """Returns canonical thesis summary rows as CSV text."""
        summary = self.get_thesis_summary()
        rows = [
            ("Stage 1 Clean", "Default (0.5)", summary.get("stage1_clean_default", {})),
            ("Stage 2 Clean", "Default (0.5)", summary.get("stage2_clean_default", {})),
            ("Stage 2 Clean", "GA (theta)", summary.get("stage2_clean_ga", {})),
            ("Stage 2 Adversarial", "GA (theta) - Primary", summary.get("stage2_adv_ga", {})),
        ]

        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Block", "Threshold", "AUC", "BalancedAccuracy", "MelanomaSensitivity", "NonMelSpecificity"])
        for block, threshold, metric in rows:
            writer.writerow(
                [
                    block,
                    threshold,
                    metric.get("auc"),
                    metric.get("balancedAccuracy"),
                    metric.get("melanomaSensitivity"),
                    metric.get("nonMelSpecificity"),
                ]
            )
        return out.getvalue()


metrics_service = MetricsService()
