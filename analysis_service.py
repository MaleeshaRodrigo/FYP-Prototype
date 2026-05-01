"""
Persisted analysis and robustness helpers.
"""

from __future__ import annotations

import torch

from database import db, utc_now
from utils import pgd_attack, predict


def run_robust_analysis(model, device, settings, image_tensor, image_id: int, user_id: int):
    db.audit("analysis_requested", actor_user_id=user_id, target_resource=f"image:{image_id}")
    prediction = predict(model, image_tensor, settings)
    label_tensor = torch.tensor([prediction.predicted_index], device=device)
    adversarial_tensor = pgd_attack(
        model,
        image_tensor,
        label_tensor,
        epsilon=0.03,
        alpha=0.01,
        steps=10,
    )
    adversarial_prediction = predict(model, adversarial_tensor, settings)
    robustness_status = (
        "Verified Robust"
        if adversarial_prediction.predicted_label == prediction.predicted_label
        else "Robustness Warning"
    )
    result_id = db.execute_returning_id(
        """
        INSERT INTO analysis_results (
            image_id, user_id, predicted_label, predicted_summary, melanoma_probability,
            confidence_score, ga_threshold, robustness_status, robustness_attack, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PGD-10', ?)
        """,
        (
            image_id,
            user_id,
            prediction.predicted_label,
            prediction.summary,
            prediction.melanoma_probability,
            prediction.screening_confidence,
            settings.ga_theta,
            robustness_status,
            utc_now(),
        ),
    )
    db.audit(
        "model_result",
        actor_user_id=user_id,
        target_resource=f"analysis:{result_id}",
        details={
            "image_id": image_id,
            "label": prediction.predicted_label,
            "melanoma_probability": prediction.melanoma_probability,
        },
    )
    db.audit(
        "robustness_check",
        actor_user_id=user_id,
        target_resource=f"analysis:{result_id}",
        success=robustness_status == "Verified Robust",
        details={"attack": "PGD-10", "status": robustness_status},
    )
    return prediction, adversarial_prediction, robustness_status, result_id
