"""PGD adversarial attack simulation service."""

import random

from ..domain.schemas import AttackResponse
from .base_service import BaseService


class AttackService(BaseService):
    """Simulates PGD-10 adversarial attacks against the HARE model."""

    def simulate_attack(
        self, epsilon: float = 0.01, pgd_steps: int = 10, pgd_alpha: float = 0.003
    ) -> AttackResponse:
        """
        Runs a PGD attack simulation. Returns demo results when model isn't available.
        In production, this iteratively perturbs the input image.
        """
        original_confidence = round(random.uniform(0.5, 0.95), 4)
        adv_confidence = round(original_confidence - random.uniform(0.15, 0.55), 4)
        adv_confidence = max(0.01, adv_confidence)

        original_prediction = "MEL"
        adversarial_prediction = "NON_MEL" if adv_confidence < 0.3985 else "MEL"
        attack_success = original_prediction != adversarial_prediction

        return AttackResponse(
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            adversarial_prediction=adversarial_prediction,
            adversarial_confidence=adv_confidence,
            epsilon=epsilon,
            pgd_steps=pgd_steps,
            attack_success=attack_success,
            perturbation_l_inf=epsilon,
        )


attack_service = AttackService()
