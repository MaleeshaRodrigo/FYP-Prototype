"""PGD adversarial attack simulation service."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.model_loader import (
    clamp_normalized,
    ga_fuse_logits,
    get_imagenet_stats,
    get_model,
    preprocess_image_bytes,
)
from ..core.runtime_state import get_ga_parameters
from ..domain.schemas import AttackResponse
from .base_service import BaseService


class AttackService(BaseService):
    """Runs notebook-aligned PGD attacks against the active model."""

    def _pgd_attack(
        self,
        model,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float,
        alpha: float,
        steps: int,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()

        x_orig = images.detach()
        x_adv = clamp_normalized(x_orig + 0.001 * torch.randn_like(x_orig))
        _, std = get_imagenet_stats(images.device)

        eps_normalized = epsilon / std
        step_normalized = alpha / std

        try:
            for _ in range(steps):
                x_adv.requires_grad_(True)
                with torch.enable_grad():
                    adv_logits = model(x_adv)["fusion_logits"]
                    loss = F.cross_entropy(adv_logits, labels)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    x_adv = x_adv + step_normalized * grad.sign()
                    delta = torch.clamp(x_adv - x_orig, -eps_normalized, eps_normalized)
                    x_adv = clamp_normalized(x_orig + delta).detach()
        finally:
            if was_training:
                model.train()

        return x_adv

    @staticmethod
    def _predict_ga(outputs: dict[str, torch.Tensor], ga_params: dict[str, float]) -> tuple[str, float]:
        probs = ga_fuse_logits(
            outputs["cnn_logits"],
            outputs["vit_logits"],
            weight_cnn=ga_params["weight_cnn"],
            temperature=ga_params["temperature"],
        )
        mel_prob = float(probs[:, 1].item())
        prediction = "MEL" if mel_prob >= float(ga_params["threshold"]) else "NON_MEL"
        return prediction, mel_prob

    def simulate_attack(
        self,
        image_bytes: bytes,
        epsilon: float = 0.01,
        pgd_steps: int = 10,
        pgd_alpha: float = 0.003,
        true_label: int | None = None,
    ) -> AttackResponse:
        """Runs PGD-linf attack and returns clean/adversarial predictions."""
        model = get_model()
        ga_params = get_ga_parameters()

        clean_inputs = preprocess_image_bytes(image_bytes)
        with torch.no_grad():
            clean_outputs = model(clean_inputs)

        original_prediction, original_confidence = self._predict_ga(clean_outputs, ga_params)
        if true_label is None:
            true_label = 1 if original_prediction == "MEL" else 0

        labels = torch.tensor([int(true_label)], dtype=torch.long, device=clean_inputs.device)
        x_adv = self._pgd_attack(
            model=model,
            images=clean_inputs,
            labels=labels,
            epsilon=epsilon,
            alpha=pgd_alpha,
            steps=pgd_steps,
        )

        with torch.no_grad():
            adv_outputs = model(x_adv)
        adversarial_prediction, adversarial_confidence = self._predict_ga(adv_outputs, ga_params)

        attack_success = original_prediction != adversarial_prediction
        perturbation_l_inf = float((x_adv - clean_inputs).abs().max().item())

        return AttackResponse(
            original_prediction=original_prediction,
            original_confidence=round(float(original_confidence), 4),
            adversarial_prediction=adversarial_prediction,
            adversarial_confidence=round(float(adversarial_confidence), 4),
            epsilon=epsilon,
            pgd_steps=pgd_steps,
            attack_success=attack_success,
            perturbation_l_inf=round(perturbation_l_inf, 6),
            true_label=int(true_label),
            used_ga=True,
        )


attack_service = AttackService()
