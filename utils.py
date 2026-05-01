"""
Shared runtime utilities for the HARE Streamlit application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from hare_model import HAREThesisModel


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_LABELS = ["NV", "MEL"]
CLASS_DIAGNOSIS: Dict[str, str] = {
    "NV": "Lower concern melanoma screening signal",
    "MEL": "High concern melanoma screening signal",
}
CLASS_LONG_NAMES: Dict[str, str] = {
    "NV": "Melanocytic nevus pattern",
    "MEL": "Melanoma pattern",
}

THESIS_METRICS_ROWS: List[Dict[str, str]] = [
    {
        "Stage": "Stage 1",
        "Condition": "Clean",
        "Threshold": "Default (0.5)",
        "Balanced Accuracy": "0.8158",
        "Sensitivity (MEL)": "0.7807",
        "Specificity (non-MEL)": "0.8508",
        "F1": "0.6105",
        "AUC": "0.9001",
    },
    {
        "Stage": "Stage 2 (TRADES)",
        "Condition": "Clean",
        "Threshold": "Default (0.5)",
        "Balanced Accuracy": "0.7966",
        "Sensitivity (MEL)": "0.7272",
        "Specificity (non-MEL)": "0.8660",
        "F1": "0.5998",
        "AUC": "0.8856",
    },
    {
        "Stage": "Stage 2 (TRADES)",
        "Condition": "Clean",
        "Threshold": "GA (0.1372)",
        "Balanced Accuracy": "0.5517",
        "Sensitivity (MEL)": "0.9992",
        "Specificity (non-MEL)": "0.1042",
        "F1": "0.2999",
        "AUC": "0.8943",
    },
    {
        "Stage": "Stage 2 (TRADES)",
        "Condition": "Adversarial PGD-20",
        "Threshold": "GA (0.1372)",
        "Balanced Accuracy": "0.3771",
        "Sensitivity (MEL)": "0.7204",
        "Specificity (non-MEL)": "0.0339",
        "F1": "0.2134",
        "AUC": "0.1032",
    },
]

THESIS_ARCHITECTURE_POINTS = [
    "ResNet-50 backbone extracts 2048-dimensional local texture features.",
    "ViT-Small-Patch16-224 extracts 384-dimensional global structure features.",
    "Three heads are used: CNN head, ViT head, and a 512-unit GELU fusion head.",
    "TRADES fine-tuning uses the fusion head as the adversarial training target.",
    "Inference is calibrated with GA late fusion using alpha, tau, and theta.",
]

THESIS_FAILURE_MODE_POINTS = [
    "The thesis operating point prioritizes melanoma sensitivity over specificity.",
    "Under PGD-20 attack at epsilon = 0.03, melanoma sensitivity remains 0.7204.",
    "The dominant adversarial failure mode is false positives, not missed melanomas.",
    "This makes the system more appropriate for screening triage than for standalone diagnosis.",
]

preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


@dataclass
class RuntimeSettings:
    model_dir: str
    active_checkpoint: str
    ga_alpha: float
    ga_tau: float
    ga_theta: float


@dataclass
class PredictionResult:
    predicted_index: int
    predicted_label: str
    predicted_name: str
    melanoma_probability: float
    screening_confidence: float
    threshold: float
    screening_bucket: str
    summary: str
    recommendation: str
    branch_probabilities: Dict[str, np.ndarray]
    fusion_probabilities: np.ndarray
    late_fusion_probabilities: np.ndarray
    outputs: Dict[str, torch.Tensor]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_env_file(path: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def get_runtime_settings(base_dir: str) -> RuntimeSettings:
    env_values = parse_env_file(os.path.join(base_dir, ".env"))

    def read_value(key: str, default: str) -> str:
        return os.environ.get(key) or env_values.get(key) or default

    return RuntimeSettings(
        model_dir=read_value("MODEL_DIR", "./models"),
        active_checkpoint=read_value("ACTIVE_CHECKPOINT", "stage2_best.pth"),
        ga_alpha=float(read_value("GA_ALPHA", "0.0582")),
        ga_tau=float(read_value("GA_TAU", "1.8162")),
        ga_theta=float(read_value("GA_THETA", "0.1372")),
    )


def checkpoint_candidates(base_dir: str, settings: RuntimeSettings) -> List[str]:
    model_dir = os.path.normpath(os.path.join(base_dir, settings.model_dir))
    preferred = [
        settings.active_checkpoint,
        "stage2_best.pth",
        "hare_stage2_robust.pth",
        "hare_final.pth",
    ]

    candidates: List[str] = []
    for filename in preferred:
        if os.path.isabs(filename):
            candidates.append(filename)
        else:
            candidates.append(os.path.join(model_dir, filename))
            candidates.append(os.path.join(base_dir, filename))

    deduped: List[str] = []
    seen = set()
    for path in candidates:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(path)
    return deduped


def extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            payload = payload["model_state_dict"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]

    if not isinstance(payload, dict):
        raise TypeError("Checkpoint did not contain a state dict.")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        clean_key = key.replace("_orig_mod.", "").replace("module.", "")
        state_dict[clean_key] = value
    return state_dict


def load_model_runtime(
    base_dir: str,
) -> Tuple[HAREThesisModel, torch.device, RuntimeSettings, str]:
    settings = get_runtime_settings(base_dir)
    device = get_device()
    model = HAREThesisModel(num_classes=2, pretrained_backbones=False).to(device)
    model.eval()

    errors: List[str] = []
    for path in checkpoint_candidates(base_dir, settings):
        if not os.path.exists(path):
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = extract_state_dict(payload)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            message = "Loaded thesis-aligned checkpoint from: {name}".format(name=os.path.basename(path))
            if missing or unexpected:
                details = []
                if missing:
                    details.append("missing keys: {count}".format(count=len(missing)))
                if unexpected:
                    details.append("unexpected keys: {count}".format(count=len(unexpected)))
                message = "{base} ({details})".format(base=message, details=", ".join(details))
            return model, device, settings, message
        except Exception as exc:
            errors.append("{name}: {error}".format(name=os.path.basename(path), error=exc))

    if errors:
        error_text = " | ".join(errors[:3])
    else:
        error_text = "No compatible checkpoint found."
    raise RuntimeError(error_text)


def prepare_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    return preprocess(image).unsqueeze(0).to(device)


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    return torch.clamp(image_tensor, 0, 1).permute(1, 2, 0).numpy()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_from_upload(upload) -> Image.Image:
    return Image.open(upload).convert("RGB")


def late_fusion_probabilities(
    cnn_logits: torch.Tensor,
    vit_logits: torch.Tensor,
    alpha_cnn: float,
    tau: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha_cnn, 0.0, 1.0))
    temperature = float(max(tau, 1e-6))
    blended = (alpha * cnn_logits + (1.0 - alpha) * vit_logits) / temperature
    return torch.softmax(blended, dim=1).detach().cpu().numpy()[0]


def _screening_copy(predicted_label: str, melanoma_probability: float) -> Tuple[str, str, str]:
    if predicted_label == "MEL":
        return (
            "high_concern",
            "High concern melanoma screening signal",
            (
                "The model found a melanoma-like pattern and recommends prompt clinical review. "
                "Use this as screening support only, not as a diagnosis."
            ),
        )

    return (
        "lower_concern",
        "Lower concern melanoma screening signal",
        (
            "The model found a lower concern pattern on this image, but this does not rule out melanoma. "
            "Reassess clinically if the lesion is changing, symptomatic, or visually suspicious."
        ),
    )


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    settings: RuntimeSettings,
) -> PredictionResult:
    with torch.no_grad():
        outputs = model(image_tensor)

    cnn_probs = torch.softmax(outputs["cnn_logits"], dim=1).cpu().numpy()[0]
    vit_probs = torch.softmax(outputs["vit_logits"], dim=1).cpu().numpy()[0]
    fusion_probs = torch.softmax(outputs["fusion_logits"], dim=1).cpu().numpy()[0]
    late_probs = late_fusion_probabilities(
        outputs["cnn_logits"],
        outputs["vit_logits"],
        settings.ga_alpha,
        settings.ga_tau,
    )

    melanoma_probability = float(late_probs[1])
    predicted_index = 1 if melanoma_probability >= settings.ga_theta else 0
    predicted_label = CLASS_LABELS[predicted_index]
    screening_confidence = melanoma_probability if predicted_index == 1 else 1.0 - melanoma_probability
    bucket, summary, recommendation = _screening_copy(predicted_label, melanoma_probability)

    return PredictionResult(
        predicted_index=predicted_index,
        predicted_label=predicted_label,
        predicted_name=CLASS_DIAGNOSIS[predicted_label],
        melanoma_probability=melanoma_probability,
        screening_confidence=float(screening_confidence),
        threshold=settings.ga_theta,
        screening_bucket=bucket,
        summary=summary,
        recommendation=recommendation,
        branch_probabilities={
            "cnn": cnn_probs,
            "vit": vit_probs,
        },
        fusion_probabilities=fusion_probs,
        late_fusion_probabilities=late_probs,
        outputs=outputs,
    )


class GradCAM:
    """Grad-CAM for the ResNet branch using fusion-head decisions."""

    def __init__(self, model: HAREThesisModel) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.cnn.layer4[-1]

        def forward_hook(module, module_input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> Tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor)
        score = outputs["fusion_logits"][:, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients or activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, self._compute_dispersion_score(cam)

    @staticmethod
    def _compute_dispersion_score(heatmap: np.ndarray) -> float:
        threshold = np.median(heatmap)
        highlighted = (heatmap > threshold).sum()
        return float(highlighted / max(1, heatmap.size))


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size)
    )
    heatmap_color = cm.get_cmap("jet")(heatmap_resized / 255.0)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    image_np = np.array(image).astype(np.uint8)
    overlay = (alpha * heatmap_color + (1.0 - alpha) * image_np).astype(np.uint8)
    return Image.fromarray(overlay)


def _imagenet_normalized_bounds(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (0.0 - mean) / std, (1.0 - mean) / std


def fgsm_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    epsilon: float = 0.03,
) -> torch.Tensor:
    perturbed = image_tensor.clone().detach().requires_grad_(True)
    logits = model(perturbed)["fusion_logits"]
    loss = F.cross_entropy(logits, label_tensor)
    loss.backward()

    std = torch.tensor(IMAGENET_STD, device=perturbed.device).view(1, 3, 1, 1)
    scaled_epsilon = epsilon / std
    adv = perturbed + scaled_epsilon * perturbed.grad.sign()
    lower, upper = _imagenet_normalized_bounds(perturbed.device)
    adv = torch.max(torch.min(adv, upper), lower)
    return adv.detach()


def pgd_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    epsilon: float = 0.03,
    alpha: Optional[float] = None,
    steps: int = 20,
) -> torch.Tensor:
    step_size = alpha if alpha is not None else epsilon / 3.0
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(1, 3, 1, 1)
    eps_scaled = epsilon / std
    step_scaled = step_size / std
    x_orig = image_tensor.detach()
    x_adv = x_orig.clone().detach()
    lower, upper = _imagenet_normalized_bounds(image_tensor.device)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv)["fusion_logits"], label_tensor)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_scaled * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-eps_scaled, max=eps_scaled)
        x_adv = torch.max(torch.min(x_orig + delta, upper), lower).detach()

    return x_adv


def risk_banner(prediction: PredictionResult) -> Tuple[str, str]:
    if prediction.predicted_label == "MEL":
        return (
            "High concern screening signal. Prompt dermatologist or clinician review is recommended.",
            "error",
        )
    return (
        "Lower concern screening signal. Continue routine clinical review and reassess if the lesion changes.",
        "success",
    )
