"""HAREMaster runtime loading and shared inference utilities."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .config import settings
from .runtime_state import (
    get_active_checkpoint,
    get_active_version as get_runtime_active_version,
    set_active_version as set_runtime_active_version,
)

try:
    import timm  # type: ignore
except Exception:
    timm = None

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CLASSES = 2
DROPOUT = 0.35

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

_loaded_model: nn.Module | None = None
_loaded_checkpoint: Path | None = None


class HAREMaster(nn.Module):
    """Notebook-aligned model with branch heads and fusion logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.cnn = models.resnet50(weights=None)
        cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        if timm is None:
            raise RuntimeError("timm is required to build HAREMaster")
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        vit_dim = self.vit.num_features

        self.cnn_head = nn.Linear(cnn_dim, num_classes)
        self.vit_head = nn.Linear(vit_dim, num_classes)
        self.fusion = nn.Sequential(nn.Linear(cnn_dim + vit_dim, 512), nn.GELU(), nn.Dropout(dropout))
        self.fusion_head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        fusion_logits = self.fusion_head(self.fusion(torch.cat([cnn_feat, vit_feat], dim=1)))
        return {
            "cnn_logits": self.cnn_head(cnn_feat),
            "vit_logits": self.vit_head(vit_feat),
            "fusion_logits": fusion_logits,
        }


class DemoModel(nn.Module):
    """Fallback model when runtime artifacts are unavailable."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mel_score = torch.sigmoid(x.mean(dim=(1, 2, 3)))
        logits = torch.stack([1.0 - mel_score, mel_score], dim=1)
        return {
            "cnn_logits": logits,
            "vit_logits": logits,
            "fusion_logits": logits,
        }


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            payload = payload["model_state_dict"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint format")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            normalized = str(key).replace("_orig_mod.", "").replace("module.", "")
            state_dict[normalized] = value
    if not state_dict:
        raise ValueError("No tensor weights found in checkpoint")
    return state_dict


def _create_model() -> nn.Module:
    if timm is None:
        logger.warning("timm is not available; using DemoModel")
        return DemoModel()
    model = HAREMaster(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    model.eval()
    return model


def _current_checkpoint_path(checkpoint_path: Path | None = None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path)
    return settings.MODEL_DIR / get_active_checkpoint()


def load_model(checkpoint_path: Path | None = None, force_reload: bool = False) -> nn.Module:
    """Loads the active checkpoint and returns a callable model."""
    global _loaded_model, _loaded_checkpoint
    path = _current_checkpoint_path(checkpoint_path)

    if _loaded_model is not None and _loaded_checkpoint == path and not force_reload:
        return _loaded_model

    if not path.exists():
        logger.warning("Checkpoint not found at %s; using DemoModel", path)
        _loaded_model = DemoModel()
        _loaded_checkpoint = path
        return _loaded_model

    try:
        payload = torch.load(path, map_location=DEVICE, weights_only=False)
        model = _create_model()
        if isinstance(model, DemoModel):
            _loaded_model = model
            _loaded_checkpoint = path
            return _loaded_model

        state_dict = _extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys while loading %s: %s", path.name, missing[:5])
        if unexpected:
            logger.warning("Unexpected keys while loading %s: %s", path.name, unexpected[:5])

        model.eval()
        _loaded_model = model
        _loaded_checkpoint = path
        logger.info("Model loaded from %s on %s", path, DEVICE)
    except Exception as exc:
        logger.exception("Failed to load model at %s, using DemoModel: %s", path, exc)
        _loaded_model = DemoModel()
        _loaded_checkpoint = path

    return _loaded_model


def get_model() -> nn.Module:
    """Returns loaded model, lazy-loading active checkpoint when needed."""
    global _loaded_model
    if _loaded_model is None:
        return load_model()
    return _loaded_model


def get_active_version() -> str:
    return get_runtime_active_version()


def set_active_version(version: str, checkpoint: str | None = None) -> None:
    set_runtime_active_version(version, checkpoint=checkpoint)
    load_model(force_reload=True)


def color_constancy(image: Image.Image) -> Image.Image:
    """Applies a simple shades-of-gray style correction used in training notebooks."""
    arr = np.asarray(image).astype(np.float32)
    mean_rgb = arr.mean(axis=(0, 1))
    avg_gray = float(mean_rgb.mean())
    scale = avg_gray / (mean_rgb + 1e-6)
    for channel in range(3):
        arr[:, :, channel] *= scale[channel]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Converts bytes to normalized batch tensor on runtime device."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = color_constancy(image)
    tensor = _TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return tensor


def get_imagenet_stats(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    target = device or DEVICE
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=target).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=target).view(1, 3, 1, 1)
    return mean, std


def clamp_normalized(images: torch.Tensor) -> torch.Tensor:
    """Clamps normalized tensor to valid pixel range after denormalization."""
    mean, std = get_imagenet_stats(images.device)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    return torch.max(torch.min(images, upper), lower)


def ga_fuse_logits(
    cnn_logits: torch.Tensor,
    vit_logits: torch.Tensor,
    weight_cnn: float,
    temperature: float,
) -> torch.Tensor:
    """Notebook-aligned 2-branch logit-level fusion."""
    fused = float(weight_cnn) * cnn_logits + (1.0 - float(weight_cnn)) * vit_logits
    return torch.softmax(fused / float(temperature), dim=1)
