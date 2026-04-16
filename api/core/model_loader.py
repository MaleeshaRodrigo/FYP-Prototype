"""HAREMaster model loading and management."""

import logging
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

_loaded_model = None
_active_version = "v8"


def load_model(checkpoint_path: Path | None = None):
    """
    Loads the HAREMaster checkpoint.
    In demo/dev mode, returns a stub if PyTorch model files aren't available.
    """
    global _loaded_model
    path = checkpoint_path or settings.MODEL_DIR / settings.ACTIVE_CHECKPOINT

    if path.exists():
        try:
            import torch
            _loaded_model = torch.load(path, map_location="cpu")
            logger.info("Model loaded from %s", path)
        except Exception:
            logger.warning("Failed to load model from %s; using demo stub", path)
            _loaded_model = _create_demo_stub()
    else:
        logger.info("Checkpoint not found at %s; using demo stub", path)
        _loaded_model = _create_demo_stub()

    return _loaded_model


def get_model():
    """Returns the currently loaded model, loading demo stub if needed."""
    global _loaded_model
    if _loaded_model is None:
        _loaded_model = _create_demo_stub()
    return _loaded_model


def get_active_version() -> str:
    return _active_version


def set_active_version(version: str):
    global _active_version
    _active_version = version


def _create_demo_stub():
    """Returns a demo stub that mimics the model interface for development."""
    class DemoModel:
        def __call__(self, x):
            import random
            confidence = random.uniform(0.1, 0.95)
            return {"confidence": confidence}

        def eval(self):
            return self

    return DemoModel()
