"""Persistent runtime state for active model and GA parameters."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..domain.clinical_targets import GA_DEFAULTS
from .config import settings


_STATE_FILE = Path(__file__).resolve().parents[2] / ".hare_runtime_state.json"
_LOCK = threading.RLock()
_CACHE: dict[str, Any] | None = None


def _default_state() -> dict[str, Any]:
    return {
        "active_version": "v8",
        "active_checkpoint": settings.ACTIVE_CHECKPOINT,
        "ga_parameters": {
            "weight_cnn": float(settings.GA_ALPHA),
            "temperature": float(settings.GA_TAU),
            "threshold": float(settings.GA_THETA),
        },
    }


def _normalize_ga_params(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "weight_cnn": float(raw.get("weight_cnn", raw.get("alpha", GA_DEFAULTS["alpha"]))),
        "temperature": float(raw.get("temperature", raw.get("tau", GA_DEFAULTS["tau"]))),
        "threshold": float(raw.get("threshold", raw.get("theta", GA_DEFAULTS["theta"]))),
    }


def _normalize_state(data: dict[str, Any]) -> dict[str, Any]:
    default = _default_state()
    state = {
        "active_version": str(data.get("active_version", default["active_version"])),
        "active_checkpoint": str(data.get("active_checkpoint", default["active_checkpoint"])),
        "ga_parameters": _normalize_ga_params(data.get("ga_parameters", {})),
    }
    return state


def _read_state_file() -> dict[str, Any]:
    if not _STATE_FILE.exists():
        return _default_state()
    try:
        payload = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return _normalize_state(payload)
    except Exception:
        pass
    return _default_state()


def _write_state_file(state: dict[str, Any]) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ensure_loaded() -> dict[str, Any]:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = _read_state_file()
            _write_state_file(_CACHE)
        return dict(_CACHE)


def get_runtime_state() -> dict[str, Any]:
    return _ensure_loaded()


def get_ga_parameters() -> dict[str, float]:
    return dict(_ensure_loaded()["ga_parameters"])


def update_ga_parameters(params: dict[str, Any]) -> dict[str, float]:
    global _CACHE
    with _LOCK:
        state = _ensure_loaded()
        state["ga_parameters"] = _normalize_ga_params(params)
        _CACHE = state
        _write_state_file(state)
        return dict(state["ga_parameters"])


def get_active_version() -> str:
    return str(_ensure_loaded()["active_version"])


def get_active_checkpoint() -> str:
    return str(_ensure_loaded()["active_checkpoint"])


def set_active_version(version_id: str, checkpoint: str | None = None) -> dict[str, Any]:
    global _CACHE
    with _LOCK:
        state = _ensure_loaded()
        state["active_version"] = str(version_id)
        if checkpoint:
            state["active_checkpoint"] = str(checkpoint)
        _CACHE = state
        _write_state_file(state)
        return dict(state)