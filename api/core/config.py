"""Application configuration loaded from environment variables."""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


_ROOT_DIR = Path(__file__).resolve().parents[2]
_API_DIR = Path(__file__).resolve().parents[1]

# Load repository and API-local environment files if present.
load_dotenv(_ROOT_DIR / ".env", override=False)
load_dotenv(_API_DIR / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [value.strip() for value in raw.split(",") if value.strip()]


class Settings:
    APP_NAME: str = "HARE Platform API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = _get_bool("DEBUG", True)

    # Model paths
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "./models"))
    ACTIVE_CHECKPOINT: str = os.getenv("ACTIVE_CHECKPOINT", "stage2_v8.pth")

    # GA-optimised defaults
    GA_THETA: float = _get_float("GA_THETA", 0.3985)
    GA_TAU: float = _get_float("GA_TAU", 0.7671)
    GA_ALPHA: float = _get_float("GA_ALPHA", 0.5467)

    # JWT settings
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = _get_int("JWT_EXPIRY_MINUTES", 60)

    # CORS
    CORS_ORIGINS: list[str] = _get_list(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
    )

    # Azure Blob Storage (for model checkpoints)
    BLOB_CONNECTION_STRING: str = os.getenv("BLOB_CONNECTION_STRING", "")

    def __init__(self):
        if not self.JWT_SECRET:
            if self.DEBUG:
                # In development, create an ephemeral key to avoid a committed default secret.
                self.JWT_SECRET = secrets.token_urlsafe(48)
            else:
                raise ValueError("JWT_SECRET must be set in .env when DEBUG is false")


settings = Settings()
