"""Application configuration loaded from environment variables."""

import os
from pathlib import Path


class Settings:
    APP_NAME: str = "HARE Platform API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Model paths
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "./models"))
    ACTIVE_CHECKPOINT: str = os.getenv("ACTIVE_CHECKPOINT", "stage2_v8.pth")

    # GA-optimised defaults
    GA_THETA: float = float(os.getenv("GA_THETA", "0.3985"))
    GA_TAU: float = float(os.getenv("GA_TAU", "0.7671"))
    GA_ALPHA: float = float(os.getenv("GA_ALPHA", "0.5467"))

    # JWT settings
    JWT_SECRET: str = os.getenv("JWT_SECRET", "hare-dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

    # CORS
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,*"
    ).split(",")

    # Azure Blob Storage (for model checkpoints)
    BLOB_CONNECTION_STRING: str = os.getenv("BLOB_CONNECTION_STRING", "")


settings = Settings()
