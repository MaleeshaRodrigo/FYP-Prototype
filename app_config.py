"""
Configuration helpers for the HARE thesis-demo application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent


def parse_env_file(path: Path = BASE_DIR / ".env") -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV_FILE_VALUES = parse_env_file()


def get_setting(key: str, default: str = "") -> str:
    return os.environ.get(key) or _ENV_FILE_VALUES.get(key) or default


@dataclass(frozen=True)
class AppConfig:
    database_url: str
    azure_storage_connection_string: str
    azure_image_container: str
    app_secret_key: str
    admin_bootstrap_email: str
    admin_bootstrap_password: str
    local_upload_dir: Path


def load_app_config() -> AppConfig:
    return AppConfig(
        database_url=get_setting("DATABASE_URL"),
        azure_storage_connection_string=get_setting("AZURE_STORAGE_CONNECTION_STRING"),
        azure_image_container=get_setting("AZURE_IMAGE_CONTAINER", "hare-images"),
        app_secret_key=get_setting("APP_SECRET_KEY", "hare-thesis-demo-local-secret"),
        admin_bootstrap_email=get_setting("ADMIN_BOOTSTRAP_EMAIL"),
        admin_bootstrap_password=get_setting("ADMIN_BOOTSTRAP_PASSWORD"),
        local_upload_dir=BASE_DIR / "data" / "uploads",
    )
