"""Experiment version history service."""

import json
from pathlib import Path

from .base_service import BaseService

FIXTURES_PATH = Path(__file__).parent.parent.parent / "shared" / "mock-fixtures"


class ExperimentService(BaseService):
    """Provides the AT experiment version history."""

    def __init__(self):
        self._history = self._load_history()

    def _load_history(self) -> list[dict]:
        history_file = FIXTURES_PATH / "version-history.json"
        if history_file.exists():
            with open(history_file) as f:
                return json.load(f)
        return []

    def get_version_history(self) -> list[dict]:
        return self._history


experiment_service = ExperimentService()
