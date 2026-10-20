"""Read params.yaml (the file DVC tracks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from prodml.config import settings


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or settings.params_file) as f:
        return yaml.safe_load(f) or {}
