"""The serving log: one JSON line per scored trip.

Metrics tell you THAT something moved (cheap, aggregated). This log keeps the actual
rows, so the drift job can later compare "what the API really received" with the
training data. A failure here must NEVER break a prediction, so errors are swallowed
(and logged).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from prodml.config import settings

logger = logging.getLogger("prodml.serving_log")
_lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool


def append(rows: Iterable[dict], path: str | None = None) -> None:
    """rows: dicts with PULocationID, DOLocationID, trip_distance, prediction."""
    path = path or settings.serving_log_path
    if not path:  # empty path = logging switched off
        return
    now = time.time()
    try:
        lines = "".join(json.dumps({"ts": now, **row}) + "\n" for row in rows)
        with _lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(lines)
    except OSError:
        logger.exception("could not write serving log")


def read(path: str | None = None, last_rows: int | None = None) -> pd.DataFrame:
    """The log as a DataFrame (empty if the file does not exist yet)."""
    path = path or settings.serving_log_path
    if not path or not Path(path).exists():
        return pd.DataFrame(
            columns=[
                "ts",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "prediction",
            ]
        )
    df = pd.read_json(path, lines=True)
    return df.tail(last_rows) if last_rows else df
