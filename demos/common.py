"""Shared toy data for the demos. No dependency on the prodml package.

The toy problem mimics the real one: predict trip duration (minutes) from the pickup
zone, the drop-off zone and the distance. It is synthetic, so the numbers only show how
each technique works, never how good a taxi model is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "_out"  # every demo writes only here
OUT.mkdir(exist_ok=True)


def make_trips(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """n fake trips: duration = 2 + 3 * distance + a small zone effect + noise."""
    rng = np.random.default_rng(seed)
    pu = rng.integers(1, 30, n)
    do = rng.integers(1, 30, n)
    distance = rng.lognormal(0.8, 0.7, n).clip(0.2, 30).round(2)
    zone_effect = (pu % 5) * 0.4 + (do % 3) * 0.3
    duration = 2 + 3 * distance + zone_effect + rng.normal(0, 1.5, n)
    return pd.DataFrame(
        {
            "PULocationID": pu,
            "DOLocationID": do,
            "trip_distance": distance,
            "duration": duration.clip(1, None).round(2),
        }
    )


def to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw trips -> the model columns. ONE function for training and serving."""
    return pd.DataFrame(
        {
            "pu": df["PULocationID"].astype(float),
            "do": df["DOLocationID"].astype(float),
            "distance": df["trip_distance"].astype(float),
        }
    )


def split(df: pd.DataFrame, ratio: float = 0.8):
    """Time-ordered split: first part trains, last part validates."""
    cut = int(len(df) * ratio)
    return df.iloc[:cut], df.iloc[cut:]