"""Generate a SYNTHETIC stand-in for one month of NYC green taxi trips.

Same columns as the real TLC file. Use it ONLY to smoke-test the pipeline when you do
not have the real parquet. Metrics on this data mean nothing.

    python scripts/make_synthetic_trips.py data/synthetic_trips.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 40_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pu = rng.choice(np.arange(1, 90), n)
    do = rng.choice(np.arange(1, 90), n)
    raw = np.abs(pu - do) * 0.12 + rng.gamma(2.0, 0.8, n)
    distance = np.clip(raw, 0.1, 40).round(2)
    speed = np.clip(rng.normal(12, 3, n), 4, 30)  # mph
    duration_min = np.clip(distance / speed * 60 + 3 + rng.normal(0, 2, n), 1.5, 80)
    pickup = pd.Timestamp("2019-08-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 30 * 24 * 60, n)), unit="m"
    )
    return pd.DataFrame(
        {
            "lpep_pickup_datetime": pickup,
            "lpep_dropoff_datetime": pickup + pd.to_timedelta(duration_min, unit="m"),
            "PULocationID": pu.astype("int64"),
            "DOLocationID": do.astype("int64"),
            "trip_distance": distance,
        }
    )


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_trips.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_parquet(out, index=False)
    sys.stdout.write(f"wrote {out}\n")
