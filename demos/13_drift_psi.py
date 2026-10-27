"""DEMO 13: data drift with PSI and KS, from scratch, on toy taxi data.

Idea: a model is only good on data that looks like its training data. If next month's
trips are longer, or come from other zones, accuracy can fall while the API still
answers 200. Drift detection compares "what the model sees now" with "what it was
trained on" and raises a flag BEFORE labels (the real durations) arrive.

Run:  uv run python demos/13_drift_psi.py

Files this technique needs in a real project (all exist in src/prodml/monitoring/):
  reference.parquet   frozen sample of the training data (never overwritten)
  serving_log.py      the API appends one line per request -> "current" data
  drift.py            psi_numeric / psi_categorical / compare   (this demo inlines them)
  store.py            keeps every run so you can ask "drifting for 2 days in a row?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import make_trips
from scipy import stats

EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: sum (cur% - ref%) * ln(cur% / ref%) over bins.

    Bins come from the REFERENCE quantiles, so each bin holds ~10% of the reference.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # anything outside training range counts
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), EPS, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def verdict(value: float) -> str:
    return "DRIFT" if value >= 0.25 else "watch" if value >= 0.10 else "ok"


def main() -> None:
    reference = make_trips(8000, seed=0)["trip_distance"].to_numpy()

    scenarios = {
        "same distribution": make_trips(2000, seed=1)["trip_distance"].to_numpy(),
        "trips 20% longer": make_trips(2000, seed=2)["trip_distance"].to_numpy() * 1.2,
        "trips 60% longer": make_trips(2000, seed=3)["trip_distance"].to_numpy() * 1.6,
    }
    rows = []
    for name, current in scenarios.items():
        ks = stats.ks_2samp(reference, current)
        value = psi(reference, current)
        rows.append((name, round(value, 3), verdict(value), f"{ks.pvalue:.1e}"))
    print(pd.DataFrame(rows, columns=["scenario", "PSI", "verdict", "KS p-value"]))

    print(
        "\nNotice: KS says 'different' (tiny p-value) even for a 20% shift, because\n"
        "with thousands of rows it detects ANY difference. PSI measures HOW MUCH it\n"
        "moved, so we alert on PSI and only report KS."
    )

    # What the bins look like: this is the picture behind the number.
    edges = np.quantile(reference, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    cur = scenarios["trips 60% longer"]
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    print("\nshare of trips per reference quintile (60% longer scenario):")
    for i, (a, b) in enumerate(zip(ref_share, cur_share), start=1):
        print(f"  bin {i}: reference {a:5.1%}   now {b:5.1%}")
    assert psi(reference, scenarios["same distribution"]) < 0.10
    assert psi(reference, cur) > 0.25


if __name__ == "__main__":
    main()
