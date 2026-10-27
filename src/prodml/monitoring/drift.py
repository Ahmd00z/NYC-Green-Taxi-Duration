"""Data drift: has the live data moved away from what the model was trained on?

Two simple, well-known tests, written with numpy/scipy only (no extra library):

  PSI  Population Stability Index. Cut the REFERENCE into bins, count how much of the
       reference and of the current data falls in each bin, and add up
       (cur% - ref%) * ln(cur% / ref%).  Rule of thumb: < 0.10 stable,
       0.10-0.25 watch it, > 0.25 drifted.  It measures HOW MUCH things moved.
  KS   Kolmogorov-Smirnov two-sample test (numeric only). Gives a p-value. With lots of
       rows it flags tiny, harmless shifts, so we REPORT it but decide with PSI.

Numeric columns are binned on the reference quantiles (each bin starts with ~10% of
the reference). Equal-width bins break on skewed data such as trip distance.
Categorical columns (zone ids) are compared by the share of each value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from prodml.config import settings

EPS = 1e-4  # an empty bin would make ln(0); this is the usual fix

NUMERIC = ["trip_distance", "prediction"]
CATEGORICAL = ["PULocationID", "DOLocationID"]


def _psi_from_shares(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, EPS, None)
    cur = np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # a constant column cannot be binned
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf  # values outside training range still count
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    return _psi_from_shares(ref_share, cur_share)


def psi_categorical(reference: pd.Series, current: pd.Series, top: int = 30) -> float:
    """Top-N values of the reference keep their own bin; everything else is 'other'.
    A value never seen in training lands in 'other', which is what we want to see."""
    keep = reference.value_counts().head(top).index
    ref = reference.where(reference.isin(keep), "other").value_counts(normalize=True)
    cur = current.where(current.isin(keep), "other").value_counts(normalize=True)
    labels = ref.index.union(cur.index)
    return _psi_from_shares(
        ref.reindex(labels, fill_value=0).to_numpy(),
        cur.reindex(labels, fill_value=0).to_numpy(),
    )


def classify(psi: float) -> str:
    if psi >= settings.psi_drift:
        return "drift"
    return "warn" if psi >= settings.psi_warn else "ok"


@dataclass
class FeatureDrift:
    feature: str
    kind: str  # numeric | categorical
    psi: float
    ks_stat: float | None
    ks_p: float | None
    status: str  # ok | warn | drift

    def as_dict(self) -> dict:
        return asdict(self)


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    min_rows: int | None = None,
) -> list[FeatureDrift]:
    """Drift of every monitored column that exists in BOTH frames."""
    min_rows = settings.min_drift_rows if min_rows is None else min_rows
    if len(current) < min_rows:
        raise ValueError(
            f"only {len(current)} current rows (need {min_rows}): "
            "too few to judge drift; collect more traffic"
        )
    out: list[FeatureDrift] = []
    for col in NUMERIC:
        if col in reference and col in current:
            ref, cur = reference[col].dropna(), current[col].dropna()
            value = psi_numeric(ref.to_numpy(), cur.to_numpy())
            ks = stats.ks_2samp(ref, cur)
            out.append(
                FeatureDrift(
                    col,
                    "numeric",
                    round(value, 4),
                    round(float(ks.statistic), 4),
                    float(ks.pvalue),
                    classify(value),
                )
            )
    for col in CATEGORICAL:
        if col in reference and col in current:
            value = psi_categorical(reference[col].dropna(), current[col].dropna())
            out.append(
                FeatureDrift(
                    col, "categorical", round(value, 4), None, None, classify(value)
                )
            )
    return out


def overall(results: list[FeatureDrift]) -> str:
    """The worst status among the features."""
    order = {"ok": 0, "warn": 1, "drift": 2}
    return max((r.status for r in results), key=order.__getitem__, default="ok")
