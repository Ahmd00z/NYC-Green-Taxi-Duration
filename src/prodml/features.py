"""Feature engineering. ONE function builds features for BOTH training and serving,
so the two can never drift apart (no training/serving skew)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def build_features(
    pickup: int | str | None, dropoff: int | str | None, trip_distance: float
) -> dict[str, Any]:
    """One trip -> the feature dict the DictVectorizer understands.

    A missing pickup/dropoff becomes "unknown" instead of crashing.
    """
    pu = "unknown" if pickup in (None, "") else pickup
    do = "unknown" if dropoff in (None, "") else dropoff
    return {"PU_DO": f"{pu}_{do}", "trip_distance": float(trip_distance)}


def dataframe_to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Apply build_features to every row of a DataFrame."""
    return [
        build_features(pu, do, dist)
        for pu, do, dist in zip(
            df["PULocationID"], df["DOLocationID"], df["trip_distance"]
        )
    ]


def extract_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple[Any, np.ndarray, Any, np.ndarray, DictVectorizer]:
    """Fit the vectorizer on train only, then transform train and validation."""
    dv = DictVectorizer()
    X_train = dv.fit_transform(dataframe_to_dicts(df_train))
    X_val = dv.transform(dataframe_to_dicts(df_val))
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, y_train, X_val, y_val, dv
