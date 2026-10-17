"""Loading the raw trips and splitting them. No features, no models here."""

import logging

import pandas as pd

from prodml.utils import timed

logger = logging.getLogger("prodml.data")


@timed
def read_dataframe(path: str) -> pd.DataFrame:
    """Read one month of NYC green taxi trips (.parquet or .csv/.csv.gz).

    Adds a `duration` column in minutes and keeps only trips of 1-60 minutes.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    logger.info("data loaded", extra={"path": path, "rows": len(df)})
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the first 80% of trips train, the last 20% validate."""
    df = df.sort_values("lpep_pickup_datetime").reset_index(drop=True)
    cutoff = int(len(df) * train_ratio)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
