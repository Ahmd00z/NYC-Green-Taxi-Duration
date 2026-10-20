"""The DVC stages:  prepare -> featurize -> train -> evaluate   (+ a data check).

Each stage is a plain function so it is unit-testable. The command line is what
pipelines/dvc.yaml calls:

    python -m prodml.pipeline prepare
    python -m prodml.pipeline featurize
    python -m prodml.pipeline train
    python -m prodml.pipeline evaluate
    python -m prodml.pipeline validate        (used by the CT workflow before training)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prodml.config import settings
from prodml.data import read_dataframe, split_data
from prodml.experiments import to_feature_frame
from prodml.logging_conf import setup_logging
from prodml.models import FEATURE_COLUMNS, load_bundle, make_family
from prodml.params import load_params
from prodml.tracking import configure_mlflow, log_family_run

logger = logging.getLogger("prodml.pipeline")

PROCESSED = Path("data/processed")
FEATURES = Path("data/features")
CANDIDATE = Path("models/candidate")
METRICS = Path("metrics")

REQUIRED_COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


# ------------------------------------------------------------------ stages
def prepare(raw_path: str, train_ratio: float, out: Path = PROCESSED) -> None:
    """raw data -> cleaned, chronologically split train/val parquet."""
    df = read_dataframe(raw_path)
    train, val = split_data(df, train_ratio)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    logger.info("prepare done", extra={"train": len(train), "val": len(val)})


def featurize(src: Path = PROCESSED, out: Path = FEATURES) -> None:
    """Apply build_features() to every trip -> the two columns the model sees."""
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        df = pd.read_parquet(src / f"{split}.parquet")
        to_feature_frame(df).to_parquet(out / f"{split}.parquet", index=False)
    logger.info("featurize done")


def train(
    params: dict,
    raw_path: str,
    src: Path = FEATURES,
    candidate: Path = CANDIDATE,
    metrics_dir: Path = METRICS,
) -> str:
    """Train ONE family (params.yaml: train.model) as a tracked MLflow run.

    Saves the model bundle to models/candidate and the run id to metrics/train_run.json,
    so the CT workflow can find the exact run later.
    """
    family_name = params["train"]["model"]
    family = make_family(family_name, params.get(family_name))
    train_df = pd.read_parquet(src / "train.parquet")
    val_df = pd.read_parquet(src / "val.parquet")

    configure_mlflow()
    result = log_family_run(
        family,
        train_df,
        val_df,
        run_name=f"dvc-{family_name}",
        data_path=raw_path,
        extra_tags={"trigger": "dvc-pipeline"},
    )

    if candidate.exists():
        shutil.rmtree(candidate)
    family.save(candidate)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "train_run.json").write_text(
        json.dumps({"run_id": result.run_id, "family": family_name}, indent=2)
    )
    logger.info("train done", extra={"run_id": result.run_id, **result.metrics})
    return result.run_id


def evaluate(
    src: Path = FEATURES, candidate: Path = CANDIDATE, metrics_dir: Path = METRICS
) -> dict[str, float]:
    """Score the saved candidate on the validation split -> metrics/scores.json.

    Computed from the saved bundle (not from MLflow), so `dvc repro` reproduces it.
    """
    val = pd.read_parquet(src / "val.parquet")
    model = load_bundle(candidate)
    pred = model.predict_frame(val[FEATURE_COLUMNS])
    y = val["duration"].to_numpy()
    scores = {
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    logger.info("evaluate done", extra=scores)
    return scores


def validate(
    raw_path: str, min_rows: int = 1000, max_null_rate: float = 0.05
) -> list[str]:
    """Cheap data checks before spending time training. Returns a list of problems."""
    path = Path(raw_path)
    if not path.exists():
        return [f"{raw_path} does not exist (did `dvc pull` run?)"]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    else:
        df = pd.read_csv(path, usecols=REQUIRED_COLUMNS)

    problems: list[str] = []
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (< {min_rows})")
    for col in REQUIRED_COLUMNS:
        rate = float(df[col].isna().mean())
        if rate > max_null_rate:
            problems.append(f"{col}: {rate:.1%} nulls (> {max_null_rate:.0%})")
    if (df["trip_distance"] < 0).any():
        problems.append("negative trip_distance values")
    return problems


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(prog="prodml.pipeline")
    ap.add_argument(
        "stage", choices=["prepare", "featurize", "train", "evaluate", "validate"]
    )
    args = ap.parse_args(argv)
    params = load_params()
    raw = settings.data_path

    if args.stage == "prepare":
        prepare(raw, params["data"]["train_ratio"])
    elif args.stage == "featurize":
        featurize()
    elif args.stage == "train":
        train(params, raw)
    elif args.stage == "evaluate":
        evaluate()
    elif args.stage == "validate":
        problems = validate(raw)
        for p in problems:
            logger.error("data validation failed", extra={"problem": p})
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
