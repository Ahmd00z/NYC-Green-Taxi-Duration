"""Mini Project 2, Step 2:  python -m prodml.experiments

Trains and tracks THREE model families on the SAME chronological split:
  linear   (the baseline you always keep),  xgboost,  a scikit-learn random forest
plus an XGBoost hyperparameter sweep (Optuna, nested runs) and an autolog demo run.
Default: 1 + 1 + 1 + (1 parent + 10 trials) + 1 autolog = 15 runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import optuna
import pandas as pd

from prodml.config import settings
from prodml.data import read_dataframe, split_data
from prodml.features import dataframe_to_dicts
from prodml.logging_conf import setup_logging
from prodml.models import FEATURE_COLUMNS, XGBoostFamily, make_family
from prodml.params import load_params
from prodml.tracking import (
    RunResult,
    configure_mlflow,
    log_family_run,
    standard_tags,
)

logger = logging.getLogger("prodml.experiments")


def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw trips -> the two columns every model sees (+ the target).

    Uses the SAME build_features() as the API, so training and serving agree.
    """
    feats = pd.DataFrame(dataframe_to_dicts(df))
    feats["duration"] = df["duration"].to_numpy()
    return feats[[*FEATURE_COLUMNS, "duration"]]


def load_splits(
    data_path: str | Path | None = None, train_ratio: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_dataframe(str(data_path or settings.data_path))
    train, val = split_data(df, train_ratio or settings.train_ratio)
    return to_feature_frame(train), to_feature_frame(val)


def run_autolog_demo(train: pd.DataFrame, val: pd.DataFrame, data_path: str) -> str:
    """Turn on mlflow.xgboost.autolog() for ONE run: see what it logs for free."""
    mlflow.xgboost.autolog(log_models=False, log_datasets=False, silent=True)
    try:
        with mlflow.start_run(run_name="xgboost-autolog-demo") as run:
            mlflow.set_tags(standard_tags("xgboost", data_path))
            XGBoostFamily({"n_estimators": 50, "max_depth": 6}).fit(
                train[FEATURE_COLUMNS],
                train["duration"].to_numpy(),
                val[FEATURE_COLUMNS],
                val["duration"].to_numpy(),
            )
    finally:
        mlflow.xgboost.autolog(disable=True)

    client = mlflow.MlflowClient()
    data = client.get_run(run.info.run_id).data
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    logger.info(
        "autolog captured",
        extra={
            "n_params": len(data.params),
            "n_metrics": len(data.metrics),
            "metric_names": sorted(data.metrics)[:6],
            "artifacts": artifacts,
        },
    )
    return run.info.run_id


def sweep_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    data_path: str,
    n_trials: int = 10,
    seed: int = 42,
) -> RunResult | None:
    """Optuna sweep: one PARENT run, one NESTED run per trial. Returns the best."""
    best: dict[str, RunResult] = {}

    with mlflow.start_run(run_name="xgboost-sweep"):
        mlflow.set_tags(standard_tags("xgboost", data_path))
        mlflow.set_tag("family", "xgboost")
        mlflow.log_params({"n_trials": n_trials, "sampler": "TPE", "sweep_seed": seed})

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.03, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            res = log_family_run(
                XGBoostFamily(params),
                train,
                val,
                run_name=f"trial-{trial.number:02d}",
                data_path=data_path,
                nested=True,
            )
            best[str(trial.number)] = res
            return res.metrics["mae"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_metric("best_mae", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        return best[str(study.best_trial.number)]


def run_all(
    data_path: str | None = None, n_trials: int = 10, include_rf: bool = True
) -> list[RunResult]:
    setup_logging()
    configure_mlflow()
    data_path = str(data_path or settings.data_path)
    params = load_params()
    train, val = load_splits(data_path)
    logger.info("splits ready", extra={"n_train": len(train), "n_val": len(val)})

    results: list[RunResult] = []
    kw = {"train": train, "val": val, "data_path": data_path}

    results.append(
        log_family_run(make_family("linear"), run_name="linear-baseline", **kw)
    )
    results.append(
        log_family_run(
            make_family("xgboost", params.get("xgboost")),
            run_name="xgboost-default",
            **kw,
        )
    )

    if include_rf:
        results.append(
            log_family_run(
                make_family("random_forest", params.get("random_forest")),
                run_name="random-forest",
                **kw,
            )
        )

    best_sweep = sweep_xgboost(train, val, data_path, n_trials=n_trials)
    if best_sweep:
        results.append(best_sweep)
    run_autolog_demo(train, val, data_path)

    ranked = sorted(results, key=lambda r: r.metrics["mae"])
    for r in ranked:
        logger.info(
            "result",
            extra={
                "run": r.run_name,
                "mae": round(r.metrics["mae"], 4),
                "run_id": r.run_id,
            },
        )
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + track all model families")
    ap.add_argument(
        "--trials", type=int, default=10, help="XGBoost sweep trials (min 10)"
    )
    ap.add_argument("--no-rf", action="store_true", help="skip the random forest")
    args = ap.parse_args()
    run_all(n_trials=args.trials, include_rf=not args.no_rf)


if __name__ == "__main__":
    main()
