"""MLflow helpers: one function logs a COMPLETE, reproducible run.

For every run we log
  params    all hyperparameters, the split strategy, the data version hash
  metrics   rmse, mae, r2, train_seconds, model_size_mb
  artifacts the model (pyfunc), residual plot, feature-importance plot, requirements.txt
  tags      git_commit, data_version, author, framework
Missing any of these = a run you cannot reproduce, so they are all logged in ONE place.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, containers)
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from prodml.config import settings  # noqa: E402
from prodml.models import FEATURE_COLUMNS, BaseFamily, bundle_size_mb  # noqa: E402
from prodml.pyfunc_model import RideDurationPyfunc  # noqa: E402

logger = logging.getLogger("prodml.tracking")


@dataclass
class RunResult:
    run_id: str
    run_name: str
    family: str
    metrics: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ setup & tags
def configure_mlflow(uri: str | None = None, experiment: str | None = None) -> None:
    mlflow.set_tracking_uri(uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment or settings.mlflow_experiment)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except Exception:
        return ""


def git_commit() -> str:
    """Current commit hash, with '-dirty' if there are uncommitted changes."""
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return "unknown"
    return sha + ("-dirty" if _git("status", "--porcelain") else "")


def git_author() -> str:
    return _git("config", "user.name") or "unknown"


def data_version(path: str | Path) -> str:
    """Identify the exact bytes of the dataset.

    With DVC: the md5 stored in `<path>.dvc` (this is the hash `dvc push` uploads).
    Without it: an md5 of the file itself, prefixed so you can tell the two apart.
    """
    path = Path(path)
    dvc_file = Path(f"{path}.dvc")
    if dvc_file.exists():
        for line in dvc_file.read_text().splitlines():
            line = line.strip().lstrip("- ").strip()
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
    if path.exists():
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return f"nodvc-{h.hexdigest()}"
    return "unknown"


def standard_tags(framework: str, data_path: str | Path) -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "data_version": data_version(data_path),
        "author": git_author(),
        "framework": framework,
    }


def pinned_requirements(packages: list[str]) -> list[str]:
    """['scikit-learn==1.9.1', ...] for the packages that are installed."""
    out = []
    for name in packages:
        try:
            out.append(f"{name}=={md.version(name)}")
        except md.PackageNotFoundError:
            continue
    return out


def freeze() -> str:
    """The exact versions of every installed package (works without pip, e.g. in uv)."""
    dists = sorted(
        {
            f"{d.metadata['Name']}=={d.version}"
            for d in md.distributions()
            if d.metadata["Name"]
        },
        key=str.lower,
    )
    return "\n".join(dists) + "\n"


MODEL_PIP_PACKAGES = {
    "linear": ["scikit-learn", "numpy", "pandas", "scipy"],
    "xgboost": ["scikit-learn", "xgboost", "numpy", "pandas", "scipy"],
    "random_forest": ["scikit-learn", "numpy", "pandas"],
}


# ------------------------------------------------------------------ plots
def residual_figure(y_true: np.ndarray, y_pred: np.ndarray):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(resid, bins=60, color="#4C78A8")
    axes[0].set_title("Residuals (actual - predicted, min)")
    axes[0].axvline(0, color="k", lw=1)
    idx = np.random.default_rng(0).choice(
        len(y_pred), min(3000, len(y_pred)), replace=False
    )
    axes[1].scatter(
        np.asarray(y_pred)[idx], resid[idx], s=4, alpha=0.4, color="#F58518"
    )
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("predicted (min)")
    axes[1].set_ylabel("residual (min)")
    axes[1].set_title("Residual vs predicted")
    fig.tight_layout()
    return fig


def importance_figure(names: list[str], values: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(names) + 1)))
    ax.barh(names[::-1], np.asarray(values)[::-1], color="#54A24B")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ the run
def log_family_run(
    family: BaseFamily,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    run_name: str,
    data_path: str | Path | None = None,
    nested: bool = False,
    extra_params: dict[str, Any] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> RunResult:
    """Fit `family` on `train`, score on `val`, and log everything to MLflow.

    `train`/`val` have the columns PU_DO, trip_distance and duration. Every family
    gets the SAME frames, so runs are comparable (same split = valid comparison).
    """
    data_path = data_path or settings.data_path
    X_tr, y_tr = train[FEATURE_COLUMNS], train["duration"].to_numpy()
    X_va, y_va = val[FEATURE_COLUMNS], val["duration"].to_numpy()

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(
            {**standard_tags(family.framework, data_path), **(extra_tags or {})}
        )
        mlflow.set_tag("family", family.name)

        mlflow.log_params(
            {
                **family.params,
                "split_strategy": "chronological",
                "train_ratio": settings.train_ratio,
                "n_train": len(train),
                "n_val": len(val),
                "data_version": data_version(data_path),
                **(extra_params or {}),
            }
        )

        start = time.perf_counter()
        family.fit(X_tr, y_tr, X_va, y_va)
        train_seconds = time.perf_counter() - start

        pred = family.predict_frame(X_va)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_va, pred))),
            "mae": float(mean_absolute_error(y_va, pred)),
            "r2": float(r2_score(y_va, pred)),
            "train_seconds": round(train_seconds, 3),
        }

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            family.save(bundle)
            metrics["model_size_mb"] = round(bundle_size_mb(bundle), 4)
            mlflow.log_metrics(metrics)

            sample = X_va.head(5)
            signature = mlflow.models.infer_signature(
                sample, family.predict_frame(sample)
            )
            mlflow.pyfunc.log_model(
                name="model",
                python_model=RideDurationPyfunc(),
                artifacts={"bundle": str(bundle)},
                signature=signature,
                pip_requirements=pinned_requirements(MODEL_PIP_PACKAGES[family.name]),
            )

        fig = residual_figure(y_va, pred)
        mlflow.log_figure(fig, "plots/residuals.png")
        plt.close(fig)

        sub = np.random.default_rng(0).choice(
            len(X_va), min(20_000, len(X_va)), replace=False
        )
        names, values = family.feature_importance(X_va.iloc[sub], y_va[sub])
        fig = importance_figure(names, values, f"{family.name}: feature importance")
        mlflow.log_figure(fig, "plots/feature_importance.png")
        plt.close(fig)

        mlflow.log_text(freeze(), "requirements.txt")

        logger.info(
            "run logged",
            extra={"run_id": run.info.run_id, "run_name": run_name, **metrics},
        )
        return RunResult(
            run_id=run.info.run_id,
            run_name=run_name,
            family=family.name,
            metrics=metrics,
            params=dict(family.params),
        )
