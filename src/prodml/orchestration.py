"""Mini Project 3, step 1: the small functions the Airflow DAGs call.

Airflow should only DECIDE the order and the retries. The real work lives here, in
plain Python that we can unit-test without Airflow installed. Each DAG task is a
one-line call into this file.

Two rules we follow for every task (Module 3, "idempotency"):
  * Running a task twice gives the same result as running it once. Our stages
    overwrite their own outputs, and registration checks whether the run is
    already registered.
  * Tasks pass only SMALL values (a run id, one number) through XCom. Big data
    stays on disk, and tasks pass its path.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from prodml import registry
from prodml.config import settings

logger = logging.getLogger("prodml.orchestration")

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_stage(stage: str, repo_root: Path | None = None) -> None:
    """Run one DVC stage (`python -m prodml.pipeline <stage>`) and fail loudly.

    A non-zero exit raises CalledProcessError, which Airflow turns into a task
    failure, and then into a retry (retries=2 in the DAG).
    """
    cmd = [sys.executable, "-m", "prodml.pipeline", stage]
    logger.info("running stage", extra={"stage": stage})
    subprocess.run(cmd, cwd=repo_root or REPO_ROOT, check=True)


def read_candidate(repo_root: Path | None = None) -> dict[str, Any]:
    """Collect what training produced: the MLflow run id, the family and the metrics.

    This is the value we push to XCom (three small fields, not a model).
    """
    root = repo_root or REPO_ROOT
    run = json.loads((root / "metrics" / "train_run.json").read_text())
    scores = json.loads((root / "metrics" / "scores.json").read_text())
    return {"run_id": run["run_id"], "family": run["family"], **scores}


def choose_branch(candidate_mae: float, margin: float | None = None) -> str:
    """The branching decision: register the candidate, or skip?

    Returns the task_id Airflow should follow. Lower MAE is better. With no
    Production model yet the candidate always wins (same rule as the CT workflow).
    """
    margin = settings.promotion_margin if margin is None else margin
    baseline = registry.production_metric("mae")
    if baseline is None or candidate_mae < baseline * (1 - margin):
        return "register_candidate"
    return "skip_registration"


def already_registered(run_id: str) -> int | None:
    """Return the model version made from this run, if one exists (idempotency)."""
    for v in registry.versions():
        if v.run_id == run_id:
            return int(v.version)
    return None


def register_candidate(run_id: str) -> dict[str, Any]:
    """Register the candidate and move it to Staging. Safe to call twice."""
    existing = already_registered(run_id)
    if existing is not None:
        logger.info("already registered", extra={"version": existing})
        return {"version": existing, "created": False}
    result = registry.promote_if_better(run_id, stage="Staging")
    return {"version": result.version, "created": result.promoted}
