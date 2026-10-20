"""Model registry: None -> Staging -> Production, and the promotion logic.

CLI (used by you and by the CI workflows):
  python -m prodml.registry show
  python -m prodml.registry register           --run-id RUN
  python -m prodml.registry set-stage          --version 1 --stage Staging
  python -m prodml.registry promote-if-better  --run-id RUN [--margin 0.0]
  python -m prodml.registry promote-production --version 1

Note: MLflow marks model "stages" as deprecated (aliases are the replacement).
They still work and the course uses them. Everything stage-related is in THIS
file, so moving to aliases later means editing one module.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from prodml.config import settings
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.registry")

STAGES = ("None", "Staging", "Production", "Archived")


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    metric: str
    candidate_run_id: str
    candidate_value: float
    version: int | None = None
    stage: str | None = None
    baseline_version: int | None = None
    baseline_value: float | None = None


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _stage_call(fn, *args, **kwargs):
    """Stage APIs emit a FutureWarning (deprecated, not removed): keep logs clean."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return fn(*args, **kwargs)


def versions(name: str | None = None) -> list[ModelVersion]:
    name = name or settings.registered_model_name
    return sorted(
        _client().search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)
    )


def version_in_stage(stage: str, name: str | None = None) -> ModelVersion | None:
    """Highest version currently in `stage`, or None."""
    in_stage = [v for v in versions(name) if v.current_stage == stage]
    return in_stage[-1] if in_stage else None


def metric_of(version: ModelVersion, metric: str) -> float:
    return float(_client().get_run(version.run_id).data.metrics[metric])


def register_run(run_id: str, name: str | None = None) -> ModelVersion:
    """Register runs:/<run_id>/model. The same run twice reuses the version."""
    name = name or settings.registered_model_name
    for v in versions(name):
        if v.run_id == run_id:
            return v
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.register_model(f"runs:/{run_id}/model", name)


def set_stage(version: int, stage: str, archive_existing: bool = False) -> ModelVersion:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    mv = _stage_call(
        _client().transition_model_version_stage,
        settings.registered_model_name,
        str(version),
        stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("stage changed", extra={"version": int(version), "stage": stage})
    return mv


def promote_if_better(
    candidate_run_id: str,
    metric: str | None = None,
    stage: str = "Staging",
    margin: float | None = None,
) -> PromotionResult:
    """Promote the candidate to `stage` ONLY if it beats the current Production model.

    Lower is better (mae / rmse). "Beats" means  candidate < baseline * (1 - margin).
    No Production model yet -> the candidate wins by default.
    A failed challenge is NOT an error: it returns promoted=False and the CLI exits 0.
    """
    metric = metric or settings.promote_metric
    margin = settings.promotion_margin if margin is None else margin
    client = _client()
    cand = float(client.get_run(candidate_run_id).data.metrics[metric])
    prod = version_in_stage("Production")

    base = metric_of(prod, metric) if prod else None
    if base is not None and not cand < base * (1 - margin):
        res = PromotionResult(
            promoted=False,
            reason=f"{metric} {cand:.4f} does not beat Production {base:.4f} "
            f"(required margin {margin:.1%})",
            metric=metric,
            candidate_run_id=candidate_run_id,
            candidate_value=cand,
            baseline_version=int(prod.version),
            baseline_value=base,
        )
        logger.info("promotion rejected", extra=asdict(res))
        return res

    mv = register_run(candidate_run_id)
    # Archives older versions in the SAME target stage (Staging), never Production.
    set_stage(int(mv.version), stage, archive_existing=True)
    res = PromotionResult(
        promoted=True,
        reason=(
            "no Production model yet"
            if base is None
            else f"{metric} {cand:.4f} beats Production {base:.4f}"
        ),
        metric=metric,
        candidate_run_id=candidate_run_id,
        candidate_value=cand,
        version=int(mv.version),
        stage=stage,
        baseline_version=int(prod.version) if prod else None,
        baseline_value=base,
    )
    logger.info("promotion accepted", extra=asdict(res))
    return res


def promote_to_production(version: int) -> ModelVersion:
    """The human-approved step: make `version` Production (archives the old one)."""
    return set_stage(version, "Production", archive_existing=True)


def production_metric(metric: str | None = None) -> float | None:
    prod = version_in_stage("Production")
    return metric_of(prod, metric or settings.promote_metric) if prod else None


# ----------------------------------------------------------------------------- CLI
def _show() -> None:
    client = _client()
    _out(f"{'ver':>3}  {'stage':<11} {'mae':>8} {'rmse':>8}  run")
    for v in versions():
        m = client.get_run(v.run_id).data.metrics
        mae = f"{m['mae']:.4f}" if "mae" in m else "-"
        rmse = f"{m['rmse']:.4f}" if "rmse" in m else "-"
        _out(
            f"{v.version:>3}  {v.current_stage:<11} {mae:>8} {rmse:>8}  {v.run_id[:8]}"
        )


def _write_github_output(res: PromotionResult) -> None:
    """Let a GitHub Actions step read the outcome (promoted / version)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"promoted={str(res.promoted).lower()}\n")
        f.write(f"version={res.version if res.version is not None else ''}\n")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(prog="prodml.registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    p = sub.add_parser("register")
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("set-stage")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p = sub.add_parser("promote-if-better")
    p.add_argument("--run-id", required=True)
    p.add_argument("--stage", default="Staging")
    p.add_argument("--margin", type=float, default=None)
    p = sub.add_parser("promote-production")
    p.add_argument("--version", type=int, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "show":
        _show()
    elif args.cmd == "register":
        _out(str(register_run(args.run_id).version))
    elif args.cmd == "set-stage":
        set_stage(args.version, args.stage)
    elif args.cmd == "promote-if-better":
        res = promote_if_better(args.run_id, stage=args.stage, margin=args.margin)
        _out(json.dumps(asdict(res)))
        _write_github_output(res)
    elif args.cmd == "promote-production":
        promote_to_production(args.version)
    return 0  # a failed challenge is not a failed build


if __name__ == "__main__":
    sys.exit(main())
