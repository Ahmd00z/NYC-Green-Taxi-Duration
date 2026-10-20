"""Model quality gate for CI. Exits 1 (= red check) if the candidate is clearly worse.

  python -m prodml.gate --candidate metrics/scores.json --baseline-file base_scores.json
  python -m prodml.gate --candidate metrics/scores.json --baseline-registry

Rule: FAIL if  candidate_mae > baseline_mae * (1 + max_regression)  (default 5%).
No baseline (first model ever) -> PASS with a note, so the very first PR can merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prodml.config import settings


def check_regression(
    candidate: float, baseline: float | None, max_regression: float
) -> tuple[bool, str]:
    """(passed, message). Lower is better."""
    if baseline is None:
        return (
            True,
            f"no baseline to compare against: candidate {candidate:.4f} accepted",
        )
    limit = baseline * (1 + max_regression)
    change = (candidate - baseline) / baseline
    if candidate > limit:
        return False, (
            f"REGRESSION: {candidate:.4f} is {change:+.1%} vs baseline {baseline:.4f} "
            f"(allowed: +{max_regression:.0%})"
        )
    return True, f"OK: {candidate:.4f} is {change:+.1%} vs baseline {baseline:.4f}"


def read_metric(path: str | Path, metric: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    return float(json.loads(p.read_text())[metric])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prodml.gate")
    ap.add_argument("--candidate", required=True, help="metrics json of the new model")
    ap.add_argument(
        "--baseline-file", help="metrics json of the baseline (e.g. from main)"
    )
    ap.add_argument(
        "--baseline-registry", action="store_true", help="use MLflow Production"
    )
    ap.add_argument("--metric", default=settings.promote_metric)
    ap.add_argument(
        "--max-regression", type=float, default=settings.gate_max_regression
    )
    args = ap.parse_args(argv)

    cand = read_metric(args.candidate, args.metric)
    if cand is None:
        sys.stderr.write(f"gate: candidate file {args.candidate} not found\n")
        return 1

    if args.baseline_registry:
        from prodml.registry import production_metric

        base = production_metric(args.metric)
    elif args.baseline_file:
        base = read_metric(args.baseline_file, args.metric)
    else:
        base = None

    ok, msg = check_regression(cand, base, args.max_regression)
    sys.stdout.write(f"gate: {msg}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
