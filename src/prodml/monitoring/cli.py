"""Command line for module 4.   python -m prodml.monitoring <command> --help

freeze    save a small frozen sample of the TRAINING data (the reference)
simulate  write a fake serving log, optionally drifted (so you can demo without traffic)
drift     compare the serving log with the reference, store the result
concept   run Page-Hinkley on the error of scored trips (needs real durations)
gate      "should we retrain?" with reasons, stored in the history
history   show the latest drift run and the per-day status
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prodml.config import settings
from prodml.logging_conf import setup_logging
from prodml.monitoring import concept, drift, retrain, serving_log, store

COLUMNS = ["PULocationID", "DOLocationID", "trip_distance"]


def _score(df: pd.DataFrame) -> pd.Series:
    """Predictions from the real model if it loads, else a stand-in line."""
    try:
        from prodml.features import dataframe_to_dicts
        from prodml.predict import DurationPredictor

        return pd.Series(
            DurationPredictor().load().predict_batch(dataframe_to_dicts(df)),
            index=df.index,
        )
    except Exception as exc:  # noqa: BLE001 - monitoring must not need a model
        sys.stderr.write(f"note: no model ({exc!r}); using 2 + 3*distance\n")
        return 2 + 3 * df["trip_distance"]


def cmd_freeze(args: argparse.Namespace) -> int:
    from prodml.data import read_dataframe, split_data

    train, _ = split_data(read_dataframe(args.data), settings.train_ratio)
    sample = train.sample(min(args.rows, len(train)), random_state=0)[COLUMNS].copy()
    sample["prediction"] = _score(sample)
    sample.to_parquet(settings.reference_path, index=False)
    print(f"froze {len(sample)} rows -> {settings.reference_path}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Write a serving log. --shift 1.5 makes trips 50% longer and zones skewed."""
    from prodml.data import read_dataframe, split_data

    _, valid = split_data(read_dataframe(args.data), settings.train_ratio)
    rng = np.random.default_rng(args.seed)
    rows = valid.sample(
        args.rows, replace=len(valid) < args.rows, random_state=args.seed
    )
    rows = rows[COLUMNS].copy()
    if args.shift != 1.0:
        rows["trip_distance"] = (rows["trip_distance"] * args.shift).round(2)
        hot = rng.random(len(rows)) < 0.4  # 40% of pickups jump to one new area
        rows.loc[hot, "PULocationID"] = 265
    rows["prediction"] = _score(rows)
    path = args.out or settings.serving_log_path
    if not args.append:
        open(path, "w").close()
    serving_log.append(rows.to_dict("records"), path)
    print(f"wrote {len(rows)} rows (shift={args.shift}) -> {path}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    ref = pd.read_parquet(settings.reference_path)
    cur = serving_log.read(args.current, args.last)
    results = drift.compare(ref, cur)
    run_id = store.save_drift(
        settings.monitoring_db, [r.as_dict() for r in results], len(cur)
    )
    print(f"drift run {run_id}: {len(cur)} current rows vs {len(ref)} reference rows")
    print(f"{'feature':<15}{'PSI':>8}{'KS':>8}{'KS p':>10}  status")
    for r in results:
        ks = "" if r.ks_stat is None else f"{r.ks_stat:8.3f}{r.ks_p:10.1e}"
        print(f"{r.feature:<15}{r.psi:8.3f}{ks:>18}  {r.status}")
    worst = drift.overall(results)
    print(f"overall: {worst}")
    return 1 if args.fail_on_drift and worst == "drift" else 0


def cmd_concept(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.scored)
    if "lpep_pickup_datetime" in df:
        df = df.sort_values("lpep_pickup_datetime")  # a stream is ordered in time
    errors = (df["duration"] - df["prediction"]).abs().to_numpy()
    at = concept.first_alarm(errors, delta=args.delta, threshold=args.threshold)
    if at is None:
        print(f"no concept drift in {len(errors)} trips (MAE {errors.mean():.2f} min)")
        return 0
    print(
        f"CONCEPT DRIFT at trip {at} of {len(errors)}: MAE before "
        f"{errors[:at].mean():.2f} min, after {errors[at:].mean():.2f} min"
    )
    return 1 if args.fail_on_drift else 0


def cmd_gate(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    last_ts = store.last_retrain_ts(settings.monitoring_db)
    last = datetime.fromtimestamp(last_ts, timezone.utc) if last_ts else None
    log = serving_log.read()
    new_rows = int((log["ts"] > last_ts).sum()) if last_ts and len(log) else len(log)
    decision = retrain.decide(
        store.daily_status(settings.monitoring_db),
        new_rows=new_rows,
        last_retrain=last,
        now=now,
        sustained_days=args.sustained_days,
        min_new_rows=args.min_new_rows,
        cooldown_days=args.cooldown_days,
    )
    store.save_decision(settings.monitoring_db, decision.retrain, decision.reasons)
    print(decision.explain())
    if decision.retrain:
        print("-> start the training pipeline (e.g. trigger the Airflow DAG)")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    for row in store.latest_drift(settings.monitoring_db):
        print(f"{row['feature']:<15}{row['psi']:8.3f}  {row['status']}")
    print("per day (True = drift):", store.daily_status(settings.monitoring_db))
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)  # keep the tables readable
    p = argparse.ArgumentParser(prog="prodml.monitoring")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze")
    f.add_argument("--data", default=settings.data_path)
    f.add_argument("--rows", type=int, default=5000)
    f.set_defaults(func=cmd_freeze)

    s = sub.add_parser("simulate")
    s.add_argument("--data", default=settings.data_path)
    s.add_argument("--rows", type=int, default=2000)
    s.add_argument("--shift", type=float, default=1.0, help="1.0 = no drift")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out")
    s.add_argument("--append", action="store_true")
    s.set_defaults(func=cmd_simulate)

    d = sub.add_parser("drift")
    d.add_argument("--current", help="serving log path (default: from settings)")
    d.add_argument("--last", type=int, help="only the newest N rows")
    d.add_argument("--fail-on-drift", action="store_true")
    d.set_defaults(func=cmd_drift)

    c = sub.add_parser("concept")
    c.add_argument("--scored", required=True, help="parquet with prediction + duration")
    c.add_argument("--delta", type=float, default=0.5)
    c.add_argument("--threshold", type=float, default=50.0)
    c.add_argument("--fail-on-drift", action="store_true")
    c.set_defaults(func=cmd_concept)

    g = sub.add_parser("gate")
    g.add_argument("--sustained-days", type=int, default=2)
    g.add_argument("--min-new-rows", type=int, default=1000)
    g.add_argument("--cooldown-days", type=int, default=7)
    g.set_defaults(func=cmd_gate)

    h = sub.add_parser("history")
    h.set_defaults(func=cmd_history)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
