"""Where drift results and retrain decisions are remembered (SQLite, one file).

Why store them at all? A drift check that is only printed is forgotten. With history
you can ask "has it been drifting for 2 days in a row?", draw it on a dashboard, and
audit why a retrain did or did not happen.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_runs (
  run_id INTEGER, ts REAL, feature TEXT, kind TEXT,
  psi REAL, ks_stat REAL, ks_p REAL, status TEXT, n_current INTEGER
);
CREATE TABLE IF NOT EXISTS retrain_decisions (
  ts REAL, retrain INTEGER, reasons TEXT
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_drift(
    db_path: str, results: Iterable[dict], n_current: int, ts: float | None = None
) -> int:
    """Store one drift run (all features share a run_id). Returns the run_id."""
    ts = time.time() if ts is None else ts
    with closing(connect(db_path)) as con, con:
        run_id = (
            con.execute("SELECT MAX(run_id) FROM drift_runs").fetchone()[0] or 0
        ) + 1
        con.executemany(
            "INSERT INTO drift_runs VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    ts,
                    r["feature"],
                    r["kind"],
                    r["psi"],
                    r["ks_stat"],
                    r["ks_p"],
                    r["status"],
                    n_current,
                )
                for r in results
            ],
        )
    return run_id


def latest_drift(db_path: str) -> list[dict]:
    """Rows of the most recent run (empty list if none)."""
    if not Path(db_path).exists():
        return []
    with closing(connect(db_path)) as con:
        return [
            dict(r)
            for r in con.execute(
                "SELECT * FROM drift_runs WHERE run_id = "
                "(SELECT MAX(run_id) FROM drift_runs)"
            )
        ]


def daily_status(db_path: str, days: int = 7) -> list[tuple[str, bool]]:
    """[(YYYY-MM-DD, was_any_feature_drifted)] for the last `days` days with a run,
    oldest first. If a day has several runs, ANY drift that day counts."""
    if not Path(db_path).exists():
        return []
    with closing(connect(db_path)) as con:
        rows = con.execute(
            "SELECT date(ts,'unixepoch') AS day, MAX(status='drift') AS drifted "
            "FROM drift_runs GROUP BY day ORDER BY day DESC LIMIT ?",
            (days,),
        ).fetchall()
    return [(r["day"], bool(r["drifted"])) for r in reversed(rows)]


def save_decision(db_path: str, retrain: bool, reasons: list[str]) -> None:
    with closing(connect(db_path)) as con, con:
        con.execute(
            "INSERT INTO retrain_decisions VALUES (?,?,?)",
            (time.time(), int(retrain), json.dumps(reasons)),
        )


def last_retrain_ts(db_path: str) -> float | None:
    """When did the gate last say YES? (None = never)"""
    if not Path(db_path).exists():
        return None
    with closing(connect(db_path)) as con:
        row = con.execute(
            "SELECT MAX(ts) FROM retrain_decisions WHERE retrain = 1"
        ).fetchone()
    return row[0]
