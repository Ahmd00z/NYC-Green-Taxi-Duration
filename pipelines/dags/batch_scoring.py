"""Mini Project 3, step 2: the NIGHTLY BATCH SCORING DAG.

    wait_for_input -> score -> notify

Every night it scores that day's trips file and writes one output partition:
    data/scoring/output/run_date=<ds>/part-00000.parquet

Idempotent on purpose: a re-run (or a backfill) of the same date overwrites the same
partition, never appends. That makes it safe to replay history:

    airflow dags backfill batch_scoring -s 2026-10-01 -e 2026-10-07
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor

from prodml.batch import batch_score
from prodml.orchestration import REPO_ROOT

# Absolute paths: Airflow does not run with the repo as its working directory.
INPUT_DIR = str(REPO_ROOT / "data/scoring/input")
OUTPUT_DIR = str(REPO_ROOT / "data/scoring/output")

default_args = {"owner": "ahmed", "retries": 2, "retry_delay": timedelta(minutes=5)}


def _score(ds: str) -> dict:
    """`ds` is the logical date (YYYY-MM-DD) Airflow gives every run."""
    report = batch_score(
        f"{INPUT_DIR}/trips_{ds}.parquet", OUTPUT_DIR, run_date=ds, chunk_size=100_000
    )
    return {
        k: report[k]
        for k in ("rows", "wall_clock_seconds", "rows_per_second", "peak_memory_mb")
    }


def _notify(ti) -> None:
    print("[notify] batch done:", json.dumps(ti.xcom_pull(task_ids="score")))


with DAG(
    dag_id="batch_scoring",
    description="Nightly: score one day of trips in chunks (flat memory)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,  # backfill is an explicit command, not a surprise
    max_active_runs=1,
    tags=["mlops", "module-3", "batch"],
) as dag:
    wait_for_input = FileSensor(
        task_id="wait_for_input",
        filepath=f"{INPUT_DIR}/trips_{{{{ ds }}}}.parquet",  # templated per run date
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=60 * 60 * 3,
        mode="reschedule",
    )
    score = PythonOperator(
        task_id="score",
        python_callable=_score,
        execution_timeout=timedelta(hours=1),
    )
    notify = PythonOperator(task_id="notify", python_callable=_notify)

    wait_for_input >> score >> notify
