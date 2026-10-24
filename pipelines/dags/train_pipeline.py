"""Mini Project 3, step 1: the WEEKLY TRAINING pipeline as an Airflow DAG.

    wait_for_data -> validate_data -> prepare -> featurize -> train -> evaluate
                                                                      |
                                                              decide (branch)
                                                           /                 \
                                               register_candidate     skip_registration
                                                           \                 /
                                                                  notify

Why Airflow when we already have DVC and a GitHub Actions workflow?
  * DVC knows WHAT depends on what (files). Airflow knows WHEN to run, what to do on
    failure (retry, time out) and how to replay history (backfill).
  * The DAG calls the very same commands as `dvc repro`, so there is one source of
    truth for the training code.

Production is still never promoted here. Registration puts the model in Staging and a
human moves it to Production, exactly like the CT workflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor

from prodml import orchestration
from prodml.config import settings

# Applied to EVERY task unless the task overrides it.
default_args = {
    "owner": "ahmed",
    "retries": 2,  # a flaky MLflow call gets two more chances ...
    "retry_delay": timedelta(minutes=5),  # ... five minutes apart
}


def _evaluate_and_publish(ti) -> dict:
    """Run evaluate, then push the SMALL result to XCom (run id + MAE)."""
    orchestration.run_stage("evaluate")
    return orchestration.read_candidate()  # returned value = XCom "return_value"


def _decide(ti) -> str:
    """BranchPythonOperator: return the task_id of the branch to follow."""
    candidate = ti.xcom_pull(task_ids="evaluate")
    branch = orchestration.choose_branch(candidate["mae"])
    print(f"candidate mae={candidate['mae']} -> {branch}")
    return branch


def _register(ti) -> dict:
    candidate = ti.xcom_pull(task_ids="evaluate")
    return orchestration.register_candidate(candidate["run_id"])


def _notify(ti) -> None:
    """Stand-in for Slack/email. A real version would POST to a webhook here."""
    candidate = ti.xcom_pull(task_ids="evaluate")
    registered = ti.xcom_pull(task_ids="register_candidate")
    if registered and registered.get("version"):
        outcome = f"registered as version {registered['version']} (Staging)"
    else:
        outcome = "not registered (did not beat Production)"
    print(f"[notify] run {candidate['run_id']} mae={candidate['mae']}: {outcome}")


with DAG(
    dag_id="train_pipeline",
    description="Weekly: validate data, train, evaluate, register to Staging if better",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,  # do not create one run for every week since start_date
    max_active_runs=1,  # two trainings at once would fight over models/candidate
    tags=["mlops", "module-3", "training"],
) as dag:
    # reschedule = free the worker slot between checks (poke mode would hold it).
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath=str(orchestration.REPO_ROOT / settings.data_path),  # absolute
        fs_conn_id="fs_default",
        poke_interval=60,
        timeout=60 * 60 * 6,  # give up after 6 hours
        mode="reschedule",
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=orchestration.run_stage,
        op_args=["validate"],
    )
    prepare = PythonOperator(
        task_id="prepare", python_callable=orchestration.run_stage, op_args=["prepare"]
    )
    featurize = PythonOperator(
        task_id="featurize",
        python_callable=orchestration.run_stage,
        op_args=["featurize"],
    )
    train = PythonOperator(
        task_id="train",
        python_callable=orchestration.run_stage,
        op_args=["train"],
        execution_timeout=timedelta(
            minutes=45
        ),  # a hung training must not block forever
    )
    evaluate = PythonOperator(task_id="evaluate", python_callable=_evaluate_and_publish)

    decide = BranchPythonOperator(task_id="decide", python_callable=_decide)
    register_candidate = PythonOperator(
        task_id="register_candidate", python_callable=_register
    )
    skip_registration = EmptyOperator(task_id="skip_registration")

    # Only ONE of the two branches runs, so the other is "skipped". Without this
    # trigger rule notify would be skipped too (default: ALL parents must succeed).
    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        trigger_rule="none_failed_min_one_success",
    )

    (wait_for_data >> validate_data >> prepare >> featurize >> train >> evaluate)
    evaluate >> decide >> [register_candidate, skip_registration] >> notify
