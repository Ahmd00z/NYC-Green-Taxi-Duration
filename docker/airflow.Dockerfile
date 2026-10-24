# Airflow image that can also import our `prodml` package (the DAGs call it).
#
# Airflow 2.10 needs old Flask/SQLAlchemy, so prodml is installed with Airflow's own
# "constraints" file: pip may not upgrade anything Airflow depends on. Because of that
# only the BASE dependencies fit (no full MLflow server, Optuna, DVC). This image runs
# the batch_scoring DAG for real; train_pipeline is parsed and drawn, but its training
# stages need the full training environment (run `dvc repro` on your own machine).
FROM apache/airflow:2.10.5-python3.12
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
WORKDIR /opt/airflow/repo
COPY --chown=airflow pyproject.toml README.md ./
COPY --chown=airflow src ./src
RUN pip install --no-cache-dir -e . \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
