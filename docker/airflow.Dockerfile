# Airflow image that can also import our `prodml` package (the DAGs call it).
FROM apache/airflow:2.10.5-python3.12
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER airflow
WORKDIR /opt/airflow/repo
COPY --chown=airflow pyproject.toml README.md ./
COPY --chown=airflow src ./src
RUN pip install --no-cache-dir -e ".[train]"
