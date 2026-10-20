# The image that runs the MLflow tracking server.
# Everything is stored in ONE folder (/mlflow): a SQLite database + the artifacts.
FROM python:3.11-slim
ARG PIP_DEFAULT_TIMEOUT=300
ENV PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT}
ENV PIP_RETRIES=10

ARG MLFLOW_VERSION=3.17.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}"

# Run as a normal user, not root. The folder is created here so that a Docker volume
# mounted on it inherits this owner.
RUN useradd --create-home --uid 1000 mlflow \
    && mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow
EXPOSE 5000