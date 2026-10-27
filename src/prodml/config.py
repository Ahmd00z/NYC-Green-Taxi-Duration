"""All paths, hyperparameters and settings live here. Nothing is hardcoded elsewhere.

Every field can be overridden with an environment variable of the same name
(upper-case), e.g.  MODELS_DIR=/app/models  LOG_LEVEL=DEBUG  MLFLOW_TRACKING_URI=...
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- data & training ---
    data_path: str = "data/green_tripdata_2019-08.parquet"
    train_ratio: float = 0.8
    params_file: str = "params.yaml"  # hyperparameters tracked by DVC

    # --- model artifacts (local mode, Module 1) ---
    models_dir: str = "models"
    pickle_name: str = "model.pkl"
    onnx_name: str = "model.onnx"
    metadata_name: str = "metadata.json"

    # --- serving ---
    serving_format: Literal["onnx", "pickle", "openvino"] = (
        "onnx"  # only used by model_source=local
    )
    log_level: str = "INFO"

    # --- Module 2: tracking + registry ---
    # "registry": load models:/<registered_model_name>/<model_stage> from MLflow.
    # "local":    load models/model.pkl|onnx from disk (offline fallback, tests).
    model_source: Literal["registry", "local"] = "registry"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "ride-duration"
    registered_model_name: str = "ride-duration-predictor"
    model_stage: str = "Production"

    # --- Module 2: promotion gate / CI ---
    promote_metric: str = "mae"  # lower is better
    promotion_margin: float = 0.0  # candidate must beat Production by this fraction
    gate_max_regression: float = 0.05  # CI fails if MAE is >5% worse than baseline

    # --- Module 3: streaming (Redis Streams) ---
    redis_url: str = "redis://localhost:6379/0"
    stream_in: str = "trips:in"  # raw trip events
    stream_out: str = "predictions:out"  # predictions
    stream_dead: str = "trips:dead"  # dead-letter: poison / repeatedly failing events
    stream_group: str = "scorers"  # consumer group shared by all workers
    stream_max_deliveries: int = 3  # after this many tries an event is dead-lettered
    stream_min_idle_ms: int = 30_000  # reclaim events a dead worker left unacked
    stream_maxlen: int = 100_000  # approximate cap for every stream

    # --- Module 4: monitoring ---
    serving_log_path: str = "data/serving_log.jsonl"  # "" switches the log off
    reference_path: str = "monitoring/reference.parquet"  # frozen training sample
    monitoring_db: str = "monitoring/monitor.db"  # SQLite: drift + retrain history
    psi_warn: float = 0.10  # PSI above this: watch it
    psi_drift: float = 0.25  # PSI above this: it has drifted
    min_drift_rows: int = 200  # fewer rows than this: do not judge


settings = Settings()
