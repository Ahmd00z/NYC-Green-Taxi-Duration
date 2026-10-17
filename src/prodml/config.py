"""All paths, hyperparameters and settings live here. Nothing is hardcoded elsewhere.

Every field can be overridden with an environment variable of the same name
(upper-case), e.g.  MODELS_DIR=/app/models  LOG_LEVEL=DEBUG
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- data & training ---
    data_path: str = "data/green_tripdata_2019-08.parquet"
    train_ratio: float = 0.8

    # --- model artifacts ---
    models_dir: str = "models"
    pickle_name: str = "model.pkl"
    onnx_name: str = "model.onnx"
    metadata_name: str = "metadata.json"

    # --- serving ---
    serving_format: Literal["onnx", "pickle"] = "onnx"
    log_level: str = "INFO"


settings = Settings()
