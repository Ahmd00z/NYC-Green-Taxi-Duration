"""Training entry point:  python -m prodml.train   (or just: prodml-train)"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml import __version__
from prodml.config import settings
from prodml.data import read_dataframe, split_data
from prodml.export import export_onnx
from prodml.features import extract_features
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.train")


def train(data_path: str, models_dir: str, train_ratio: float = 0.8) -> dict:
    """Load -> split -> features -> fit -> evaluate -> save (pickle + ONNX + metadata).

    Returns the validation metrics. Taking paths as ARGUMENTS (not globals)
    is what lets the tests train a tiny model in a temp folder.
    """
    df = read_dataframe(data_path)
    df_train, df_val = split_data(df, train_ratio)
    X_train, y_train, X_val, y_val, dv = extract_features(df_train, df_val)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / settings.pickle_name, "wb") as f:
        pickle.dump((dv, model), f)
    export_onnx(model, len(dv.feature_names_), out / settings.onnx_name)

    metadata = {
        "model_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn LinearRegression",
        "feature_names": ["PU_DO", "trip_distance"],
        "n_vectorizer_features": len(dv.feature_names_),
        "n_train_rows": len(df_train),
        "n_val_rows": len(df_val),
        "val_mae": round(mae, 4),
        "val_rmse": round(rmse, 4),
    }
    (out / settings.metadata_name).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    setup_logging()
    logger.info("training started", extra={"data_path": settings.data_path})
    metadata = train(settings.data_path, settings.models_dir, settings.train_ratio)
    logger.info("training finished", extra=metadata)


if __name__ == "__main__":
    main()
