"""Serialization: export the model to ONNX, and compare pickle vs ONNX."""

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.export")


def export_onnx(model: Any, num_features: int, path: str | Path) -> str:
    """Convert a fitted scikit-learn model to ONNX.

    `None` as the first dimension = dynamic batch size (1 row or 500 rows, same file).
    """
    onnx_model = convert_sklearn(
        model, initial_types=[("float_input", FloatTensorType([None, num_features]))]
    )
    Path(path).write_bytes(onnx_model.SerializeToString())
    logger.info("onnx exported", extra={"path": str(path)})
    return str(path)


def _load_both(models_dir: str | Path) -> tuple[Any, Any, ort.InferenceSession]:
    """Load vectorizer + sklearn model (pickle) and the ONNX session."""
    models_dir = Path(models_dir)
    with open(models_dir / settings.pickle_name, "rb") as f:
        dv, lr = pickle.load(f)  # safe: we produced this file ourselves
    session = ort.InferenceSession(
        str(models_dir / settings.onnx_name), providers=["CPUExecutionProvider"]
    )
    return dv, lr, session


def parity_predictions(
    models_dir: str | Path, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Run the same rows through the pickle model and the ONNX model."""
    dv, lr, session = _load_both(models_dir)
    X = dv.transform(rows).toarray().astype(np.float32)
    pred_pkl = lr.predict(X)
    pred_onnx = session.run(None, {session.get_inputs()[0].name: X})[0].ravel()
    return pred_pkl, pred_onnx


def benchmark(
    models_dir: str | Path, rows: list[dict[str, Any]], repeats: int = 30
) -> dict[str, dict[str, float]]:
    """Time pickle vs ONNX on the same rows. Returns mean and p95 latency in ms."""
    dv, lr, session = _load_both(models_dir)
    X = dv.transform(rows).toarray().astype(np.float32)
    name = session.get_inputs()[0].name

    def measure(fn: Any) -> dict[str, float]:
        fn()  # warm-up run, not counted
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
        return {
            "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    return {
        "pickle": measure(lambda: lr.predict(X)),
        "onnx": measure(lambda: session.run(None, {name: X})),
    }


def main() -> None:  # pragma: no cover
    """`python -m prodml.export` -> benchmark on the last 500 validation rows."""
    from prodml.data import read_dataframe, split_data
    from prodml.features import dataframe_to_dicts

    setup_logging()
    _, df_val = split_data(read_dataframe(settings.data_path), settings.train_ratio)
    rows = dataframe_to_dicts(df_val.tail(500))
    pred_pkl, pred_onnx = parity_predictions(settings.models_dir, rows)
    logger.info(
        "parity check",
        extra={
            "rows": len(rows),
            "allclose": bool(np.allclose(pred_pkl, pred_onnx, atol=1e-4)),
            "max_abs_diff": float(np.max(np.abs(pred_pkl - pred_onnx))),
        },
    )
    logger.info("benchmark", extra=benchmark(settings.models_dir, rows))


if __name__ == "__main__":
    main()
