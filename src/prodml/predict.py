"""DurationPredictor: the ONE class everything else talks to.

The API, (later) BentoML, and the optimized ONNX model all sit behind this
interface:  .load()  .predict_one()  .predict_batch()
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from prodml import __version__
from prodml.config import settings
from prodml.utils import timed


class DurationPredictor:
    def __init__(self, models_dir: str | None = None) -> None:
        self.models_dir = Path(models_dir or settings.models_dir)
        self.serving_format = settings.serving_format  # "onnx" or "pickle"
        self.metadata: dict[str, Any] = {}
        self._dv: Any = None
        self._model: Any = None
        self._session: ort.InferenceSession | None = None
        self._artifact_sha256 = ""

    # ---------------------------------------------------------------- loading
    def load(self) -> "DurationPredictor":
        """Load artifacts from disk. Raises FileNotFoundError if not trained yet."""
        pkl_path = self.models_dir / settings.pickle_name
        with open(pkl_path, "rb") as f:
            # Only unpickle files WE produced: pickle can run arbitrary code on load.
            self._dv, self._model = pickle.load(f)

        served = pkl_path
        if self.serving_format == "onnx":
            served = self.models_dir / settings.onnx_name
            self._session = ort.InferenceSession(
                str(served), providers=["CPUExecutionProvider"]
            )
        self._artifact_sha256 = hashlib.sha256(served.read_bytes()).hexdigest()

        meta_path = self.models_dir / settings.metadata_name
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        return self

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", __version__))

    # ------------------------------------------------------------- prediction
    def _score(self, rows: list[dict[str, Any]]) -> np.ndarray:
        X = self._dv.transform(rows)
        if self._session is not None:
            X_dense = X.toarray().astype(np.float32)
            name = self._session.get_inputs()[0].name
            return self._session.run(None, {name: X_dense})[0].ravel()
        return self._model.predict(X)

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """features = {"PU_DO": "43_236", "trip_distance": 2.5} -> minutes."""
        return self.predict_batch([features])[0]

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[float]:
        """Many trips in one vectorized call (much faster than a Python loop)."""
        return [float(p) for p in self._score(rows)]

    # --------------------------------------------------------------- metadata
    def get_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "model_version": self.version,
            "serving_format": self.serving_format,
            "artifact_sha256": self._artifact_sha256,
        }
