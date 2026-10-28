"""DurationPredictor: the ONE class everything else talks to.

The API, (later) BentoML, and the optimized ONNX model all sit behind this
interface:  .load()  .predict_one()  .predict_batch()

Module 2: the model comes from the MLflow REGISTRY by stage, never by file path:

    models:/ride-duration-predictor/Production

so promoting a new model in the registry changes what is served with zero code changes
and zero rebuilds. The Module-1 behaviour (load models/model.pkl|onnx from disk)
is still available as an explicit offline fallback: MODEL_SOURCE=local, or
pass `models_dir=`.
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
from prodml.runtimes import make_ort_session
from prodml.utils import timed

# Rows scored per model call when a whole DataFrame is given (see predict_frame).
SCORE_SLICE_ROWS = 1000


class DurationPredictor:
    def __init__(
        self, models_dir: str | None = None, source: str | None = None
    ) -> None:
        # Passing a directory explicitly means "local files", as in Module 1.
        self.source = source or ("local" if models_dir else settings.model_source)
        self.models_dir = Path(models_dir or settings.models_dir)
        self.serving_format = settings.serving_format  # "onnx" or "pickle" (local only)
        self.metadata: dict[str, Any] = {}
        self._dv: Any = None
        self._model: Any = None
        self._session: ort.InferenceSession | None = None
        self._pyfunc: Any = None
        self._openvino: Any = None
        self._artifact_sha256 = ""

    # ---------------------------------------------------------------- loading
    def load(self) -> "DurationPredictor":
        """Load the model. Raises if it cannot (the API then answers 503)."""
        if self.source == "registry":
            self._load_registry()
        else:
            self._load_local()
        return self

    def _load_registry(self) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        name, stage = settings.registered_model_name, settings.model_stage
        self._pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")

        client = MlflowClient()
        in_stage = [
            v
            for v in client.search_model_versions(f"name='{name}'")
            if v.current_stage == stage
        ]
        mv = max(in_stage, key=lambda v: int(v.version))
        run = client.get_run(mv.run_id)
        self.metadata = {
            "model_version": f"{name}/v{mv.version}",
            "registry_version": int(mv.version),
            "stage": stage,
            "source": "registry",
            "run_id": mv.run_id,
            "framework": run.data.tags.get("framework"),
            "family": run.data.tags.get("family"),
            "git_commit": run.data.tags.get("git_commit"),
            "data_version": run.data.tags.get("data_version"),
            "val_mae": round(run.data.metrics.get("mae", float("nan")), 4),
            "val_rmse": round(run.data.metrics.get("rmse", float("nan")), 4),
        }
        self.serving_format = "registry"
        self._artifact_sha256 = hashlib.sha256(
            f"{mv.run_id}:{mv.version}".encode()
        ).hexdigest()

    def _load_local(self) -> None:
        pkl_path = self.models_dir / settings.pickle_name
        with open(pkl_path, "rb") as f:
            # Only unpickle files WE produced: pickle can run arbitrary code on load.
            self._dv, self._model = pickle.load(f)

        served = pkl_path
        if self.serving_format == "onnx":
            served = self.models_dir / settings.onnx_name
            # Module 3: ONNX Runtime with every graph optimization turned on.
            self._session = make_ort_session(served)
        elif self.serving_format == "openvino":
            # Module 3: OpenVINO runtime, loaded from the IR made by prodml.runtimes.
            from prodml.runtimes import OPENVINO_SUBDIR, OPENVINO_XML, OpenVINOSession

            served = self.models_dir / OPENVINO_SUBDIR / OPENVINO_XML
            self._openvino = OpenVINOSession(served)
            # The weights file (model.bin) is what really changes, so hash that one.
            served = served.with_suffix(".bin")
        self._artifact_sha256 = hashlib.sha256(served.read_bytes()).hexdigest()

        meta_path = self.models_dir / settings.metadata_name
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", __version__))

    # ------------------------------------------------------------- prediction
    def _score(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if self._pyfunc is not None:
            import pandas as pd

            return np.asarray(self._pyfunc.predict(pd.DataFrame(rows))).ravel()
        X = self._dv.transform(rows)
        if self._openvino is not None:
            return self._openvino.run(X.toarray().astype(np.float32))
        if self._session is not None:
            X_dense = X.toarray().astype(np.float32)
            name = self._session.get_inputs()[0].name
            return self._session.run(None, {name: X_dense})[0].ravel()
        return self._model.predict(X)

    def predict_frame(self, frame: Any) -> np.ndarray:
        """Score a DataFrame with columns PU_DO and trip_distance (used by batch jobs).

        With a registry model this skips the slow dict conversion and goes straight
        into the pyfunc model, which is what makes million-row batches fast.
        """
        if self._pyfunc is not None:
            return np.asarray(self._pyfunc.predict(frame)).ravel()
        records = frame[["PU_DO", "trip_distance"]].to_dict(orient="records")
        # The vectorizer makes one column per zone pair (~20 000), and the ONNX and
        # OpenVINO paths turn that into a DENSE array. 100 000 rows would need ~16 GB,
        # so we score in small slices to keep memory flat.
        step = SCORE_SLICE_ROWS
        parts = [
            self._score(records[i : i + step]) for i in range(0, len(records), step)
        ]
        return np.concatenate(parts) if parts else np.empty(0)

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
