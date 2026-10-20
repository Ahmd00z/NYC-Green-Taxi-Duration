"""The MLflow wrapper. Every family is logged as the SAME kind of pyfunc model, so the
serving layer (predict.py) never has to know which family Production is.

Input : DataFrame with columns PU_DO (string, e.g. "43_236") and trip_distance
Output: numpy array of predicted durations in minutes
"""

from __future__ import annotations

import mlflow.pyfunc
import numpy as np
import pandas as pd

from prodml.models import FEATURE_COLUMNS, load_bundle


class RideDurationPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:  # noqa: ANN001
        self._model = load_bundle(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None) -> np.ndarray:  # noqa: ANN001
        frame = pd.DataFrame(model_input)[FEATURE_COLUMNS]
        return np.asarray(self._model.predict_frame(frame), dtype=float)
