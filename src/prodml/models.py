"""Three model families behind ONE interface: linear, xgboost, random_forest.

Every family:
  * trains on a frame with columns  PU_DO, trip_distance  (+ a `duration` target array),
  * predicts from the SAME two columns (exactly what the API sends at serving
    time, so training and serving can never drift apart),
  * saves itself into a plain directory ("bundle") and loads back from it.

No MLflow in this file: it is pure ML code, easy to unit-test. The MLflow wrapper
lives in pyfunc_model.py, the tracking helpers in tracking.py.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

FEATURE_COLUMNS = ["PU_DO", "trip_distance"]
FAMILIES = ("linear", "xgboost", "random_forest")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame[FEATURE_COLUMNS].to_dict(orient="records")


def _top(
    names: list[str], values: np.ndarray, k: int = 20
) -> tuple[list[str], np.ndarray]:
    order = np.argsort(np.abs(values))[::-1][:k]
    return [names[i] for i in order], np.asarray(values)[order]


class BaseFamily:
    """Shared plumbing. Subclasses implement fit / predict_frame / save / load."""

    name = "base"
    framework = "unknown"
    defaults: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**self.defaults, **(params or {})}

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseFamily:
        raise NotImplementedError

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(
        self, X: pd.DataFrame | None = None, y: np.ndarray | None = None
    ) -> tuple[list[str], np.ndarray]:
        raise NotImplementedError

    def save(self, directory: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: str | Path) -> BaseFamily:
        raise NotImplementedError

    def _write_meta(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(
            json.dumps({"family": self.name, "params": self.params}, default=str)
        )


# --------------------------------------------------------------------------- linear
class LinearFamily(BaseFamily):
    """The baseline: DictVectorizer + LinearRegression (same as Module 1)."""

    name = "linear"
    framework = "scikit-learn"

    def fit(self, X, y, X_val=None, y_val=None):
        self.dv = DictVectorizer()
        self.model = LinearRegression().fit(self.dv.fit_transform(_records(X)), y)
        return self

    def predict_frame(self, X):
        return self.model.predict(self.dv.transform(_records(X)))

    def feature_importance(self, X=None, y=None):
        return _top(list(self.dv.feature_names_), np.abs(self.model.coef_))

    def save(self, directory):
        directory = Path(directory)
        self._write_meta(directory)
        with open(directory / "bundle.pkl", "wb") as f:
            pickle.dump((self.dv, self.model), f)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        obj = cls(json.loads((directory / "meta.json").read_text())["params"])
        with open(directory / "bundle.pkl", "rb") as f:  # a file WE produced
            obj.dv, obj.model = pickle.load(f)
        return obj


# --------------------------------------------------------------------------- xgboost
class XGBoostFamily(BaseFamily):
    """Gradient-boosted trees with NATIVE categorical zones (fast: 3 columns).

    One-hot encoding the 'PU_DO' pair gives ~20k sparse columns and makes boosting
    very slow, so this family splits the pair back into two categorical columns
    (pickup zone, dropoff zone) plus trip_distance. Zones never seen in training
    become 'missing', which XGBoost handles natively.
    """

    name = "xgboost"
    framework = "xgboost"
    defaults = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": 42,
    }
    COLUMNS = ["pickup_zone", "dropoff_zone", "trip_distance"]

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        pu, do = parse_zones(X["PU_DO"])

        def categorical(ids: np.ndarray, vocab: list[int]) -> pd.Categorical:
            # a zone not seen in training becomes NaN ("missing"), which XGBoost handles
            s = pd.Series(ids)
            return pd.Categorical(s.where(s.isin(vocab)), categories=vocab)

        return pd.DataFrame(
            {
                "pickup_zone": categorical(pu, self.pu_vocab),
                "dropoff_zone": categorical(do, self.do_vocab),
                "trip_distance": X["trip_distance"].to_numpy(dtype=float),
            }
        )

    def fit(self, X, y, X_val=None, y_val=None):
        from xgboost import XGBRegressor

        pu, do = parse_zones(X["PU_DO"])
        self.pu_vocab = [int(v) for v in np.unique(pu[pu >= 0])]
        self.do_vocab = [int(v) for v in np.unique(do[do >= 0])]
        self.model = XGBRegressor(
            tree_method="hist",
            enable_categorical=True,
            eval_metric="mae",
            n_jobs=-1,
            **self.params,
        )
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(self._frame(X_val), y_val)]
        self.model.fit(self._frame(X), y, eval_set=eval_set, verbose=False)
        return self

    def predict_frame(self, X):
        return self.model.predict(self._frame(X))

    def feature_importance(self, X=None, y=None):
        gain = self.model.get_booster().get_score(importance_type="gain")
        values = np.array([gain.get(c, 0.0) for c in self.COLUMNS])
        return list(self.COLUMNS), values

    def save(self, directory):
        directory = Path(directory)
        self._write_meta(directory)
        self.model.save_model(directory / "model.json")
        (directory / "vocab.json").write_text(
            json.dumps({"pu": self.pu_vocab, "do": self.do_vocab})
        )

    @classmethod
    def load(cls, directory):
        from xgboost import XGBRegressor

        directory = Path(directory)
        obj = cls(json.loads((directory / "meta.json").read_text())["params"])
        obj.model = XGBRegressor(enable_categorical=True)
        obj.model.load_model(directory / "model.json")
        vocab = json.loads((directory / "vocab.json").read_text())
        obj.pu_vocab, obj.do_vocab = vocab["pu"], vocab["do"]
        return obj


# --------------------------------------------------------------------------- zones
def parse_zones(pu_do: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """'43_236' -> (43, 236). Anything unparseable (e.g. 'unknown_5') becomes -1."""
    parts = pu_do.astype(str).str.split("_", n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = np.nan
    pu = pd.to_numeric(parts[0], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    do = pd.to_numeric(parts[1], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    return pu, do


# ----------------------------------------------------------------- random forest
class RandomForestFamily(BaseFamily):
    """Bagged decision trees (scikit-learn): no extra dependency, hard to overfit.

    Zones are fed as plain integers (pickup zone, dropoff zone) plus trip_distance.
    A zone never seen in training (or an unparseable one) is encoded as -1, and the
    trees simply route it down the 'small id' side, so serving never crashes.
    """

    name = "random_forest"
    framework = "scikit-learn"
    defaults = {
        "n_estimators": 100,
        "max_depth": 14,
        "min_samples_leaf": 10,
        "max_features": 1.0,
        "random_state": 42,
    }
    COLUMNS = ["pickup_zone", "dropoff_zone", "trip_distance"]

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        pu, do = parse_zones(X["PU_DO"])
        return np.column_stack(
            [pu, do, X["trip_distance"].to_numpy(dtype=float)]
        ).astype(np.float64)

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_jobs=-1, **self.params)
        self.model.fit(self._matrix(X), np.asarray(y, dtype=float))
        return self

    def predict_frame(self, X):
        return self.model.predict(self._matrix(X))

    def feature_importance(self, X=None, y=None):
        return list(self.COLUMNS), np.asarray(self.model.feature_importances_)

    def save(self, directory):
        directory = Path(directory)
        self._write_meta(directory)
        with open(directory / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        obj = cls(json.loads((directory / "meta.json").read_text())["params"])
        with open(directory / "model.pkl", "rb") as f:  # a file WE produced
            obj.model = pickle.load(f)
        return obj


_REGISTRY: dict[str, type[BaseFamily]] = {
    "linear": LinearFamily,
    "xgboost": XGBoostFamily,
    "random_forest": RandomForestFamily,
}


def make_family(name: str, params: dict[str, Any] | None = None) -> BaseFamily:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model family {name!r}; choose from {FAMILIES}")
    return _REGISTRY[name](params)


def load_bundle(directory: str | Path) -> BaseFamily:
    """Load any family from a bundle directory (reads meta.json to find the class)."""
    meta = json.loads((Path(directory) / "meta.json").read_text())
    return _REGISTRY[meta["family"]].load(directory)


def bundle_size_mb(directory: str | Path) -> float:
    return (
        sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file()) / 1e6
    )
