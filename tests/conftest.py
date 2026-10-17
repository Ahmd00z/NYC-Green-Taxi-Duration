"""Shared fixtures. Tests train a TINY model on fake data in a temp folder, so
they never depend on the real dataset or on your local models/ folder."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.train import train


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> str:
    """600 fake trips: duration ~ 3 minutes per mile + noise."""
    rng = np.random.default_rng(0)
    n = 600
    pickup = pd.Timestamp("2019-08-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 30 * 24 * 60, n)), unit="m"
    )
    distance = rng.uniform(0.5, 10, n).round(2)
    duration_min = 3 * distance + rng.normal(0, 1, n) + 2
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pickup,
            "lpep_dropoff_datetime": pickup + pd.to_timedelta(duration_min, unit="m"),
            "PULocationID": rng.choice([43, 74, 75], n),
            "DOLocationID": rng.choice([236, 237, 238], n),
            "trip_distance": distance,
        }
    )
    path = tmp_path_factory.mktemp("data") / "trips.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def model_dir(synthetic_csv, tmp_path_factory) -> str:
    """Train once per test session into a temp folder."""
    out = str(tmp_path_factory.mktemp("models"))
    train(synthetic_csv, out)
    return out


@pytest.fixture
def sample_features() -> dict:
    return {"PU_DO": "43_236", "trip_distance": 2.5}


@pytest.fixture(scope="session")
def trained_model(model_dir) -> DurationPredictor:
    """A real predictor loaded from the tiny trained model."""
    return DurationPredictor(model_dir).load()


@pytest.fixture
def client(model_dir, monkeypatch):
    """FastAPI TestClient. The app's startup loads the tiny model from model_dir."""
    monkeypatch.setattr(settings, "models_dir", model_dir)
    with TestClient(app) as c:
        yield c
