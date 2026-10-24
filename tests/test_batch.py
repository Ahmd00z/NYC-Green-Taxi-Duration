"""Tests for the chunked batch scorer."""

import json

import numpy as np
import pandas as pd
import pytest

from prodml import batch
from prodml.features import build_features
from prodml.predict import DurationPredictor


@pytest.fixture
def trips(tmp_path) -> str:
    rng = np.random.default_rng(1)
    n = 250
    df = pd.DataFrame(
        {
            "PULocationID": rng.choice([43, 74, 75], n),
            "DOLocationID": rng.choice([236, 237, 238], n),
            "trip_distance": rng.uniform(0.5, 10, n).round(2),
        }
    )
    path = tmp_path / "trips.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def test_features_frame_matches_build_features():
    df = pd.DataFrame(
        {
            "PULocationID": [43, None],
            "DOLocationID": [236, 5],
            "trip_distance": [2.5, -1],
        }
    )
    out = batch.features_frame(df)
    for i, row in df.iterrows():
        pu = None if pd.isna(row.PULocationID) else int(row.PULocationID)
        expected = build_features(pu, int(row.DOLocationID), max(row.trip_distance, 0))
        assert out.iloc[i]["PU_DO"] == expected["PU_DO"]
        assert out.iloc[i]["trip_distance"] == pytest.approx(expected["trip_distance"])


def test_batch_score_chunks_and_is_idempotent(trips, model_dir, tmp_path):
    predictor = DurationPredictor(model_dir).load()
    out = tmp_path / "out"
    r1 = batch.batch_score(
        trips, out, "2026-10-01", chunk_size=100, predictor=predictor
    )
    r2 = batch.batch_score(
        trips, out, "2026-10-01", chunk_size=100, predictor=predictor
    )
    files = list((out / "run_date=2026-10-01").glob("part-*.parquet"))
    assert len(files) == 1  # replaced, never appended
    scored = pd.read_parquet(files[0])
    assert len(scored) == 250 == r1["rows"] == r2["rows"]
    assert "predicted_duration_min" in scored.columns
    assert scored["predicted_duration_min"].notna().all()
    report = json.loads((out / "run_date=2026-10-01" / "_report.json").read_text())
    assert report["rows"] == 250


def test_cost_per_million():
    # 100k rows/s on a $0.10/h machine = 10 s per million = $0.000278 per million.
    assert batch.cost_per_million(100_000, 0.10) == pytest.approx(0.10 * 10 / 3600)


def test_make_sample(trips, tmp_path):
    base = pd.read_parquet(trips).assign(
        lpep_pickup_datetime=pd.Timestamp("2019-08-01")
    )
    src = tmp_path / "src.parquet"
    base.to_parquet(src, index=False)
    out = batch.make_sample(tmp_path / "big.parquet", 1000, source_path=src)
    assert len(pd.read_parquet(out)) == 1000
