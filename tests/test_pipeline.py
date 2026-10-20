"""The DVC stages as plain functions: prepare -> featurize -> train -> evaluate."""

import json

import pandas as pd
import pytest

from prodml import pipeline
from prodml.models import FEATURE_COLUMNS, load_bundle

PARAMS = {
    "data": {"train_ratio": 0.8},
    "train": {"model": "xgboost", "seed": 42},
    "xgboost": {"n_estimators": 10, "max_depth": 3},
}


@pytest.fixture
def staged(tmp_path, synthetic_csv):
    processed, features = tmp_path / "processed", tmp_path / "features"
    pipeline.prepare(synthetic_csv, 0.8, out=processed)
    pipeline.featurize(src=processed, out=features)
    return processed, features


def test_prepare_writes_a_chronological_split(staged):
    processed, _ = staged
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    assert len(train) > len(val) > 0
    assert train["lpep_pickup_datetime"].max() <= val["lpep_pickup_datetime"].min()


def test_featurize_gives_exactly_the_model_columns(staged):
    _, features = staged
    for split in ("train", "val"):
        df = pd.read_parquet(features / f"{split}.parquet")
        assert list(df.columns) == [*FEATURE_COLUMNS, "duration"]


def test_train_then_evaluate_reproduces_the_metric(staged, tracking_uri, tmp_path):
    _, features = staged
    cand, metrics = tmp_path / "candidate", tmp_path / "metrics"
    run_id = pipeline.train(
        PARAMS, "unused.parquet", src=features, candidate=cand, metrics_dir=metrics
    )
    assert json.loads((metrics / "train_run.json").read_text())["run_id"] == run_id
    assert load_bundle(cand).name == "xgboost"

    scores = pipeline.evaluate(src=features, candidate=cand, metrics_dir=metrics)
    assert set(scores) == {"mae", "rmse", "r2"}
    assert json.loads((metrics / "scores.json").read_text()) == scores

    # evaluating twice gives the SAME numbers: what makes `dvc repro` trustworthy
    assert (
        pipeline.evaluate(src=features, candidate=cand, metrics_dir=metrics) == scores
    )


def test_train_overwrites_a_previous_candidate(staged, tracking_uri, tmp_path):
    _, features = staged
    cand = tmp_path / "candidate"
    for _ in range(2):
        pipeline.train(
            PARAMS, "x", src=features, candidate=cand, metrics_dir=tmp_path / "m"
        )
    assert (cand / "meta.json").exists()


def test_validate_accepts_good_data(synthetic_csv):
    assert pipeline.validate(synthetic_csv, min_rows=100) == []


def test_validate_reports_every_problem(tmp_path):
    bad = tmp_path / "bad.csv"
    rows = [
        "lpep_pickup_datetime,lpep_dropoff_datetime,PULocationID,DOLocationID,trip_distance"
    ]
    rows += ["2019-08-01 00:00:00,2019-08-01 00:10:00,1,2,-3"] * 5
    rows += [",,,,"] * 5
    bad.write_text("\n".join(rows))
    problems = pipeline.validate(str(bad), min_rows=100)
    text = " | ".join(problems)
    assert (
        "only 10 rows" in text and "nulls" in text and "negative trip_distance" in text
    )


def test_validate_missing_file_hints_at_dvc_pull(tmp_path):
    problems = pipeline.validate(str(tmp_path / "nope.parquet"))
    assert len(problems) == 1 and "dvc pull" in problems[0]


def test_cli_validate_exit_codes(synthetic_csv, tmp_path, monkeypatch):
    from prodml.config import settings

    monkeypatch.setattr(settings, "data_path", str(tmp_path / "missing.parquet"))
    assert pipeline.main(["validate"]) == 1
    monkeypatch.setattr(settings, "data_path", synthetic_csv)
    monkeypatch.setattr(pipeline, "validate", lambda *a, **k: [])
    assert pipeline.main(["validate"]) == 0


def test_train_tracked_logs_the_family_named_in_params_yaml(
    tracking_uri, synthetic_csv, monkeypatch
):
    """`python -m prodml.train` (no flags) = one tracked MLflow run."""
    from prodml.config import settings
    from prodml.train import train_tracked

    monkeypatch.setattr(settings, "data_path", synthetic_csv)
    result = train_tracked()
    assert result["run_id"] and result["mae"] > 0


def test_train_main_local_flag_keeps_module_1_behaviour(
    synthetic_csv, tmp_path, monkeypatch
):
    import sys

    from prodml.config import settings
    from prodml.train import main

    monkeypatch.setattr(settings, "data_path", synthetic_csv)
    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(sys, "argv", ["prodml-train", "--local"])
    main()
    assert (tmp_path / "models" / settings.onnx_name).exists()
    assert (tmp_path / "models" / settings.metadata_name).exists()
