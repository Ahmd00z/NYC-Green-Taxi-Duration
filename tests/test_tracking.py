"""MLflow tracking: every run must carry its params, metrics, artifacts AND tags."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from prodml.experiments import load_splits, run_autolog_demo, sweep_xgboost
from prodml.models import make_family
from prodml.tracking import (
    data_version,
    freeze,
    log_family_run,
    pinned_requirements,
    standard_tags,
)

FAST = {"n_estimators": 10, "max_depth": 3}


@pytest.fixture(scope="module")
def frames(synthetic_csv):
    return load_splits(synthetic_csv, 0.8)


def test_data_version_prefers_the_dvc_hash(tmp_path):
    data = tmp_path / "trips.parquet"
    data.write_bytes(b"abc")
    (tmp_path / "trips.parquet.dvc").write_text(
        "outs:\n- md5: 0123456789abcdef0123456789abcdef\n  path: trips.parquet\n"
    )
    assert data_version(data) == "0123456789abcdef0123456789abcdef"


def test_data_version_without_dvc_hashes_the_file_itself(tmp_path):
    data = tmp_path / "trips.csv"
    data.write_bytes(b"abc")
    version = data_version(data)
    assert version.startswith("nodvc-")
    data.write_bytes(b"abcd")
    assert data_version(data) != version  # different bytes, different version


def test_data_version_unknown_when_nothing_exists(tmp_path):
    assert data_version(tmp_path / "missing.parquet") == "unknown"


def test_standard_tags_has_the_four_required_tags(synthetic_csv):
    tags = standard_tags("xgboost", synthetic_csv)
    assert set(tags) == {"git_commit", "data_version", "author", "framework"}
    assert tags["framework"] == "xgboost"


def test_pinned_requirements_pins_installed_packages_and_skips_missing():
    pins = pinned_requirements(["numpy", "definitely-not-installed-xyz"])
    assert len(pins) == 1 and pins[0].startswith("numpy==")


def test_freeze_lists_versions():
    assert "numpy==" in freeze().lower()


def test_a_run_logs_params_metrics_tags_and_artifacts(
    tracking_uri, frames, synthetic_csv
):
    train, val = frames
    res = log_family_run(
        make_family("xgboost", FAST),
        train,
        val,
        run_name="t-xgb",
        data_path=synthetic_csv,
    )
    run = MlflowClient().get_run(res.run_id)

    # params: hyperparameters + split seed/strategy + data version
    assert run.data.params["max_depth"] == "3"
    assert run.data.params["split_strategy"] == "chronological"
    assert run.data.params["data_version"] == run.data.tags["data_version"]
    # metrics
    for name in ("rmse", "mae", "r2", "train_seconds", "model_size_mb"):
        assert name in run.data.metrics
    # tags (the lineage)
    for name in ("git_commit", "data_version", "author", "framework", "family"):
        assert run.data.tags[name]
    # artifacts: both plots + requirements. (In MLflow 3 the model itself is a "logged
    # model" entity, not a run artifact: test_logged_model_loads_back_and_predicts
    # proves it is there by loading runs:/<id>/model.)
    paths = {a.path for a in MlflowClient().list_artifacts(res.run_id)}
    assert {"plots", "requirements.txt"} <= paths
    plots = {a.path for a in MlflowClient().list_artifacts(res.run_id, "plots")}
    assert plots == {"plots/residuals.png", "plots/feature_importance.png"}


def test_logged_model_loads_back_and_predicts(tracking_uri, frames, synthetic_csv):
    train, val = frames
    res = log_family_run(
        make_family("linear"), train, val, run_name="t-lin", data_path=synthetic_csv
    )
    model = mlflow.pyfunc.load_model(f"runs:/{res.run_id}/model")
    pred = model.predict(val[["PU_DO", "trip_distance"]].head(5))
    assert len(pred) == 5


def test_sweep_creates_parent_plus_nested_trials(tracking_uri, frames, synthetic_csv):
    train, val = frames
    best = sweep_xgboost(train, val, synthetic_csv, n_trials=3)
    runs = mlflow.search_runs(experiment_names=["test-exp"], output_format="list")
    nested = [r for r in runs if r.data.tags.get("mlflow.parentRunId")]
    parents = [r for r in runs if r.info.run_name == "xgboost-sweep"]
    assert len(nested) == 3 and len(parents) == 1
    assert best is not None and best.metrics["mae"] == min(
        r.data.metrics["mae"] for r in nested
    )


def test_autolog_demo_captures_params_and_metrics(tracking_uri, frames, synthetic_csv):
    train, val = frames
    run_id = run_autolog_demo(train, val, synthetic_csv)
    data = MlflowClient().get_run(run_id).data
    assert len(data.params) > 5  # autolog recorded the hyperparameters for free
    assert any("mae" in m for m in data.metrics)
