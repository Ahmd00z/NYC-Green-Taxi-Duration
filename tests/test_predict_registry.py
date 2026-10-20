"""Module 2 headline: serve by STAGE, so swapping the model needs no code change."""

import pytest

from prodml import registry
from prodml.experiments import load_splits
from prodml.models import make_family
from prodml.predict import DurationPredictor
from prodml.tracking import log_family_run

TRIP = {"PU_DO": "43_236", "trip_distance": 2.5}


@pytest.fixture
def two_versions(tracking_uri, synthetic_csv):
    train, val = load_splits(synthetic_csv, 0.8)
    kw = {"train": train, "val": val, "data_path": synthetic_csv}
    lin = log_family_run(make_family("linear"), run_name="lin", **kw)
    xgb = log_family_run(
        make_family("xgboost", {"n_estimators": 5, "max_depth": 2}),
        run_name="xgb",
        **kw,
    )
    v1 = int(registry.register_run(lin.run_id).version)
    v2 = int(registry.register_run(xgb.run_id).version)
    return v1, v2


def test_loads_the_production_model_by_stage_uri(two_versions):
    v1, v2 = two_versions
    registry.promote_to_production(v1)
    predictor = DurationPredictor(source="registry").load()
    meta = predictor.get_metadata()
    assert meta["registry_version"] == v1
    assert meta["stage"] == "Production"
    assert meta["family"] == "linear"
    assert meta["git_commit"] and meta["data_version"]  # lineage travels with the model
    assert isinstance(predictor.predict_one(TRIP), float)
    assert len(predictor.predict_batch([TRIP, TRIP])) == 2


def test_promoting_another_version_changes_predictions_with_zero_code_change(
    two_versions,
):
    v1, v2 = two_versions
    registry.promote_to_production(v1)
    before = DurationPredictor(source="registry").load()
    registry.promote_to_production(v2)
    after = DurationPredictor(source="registry").load()  # same code, same settings
    assert before.get_metadata()["registry_version"] == v1
    assert after.get_metadata()["registry_version"] == v2
    assert before.predict_one(TRIP) != after.predict_one(TRIP)


def test_a_version_left_in_none_is_never_served(two_versions):
    v1, v2 = two_versions
    registry.promote_to_production(v1)  # v2 stays in None
    assert (
        DurationPredictor(source="registry").load().get_metadata()["registry_version"]
        == v1
    )


def test_no_production_model_fails_loudly(tracking_uri):
    with pytest.raises(Exception):  # the API turns this into a 503
        DurationPredictor(source="registry").load()


def test_models_dir_argument_still_means_local_files(model_dir):
    """Module 1 behaviour is kept: passing a directory loads files, not the registry."""
    predictor = DurationPredictor(model_dir).load()
    assert predictor.source == "local"
    assert predictor.get_metadata()["serving_format"] in {"onnx", "pickle"}
