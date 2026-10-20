"""The three model families share one interface. Same tests run against all of them."""

import numpy as np
import pandas as pd
import pytest

from prodml.experiments import load_splits
from prodml.models import (
    FAMILIES,
    FEATURE_COLUMNS,
    bundle_size_mb,
    load_bundle,
    make_family,
    parse_zones,
)


def _families():
    return list(FAMILIES)


FAST = {
    "linear": {},
    "xgboost": {"n_estimators": 20, "max_depth": 4},
    "random_forest": {"n_estimators": 10, "max_depth": 6},
}


@pytest.fixture(scope="module")
def frames(synthetic_csv):
    return load_splits(synthetic_csv, 0.8)


@pytest.mark.parametrize("name", _families())
def test_fit_predict_shape_and_sanity(name, frames):
    train, val = frames
    model = make_family(name, FAST[name]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    pred = model.predict_frame(val[FEATURE_COLUMNS])
    assert pred.shape == (len(val),)
    assert np.isfinite(pred).all()
    # the fake data is "3 minutes per mile + 2": any real model beats a constant guess
    mae = np.mean(np.abs(pred - val["duration"].to_numpy()))
    baseline = np.mean(np.abs(val["duration"].mean() - val["duration"].to_numpy()))
    assert mae < baseline


@pytest.mark.parametrize("name", _families())
def test_save_load_roundtrip_gives_identical_predictions(name, frames, tmp_path):
    train, val = frames
    model = make_family(name, FAST[name]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    model.save(tmp_path / "bundle")
    loaded = load_bundle(tmp_path / "bundle")
    assert loaded.name == name
    X = val[FEATURE_COLUMNS]
    np.testing.assert_allclose(loaded.predict_frame(X), model.predict_frame(X))
    assert bundle_size_mb(tmp_path / "bundle") > 0


@pytest.mark.parametrize("name", _families())
def test_unseen_and_unknown_zones_do_not_crash(name, frames):
    train, _ = frames
    model = make_family(name, FAST[name]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    odd = pd.DataFrame(
        {"PU_DO": ["999_998", "unknown_236", "43_unknown"], "trip_distance": [1, 2, 3]}
    )
    assert np.isfinite(model.predict_frame(odd)).all()


@pytest.mark.parametrize("name", _families())
def test_feature_importance_has_names_and_values(name, frames):
    train, val = frames
    model = make_family(name, FAST[name]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    names, values = model.feature_importance(
        val[FEATURE_COLUMNS], val["duration"].to_numpy()
    )
    assert len(names) == len(values) > 0


def test_same_seed_same_model(frames):
    train, val = frames
    a = make_family("xgboost", FAST["xgboost"]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    b = make_family("xgboost", FAST["xgboost"]).fit(
        train[FEATURE_COLUMNS], train["duration"].to_numpy()
    )
    np.testing.assert_allclose(
        a.predict_frame(val[FEATURE_COLUMNS]), b.predict_frame(val[FEATURE_COLUMNS])
    )


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown model family"):
        make_family("catboost")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("43_236", (43, 236)),
        ("unknown_5", (-1, 5)),
        ("7_unknown", (7, -1)),
        ("garbage", (-1, -1)),
    ],
)
def test_parse_zones(raw, expected):
    pu, do = parse_zones(pd.Series([raw]))
    assert (int(pu[0]), int(do[0])) == expected
