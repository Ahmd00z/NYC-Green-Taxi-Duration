import pytest


def test_predict_returns_sane_float(trained_model, sample_features):
    result = trained_model.predict_one(sample_features)
    assert isinstance(result, float)
    assert 0 < result < 60  # trips in training are 1-60 minutes


def test_predict_is_deterministic(trained_model, sample_features):
    assert trained_model.predict_one(sample_features) == trained_model.predict_one(
        sample_features
    )


def test_batch_matches_single(trained_model, sample_features):
    batch = trained_model.predict_batch([sample_features, sample_features])
    assert batch == pytest.approx([trained_model.predict_one(sample_features)] * 2)


def test_longer_trip_takes_longer(trained_model):
    short = trained_model.predict_one({"PU_DO": "43_236", "trip_distance": 1.0})
    long = trained_model.predict_one({"PU_DO": "43_236", "trip_distance": 9.0})
    assert long > short


def test_metadata_has_required_fields(trained_model):
    meta = trained_model.get_metadata()
    for key in ("model_version", "trained_at", "feature_names", "artifact_sha256"):
        assert key in meta


def test_predict_frame_scores_in_slices(model_dir, monkeypatch):
    """A big frame must be cut into small slices (the dense matrix would explode)."""
    import pandas as pd

    from prodml import predict as predict_module

    p = predict_module.DurationPredictor(model_dir).load()
    monkeypatch.setattr(predict_module, "SCORE_SLICE_ROWS", 4)
    sizes = []
    original = p._score
    monkeypatch.setattr(
        p, "_score", lambda rows: sizes.append(len(rows)) or original(rows)
    )
    frame = pd.DataFrame({"PU_DO": ["43_236"] * 10, "trip_distance": [2.5] * 10})
    out = p.predict_frame(frame)
    assert sizes == [4, 4, 2] and len(out) == 10
