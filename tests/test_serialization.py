import numpy as np

from prodml.config import settings
from prodml.export import benchmark, parity_predictions
from prodml.features import dataframe_to_dicts
from prodml.predict import DurationPredictor


def _rows():
    return [
        {"PU_DO": f"{pu}_{do}", "trip_distance": float(d)}
        for pu, do, d in [(43, 236, 1.0), (74, 237, 5.5), (75, 238, 9.0)] * 170
    ]  # 510 rows


def test_pickle_and_onnx_agree(model_dir):
    """The ONNX model must give the same answers as the pickle model."""
    rows = _rows()
    pred_pkl, pred_onnx = parity_predictions(model_dir, rows)
    assert len(rows) >= 500
    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)


def test_benchmark_reports_both_formats(model_dir):
    result = benchmark(model_dir, _rows(), repeats=3)
    assert set(result) == {"pickle", "onnx"}
    assert result["onnx"]["mean_ms"] > 0 and result["pickle"]["p95_ms"] > 0


def test_service_can_serve_with_either_format(model_dir, monkeypatch, sample_features):
    answers = []
    for fmt in ("onnx", "pickle"):
        monkeypatch.setattr(settings, "serving_format", fmt)
        answers.append(DurationPredictor(model_dir).load().predict_one(sample_features))
    assert np.isclose(answers[0], answers[1], atol=1e-4)


def test_dataframe_to_dicts_round_trip(synthetic_csv):
    from prodml.data import read_dataframe

    dicts = dataframe_to_dicts(read_dataframe(synthetic_csv))
    assert set(dicts[0]) == {"PU_DO", "trip_distance"}
