"""Tests for the ONNX/OpenVINO runtimes and the Airflow helper functions."""

import json

import numpy as np
import pytest

from prodml import orchestration, runtimes
from prodml.config import settings


def test_core_onnx_and_openvino_match_pickle(model_dir, tmp_path):
    import pickle

    paths = runtimes.export_all(model_dir)
    with open(f"{model_dir}/{settings.pickle_name}", "rb") as f:
        dv, lr = pickle.load(f)
    rows = [
        {"PU_DO": "43_236", "trip_distance": 2.5},
        {"PU_DO": "74_237", "trip_distance": 7.0},
    ]
    x = dv.transform(rows).toarray().astype(np.float32)
    expected = lr.predict(x)
    ort = runtimes.make_ort_session(paths["core_onnx"])
    got_ort = ort.run(None, {ort.get_inputs()[0].name: x})[0].ravel()
    got_ov = runtimes.OpenVINOSession(paths["openvino_xml"]).run(x).ravel()
    assert np.allclose(expected, got_ort, rtol=5e-3)
    assert np.allclose(expected, got_ov, rtol=5e-3)


def test_choose_branch(monkeypatch):
    monkeypatch.setattr(orchestration.registry, "production_metric", lambda m: None)
    assert orchestration.choose_branch(5.0) == "register_candidate"
    monkeypatch.setattr(orchestration.registry, "production_metric", lambda m: 3.0)
    assert orchestration.choose_branch(2.0, margin=0.01) == "register_candidate"
    assert orchestration.choose_branch(3.5, margin=0.01) == "skip_registration"


def test_read_candidate(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/train_run.json").write_text(
        json.dumps({"run_id": "abc", "family": "xgboost"})
    )
    (tmp_path / "metrics/scores.json").write_text(json.dumps({"mae": 3.1, "r2": 0.8}))
    out = orchestration.read_candidate(tmp_path)
    assert out == {"run_id": "abc", "family": "xgboost", "mae": 3.1, "r2": 0.8}


def test_register_is_idempotent(monkeypatch):
    class V:
        run_id, version = "abc", "7"

    monkeypatch.setattr(orchestration.registry, "versions", lambda: [V()])
    assert orchestration.register_candidate("abc") == {"version": 7, "created": False}


def test_run_stage_fails_loudly(tmp_path):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        orchestration.run_stage("no_such_stage", repo_root=tmp_path)
