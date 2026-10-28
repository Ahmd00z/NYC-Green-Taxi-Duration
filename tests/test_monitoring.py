"""Module 4 tests: drift maths, Page-Hinkley, store, retrain gate, /metrics, CLI."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from prodml.config import settings
from prodml.monitoring import concept, drift, retrain, serving_log, store
from prodml.monitoring.cli import main as cli_main

PAYLOAD = {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5}


def frame(n=3000, seed=0, scale=1.0, pu=(43, 74, 75)):
    rng = np.random.default_rng(seed)
    distance = rng.lognormal(0.8, 0.6, n) * scale
    return pd.DataFrame(
        {
            "PULocationID": rng.choice(pu, n),
            "DOLocationID": rng.choice([236, 237, 238], n),
            "trip_distance": distance,
            "prediction": 2 + 3 * distance,
        }
    )


# ------------------------------------------------------------------ PSI / KS
def test_psi_is_near_zero_for_the_same_distribution():
    a = np.random.default_rng(1).normal(size=5000)
    b = np.random.default_rng(2).normal(size=5000)
    assert drift.psi_numeric(a, b) < 0.05


def test_psi_grows_with_the_shift():
    a = np.random.default_rng(1).normal(size=5000)
    small = drift.psi_numeric(a, a + 0.2)
    big = drift.psi_numeric(a, a + 1.5)
    assert small < big and big > 0.25


def test_psi_constant_column_does_not_crash():
    assert drift.psi_numeric(np.ones(100), np.ones(100)) == 0.0


def test_psi_categorical_flags_a_new_zone():
    ref = pd.Series([43] * 500 + [74] * 500)
    cur = pd.Series([43] * 300 + [74] * 300 + [265] * 400)  # unseen zone
    assert drift.psi_categorical(ref, cur) > 0.25


def test_classify_thresholds():
    assert drift.classify(0.01) == "ok"
    assert drift.classify(settings.psi_warn) == "warn"
    assert drift.classify(settings.psi_drift) == "drift"


def test_compare_no_drift_vs_drift():
    ref = frame(seed=0)
    same = drift.compare(ref, frame(seed=1))
    assert drift.overall(same) == "ok"
    moved = drift.compare(ref, frame(seed=2, scale=1.6, pu=(43, 265)))
    status = {r.feature: r.status for r in moved}
    assert status["trip_distance"] == "drift" and status["PULocationID"] == "drift"
    assert drift.overall(moved) == "drift"


def test_compare_refuses_too_few_rows():
    with pytest.raises(ValueError, match="too few"):
        drift.compare(frame(), frame(n=10))


def test_compare_skips_columns_missing_on_one_side():
    out = drift.compare(frame(), frame(seed=1).drop(columns=["prediction"]))
    assert "prediction" not in {r.feature for r in out}


# ------------------------------------------------------------- Page-Hinkley
def test_page_hinkley_silent_on_stable_error():
    errors = np.abs(np.random.default_rng(0).normal(2, 0.5, 2000))
    assert concept.first_alarm(errors) is None


def test_page_hinkley_alarms_after_the_error_rises():
    rng = np.random.default_rng(0)
    errors = np.abs(np.concatenate([rng.normal(2, 0.5, 500), rng.normal(5, 0.5, 500)]))
    at = concept.first_alarm(errors)
    assert at is not None and 500 <= at < 700


def test_page_hinkley_false_alarm_rate_is_low_on_noisy_quiet_streams():
    """Defaults must survive realistic noise (this caught a bad first default)."""
    alarms = sum(
        concept.first_alarm(np.abs(np.random.default_rng(s).normal(0, 1.6, 1500)))
        is not None
        for s in range(60)
    )
    assert alarms <= 2


def test_page_hinkley_detects_a_shift_hidden_in_noise():
    delays = []
    for s in range(30):
        rng = np.random.default_rng(s)
        errors = np.concatenate(
            [
                np.abs(rng.normal(0, 1.6, 800)),
                np.abs(rng.normal(0, 1.6, 400)) * 1.25 + 3,
            ]
        )
        at = concept.first_alarm(errors)
        assert at is not None and at >= 800  # found it, and not before it happened
        delays.append(at - 800)
    assert np.median(delays) < 100


def test_page_hinkley_ignores_a_single_spike():
    errors = np.full(300, 2.0)
    errors[150] = 15.0
    assert concept.first_alarm(errors) is None


# ----------------------------------------------------------------- the store
def test_store_roundtrip_and_daily_status(tmp_path):
    db = str(tmp_path / "s.db")
    assert store.latest_drift(db) == []
    row = {
        "feature": "trip_distance",
        "kind": "numeric",
        "psi": 0.4,
        "ks_stat": 0.3,
        "ks_p": 0.0,
        "status": "drift",
    }
    day = 86_400.0
    store.save_drift(db, [row], 100, ts=1_700_000_000)
    store.save_drift(
        db, [{**row, "status": "ok", "psi": 0.0}], 100, ts=1_700_000_000 + day
    )
    assert store.latest_drift(db)[0]["status"] == "ok"
    assert [d for _, d in store.daily_status(db)] == [True, False]


def test_last_retrain_ts(tmp_path):
    db = str(tmp_path / "s.db")
    assert store.last_retrain_ts(db) is None
    store.save_decision(db, False, ["x"])
    assert store.last_retrain_ts(db) is None  # a HOLD is not a retrain
    store.save_decision(db, True, ["y"])
    assert store.last_retrain_ts(db) is not None


# --------------------------------------------------------------- retrain gate
NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
DRIFTED = [("d1", True), ("d2", True)]


def test_gate_says_yes_when_all_three_pass():
    d = retrain.decide(DRIFTED, 5000, None, NOW)
    assert d.retrain and len(d.reasons) == 3


@pytest.mark.parametrize(
    "daily, rows, last",
    [
        ([("d1", False), ("d2", True)], 5000, None),  # one noisy day
        (DRIFTED, 10, None),  # not enough data
        (DRIFTED, 5000, NOW - timedelta(days=1)),  # too soon
    ],
)
def test_gate_says_no_if_any_gate_fails(daily, rows, last):
    d = retrain.decide(daily, rows, last, NOW)
    assert not d.retrain
    assert d.reasons  # a refusal always explains itself


# ------------------------------------------------- API: /metrics + serving log
def test_metrics_endpoint_counts_predictions(client):
    client.post("/predict", json=PAYLOAD)
    client.post("/predict", json={**PAYLOAD, "trip_distance": -1})  # 422
    text = client.get("/metrics").text
    assert "prodml_predictions_total" in text
    assert 'prodml_http_requests_total{route="/predict",status="200"}' in text
    assert 'status="422"' in text
    assert "prodml_input_trip_distance_miles_bucket" in text
    assert "prodml_drift_last_run_timestamp_seconds" in text


def test_metric_labels_stay_bounded(client):
    for i in range(5):
        client.get(f"/does-not-exist-{i}")
    text = client.get("/metrics").text
    assert "does-not-exist" not in text  # unmatched URLs share ONE label value
    assert 'route="unmatched"' in text


def test_serving_log_gets_one_line_per_trip(client):
    client.post("/predict", json=PAYLOAD)
    client.post("/predict/batch", json={"inputs": [PAYLOAD, PAYLOAD]})
    log = serving_log.read()
    assert len(log) == 3 and set(log["PULocationID"]) == {43}
    assert log["prediction"].notna().all()


def test_serving_log_failure_never_breaks_a_prediction(client, monkeypatch):
    monkeypatch.setattr(settings, "serving_log_path", "/proc/cannot/write.jsonl")
    assert client.post("/predict", json=PAYLOAD).status_code == 200


def test_serving_log_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(settings, "serving_log_path", "")
    assert client.post("/predict", json=PAYLOAD).status_code == 200


def test_drift_numbers_reach_metrics(client, tmp_path):
    store.save_drift(
        settings.monitoring_db,
        [
            {
                "feature": "trip_distance",
                "kind": "numeric",
                "psi": 0.31,
                "ks_stat": 0.2,
                "ks_p": 0.0,
                "status": "drift",
            }
        ],
        500,
    )
    text = client.get("/metrics").text
    assert 'prodml_drift_psi{feature="trip_distance"} 0.31' in text


# ------------------------------------------------------------------------ CLI
def test_cli_drift_and_gate_flow(tmp_path, capsys):
    frame(seed=0).to_parquet(settings.reference_path)
    serving_log.append(frame(n=800, seed=3, scale=1.7).to_dict("records"))
    assert cli_main(["drift", "--fail-on-drift"]) == 1  # drifted -> CI would go red
    assert "overall: drift" in capsys.readouterr().out
    assert cli_main(["gate", "--min-new-rows", "100"]) == 0
    assert "HOLD" in capsys.readouterr().out  # only ONE day of drift so far


def test_cli_drift_ok_exits_zero(capsys):
    frame(seed=0).to_parquet(settings.reference_path)
    serving_log.append(frame(n=800, seed=4).to_dict("records"))
    assert cli_main(["drift", "--fail-on-drift"]) == 0


def test_cli_concept(tmp_path, capsys):
    rng = np.random.default_rng(0)
    pred = rng.uniform(5, 30, 1000)
    actual = pred + np.concatenate([rng.normal(0, 1, 500), rng.normal(6, 1, 500)])
    path = tmp_path / "scored.parquet"
    pd.DataFrame({"prediction": pred, "duration": actual}).to_parquet(path)
    assert cli_main(["concept", "--scored", str(path), "--fail-on-drift"]) == 1
    assert "CONCEPT DRIFT" in capsys.readouterr().out
