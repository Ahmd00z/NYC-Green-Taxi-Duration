"""API tests. Note the `monkeypatch` tests: they swap the real model for a fake
one, so those tests don't depend on training at all."""

from prodml.api.main import app

PAYLOAD = {"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5}


class FakePredictor:
    version = "test"

    def predict_one(self, features):
        return 12.5

    def predict_batch(self, rows):
        return [10.0] * len(rows)


class BrokenPredictor(FakePredictor):
    def predict_one(self, features):
        raise RuntimeError("secret internal detail")


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_health_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(app.state, "predictor", None)
    assert client.get("/health").status_code == 503


def test_metadata(client):
    body = client.get("/metadata").json()
    assert body["feature_names"] == ["PU_DO", "trip_distance"]
    assert len(body["artifact_sha256"]) == 64


def test_predict_happy_path_with_fake_model(client, monkeypatch):
    monkeypatch.setattr(app.state, "predictor", FakePredictor())
    response = client.post("/predict", json=PAYLOAD)
    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] == 12.5
    assert set(body) == {"prediction", "model_version", "correlation_id", "latency_ms"}


def test_correlation_id_matches_header(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.headers["X-Request-ID"] == response.json()["correlation_id"]


def test_predict_with_real_model(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert 0 < response.json()["prediction"] < 60


def test_batch(client, monkeypatch):
    monkeypatch.setattr(app.state, "predictor", FakePredictor())
    response = client.post("/predict/batch", json={"inputs": [PAYLOAD, PAYLOAD]})
    assert response.status_code == 200
    assert response.json()["predictions"] == [10.0, 10.0]


def test_negative_distance_is_422_with_readable_message(client):
    response = client.post("/predict", json={**PAYLOAD, "trip_distance": -5})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["field"] == "trip_distance"
    assert "Traceback" not in response.text


def test_missing_field_is_422(client):
    assert client.post("/predict", json={"PULocationID": 43}).status_code == 422


def test_empty_batch_is_422(client):
    assert client.post("/predict/batch", json={"inputs": []}).status_code == 422


def test_unexpected_error_is_500_and_does_not_leak(client, monkeypatch):
    monkeypatch.setattr(app.state, "predictor", BrokenPredictor())
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert "X-Request-ID" in response.headers


def test_startup_without_model_does_not_crash(monkeypatch, tmp_path):
    """No trained model on disk -> server still starts, /health says 503."""
    from fastapi.testclient import TestClient

    from prodml.config import settings

    monkeypatch.setattr(settings, "models_dir", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503


def test_far_trip_logs_a_warning(client, monkeypatch, caplog):
    monkeypatch.setattr(app.state, "predictor", FakePredictor())
    with caplog.at_level("WARNING"):
        response = client.post("/predict", json={**PAYLOAD, "trip_distance": 150})
    assert response.status_code == 200
    assert any(r.levelname == "WARNING" for r in caplog.records)
