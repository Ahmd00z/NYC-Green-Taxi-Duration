"""Prometheus metrics for the API (the "metric contract").

Rules this file follows:
  * names are  prodml_<what>_<unit>  and every counter ends with _total
  * base units only (seconds, miles, minutes), never milliseconds
  * labels have a SMALL, FIXED set of values. Never a request id, a zone id or a raw
    feature value as a label: each distinct value creates a new time series, and
    thousands of series is how a Prometheus server dies (cardinality).
    Per-request detail goes to the serving log instead.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

from prodml.config import settings

HTTP_REQUESTS = Counter(
    "prodml_http_requests_total",
    "HTTP requests by route and status code",
    ["route", "status"],
)
HTTP_LATENCY = Histogram(
    "prodml_http_request_duration_seconds",
    "Request duration by route",
    ["route"],
    # buckets chosen around a 100-500 ms SLO, not the library defaults
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
INFLIGHT = Gauge("prodml_inflight_requests", "Requests being handled right now")
PREDICTIONS = Counter(
    "prodml_predictions_total", "Predictions served", ["model_version"]
)
PREDICTED_MINUTES = Histogram(
    "prodml_prediction_minutes",
    "Distribution of predicted durations (output drift shows up here first)",
    ["model_version"],
    buckets=(2, 5, 8, 10, 15, 20, 30, 45, 60, 120),
)
INPUT_DISTANCE = Histogram(
    "prodml_input_trip_distance_miles",
    "Distribution of requested trip distances (input drift proxy)",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 50, 100),
)
MODEL_INFO = Gauge("prodml_model_info", "Always 1; which model is loaded", ["version"])


def set_model_info(version: str) -> None:
    MODEL_INFO.clear()  # only the CURRENT version is reported
    MODEL_INFO.labels(version=version).set(1)


def record_predictions(
    version: str, predictions: Iterable[float], distances: Iterable[float]
) -> None:
    for value in predictions:
        PREDICTIONS.labels(version).inc()
        PREDICTED_MINUTES.labels(version).observe(value)
    for distance in distances:
        INPUT_DISTANCE.observe(distance)


# ---------------------------------------------------------------------------
# Drift numbers computed by the OFFLINE drift job, shown on /metrics at scrape time.
# This is the bridge: batch job -> SQLite -> /metrics -> Prometheus -> alert rule.
class DriftCollector:
    """Reads the latest drift run from SQLite each time Prometheus scrapes."""

    def collect(self):  # noqa: D102 - prometheus_client calls this
        psi = GaugeMetricFamily(
            "prodml_drift_psi", "Latest PSI per monitored feature", labels=["feature"]
        )
        age = GaugeMetricFamily(
            "prodml_drift_last_run_timestamp_seconds",
            "Unix time of the latest drift run (0 = never ran)",
        )
        last_ts = 0.0
        try:
            from prodml.monitoring.store import latest_drift

            for row in latest_drift(settings.monitoring_db):
                psi.add_metric([row["feature"]], row["psi"])
                last_ts = max(last_ts, row["ts"])
        except (sqlite3.Error, OSError):  # no database yet is a normal state
            pass
        age.add_metric([], last_ts)
        yield psi
        yield age


_registered = False


def register_drift_collector(registry: CollectorRegistry = REGISTRY) -> None:
    """Idempotent: safe to call on every app start and in every test."""
    global _registered
    if not _registered:
        registry.register(DriftCollector())
        _registered = True


def render() -> bytes:
    return generate_latest(REGISTRY)
