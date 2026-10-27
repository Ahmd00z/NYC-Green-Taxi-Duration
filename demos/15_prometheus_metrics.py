"""DEMO 15: Prometheus metrics for a model API: the 4 metric types and the label rule.

Run:  uv run python demos/15_prometheus_metrics.py
      (starts a tiny API on :8200, sends traffic, prints /metrics, stops)
Then, to SEE it in Prometheus, point a scrape job at localhost:8200
(see monitoring/prometheus.yml in the project; change the port).

Counter    only goes up (requests). Query it with rate(), never the raw value.
Gauge      goes up and down (in-flight requests, "which model is loaded").
Histogram  buckets: lets Prometheus compute p50/p95/p99 across many instances.
           An average would hide the slow tail.
Label rule a label with many values (user id, zone id, URL) = one time series each =
           an out-of-memory Prometheus. Use route TEMPLATES and small sets only.
"""

from __future__ import annotations

import random
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

app = FastAPI()
REQUESTS = Counter("demo_requests_total", "requests", ["route", "status"])
INFLIGHT = Gauge("demo_inflight_requests", "requests in progress")
LATENCY = Histogram(
    "demo_request_seconds", "latency", ["route"], buckets=(0.01, 0.05, 0.1, 0.25, 0.5)
)


@app.middleware("http")
async def measure(request: Request, call_next):
    INFLIGHT.inc()
    start = time.perf_counter()
    response = await call_next(request)
    INFLIGHT.dec()
    # the route TEMPLATE, never the raw URL (bounded number of label values)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    REQUESTS.labels(route, str(response.status_code)).inc()
    LATENCY.labels(route).observe(time.perf_counter() - start)
    return response


@app.get("/predict")
def predict(distance: float) -> dict:
    time.sleep(random.choice([0.01] * 8 + [0.2, 0.4]))  # 20% slow tail
    return {"minutes": 2 + 3 * distance}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")


def main() -> None:
    server = uvicorn.Server(uvicorn.Config(app, port=8200, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    for i in range(60):
        requests.get(f"http://localhost:8200/predict?distance={i % 7 + 1}")
    requests.get("http://localhost:8200/no-such-page-1")  # both land on ONE label
    requests.get("http://localhost:8200/no-such-page-2")
    text = requests.get("http://localhost:8200/metrics").text
    keep = ("demo_requests_total", "demo_inflight", "demo_request_seconds_bucket")
    print("\n".join(line for line in text.splitlines() if line.startswith(keep)))
    print(
        "\nPromQL you would run in Prometheus:\n"
        "  rate(demo_requests_total[1m])\n"
        "  histogram_quantile(0.95,\n"
        "    sum by (le) (rate(demo_request_seconds_bucket[1m])))"
    )
    assert 'route="unmatched"' in text and "no-such-page" not in text
    server.should_exit = True


if __name__ == "__main__":
    main()
