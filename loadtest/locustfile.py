"""Mini Project 3, step 5: the load test (Locust).

One file, two targets, same traffic mix, so the comparison is fair:

    TARGET=fastapi  ->  POST /predict, POST /predict/batch, GET /health
    TARGET=bento    ->  POST /predict, POST /predict (several rows), POST /health

Traffic mix (Module 3): 80% single prediction, 15% small batch, 5% health/metadata.
Each simulated user waits 1-3 s between requests and sends DIFFERENT payloads, so the
test is not just hitting one cached value.

Headless run, ramping to 100 users (10 new users per second), 2 minutes:

    TARGET=fastapi locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 -t 2m --csv reports/load_fastapi
    TARGET=bento   locust -f loadtest/locustfile.py --host http://localhost:3000 \
        --headless -u 100 -r 10 -t 2m --csv reports/load_bento

The CSV files hold p50/p95/p99 and requests per second for the report table.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

TARGET = os.getenv("TARGET", "fastapi")
# Think time. Use WAIT_MIN=0 WAIT_MAX=0.05 for a stress run that saturates the server.
WAIT_MIN = float(os.getenv("WAIT_MIN", "1"))
WAIT_MAX = float(os.getenv("WAIT_MAX", "3"))


def _trip() -> dict:
    return {
        "PULocationID": random.randint(1, 265),
        "DOLocationID": random.randint(1, 265),
        "trip_distance": round(random.uniform(0.3, 25), 2),
    }


def _row() -> list:
    t = _trip()
    return [t["PULocationID"], t["DOLocationID"], t["trip_distance"]]


class TaxiUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)  # think time of a real client

    @task(80)
    def single(self) -> None:
        if TARGET == "bento":
            self.client.post("/predict", json={"trips": [_row()]}, name="/predict")
        else:
            self.client.post("/predict", json=_trip(), name="/predict")

    @task(15)
    def small_batch(self) -> None:
        n = random.randint(5, 20)
        if TARGET == "bento":
            body = {"trips": [_row() for _ in range(n)]}
            self.client.post("/predict", json=body, name="/predict (batch)")
        else:
            body = {"inputs": [_trip() for _ in range(n)]}
            self.client.post("/predict/batch", json=body, name="/predict/batch")

    @task(5)
    def health(self) -> None:
        if TARGET == "bento":
            self.client.post("/health", json={}, name="/health")
        else:
            self.client.get("/health", name="/health")
