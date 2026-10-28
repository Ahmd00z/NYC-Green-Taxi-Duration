"""Mini Project 3, step 3: the BentoML service (category 2: adaptive batching).

Why not just FastAPI? FastAPI scores ONE request at a time, so 50 users at once means
50 separate tiny model calls. BentoML collects the requests that arrive within a few
milliseconds into ONE bigger array and calls the model once. A model call costs about
the same for 1 row or 30 rows, so throughput goes up and the tail latency goes down.

    bentoml serve serving.service:TaxiDuration --port 3000

Request body: a JSON object with key "trips", one trip per row:
[PULocationID, DOLocationID, trip_distance]

    curl -X POST localhost:3000/predict -H 'content-type: application/json' \
         -d '{"trips": [[43, 236, 2.5]]}'

Even a single trip is sent as a list with one row. Many users sending one row each are
merged into one batch by BentoML; the clients never notice.
"""

from __future__ import annotations

import os

import bentoml
import numpy as np
import pandas as pd

from prodml.batch import features_frame
from prodml.predict import DurationPredictor

# Which model to serve: "registry" (models:/<name>/Production) or "local" (models/ dir).
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
# A label the canary setup uses to tell the two model containers apart in responses.
RELEASE_LABEL = os.getenv("RELEASE_LABEL", "stable")
# Batching knobs. BentoML REJECTS (503) a request when it expects to be answered later
# than max_latency_ms, so a value that is too small turns overload into errors.
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", "500"))
# Test knob: add fake latency (ms). Used ONLY to trigger the canary auto-rollback.
FAKE_DELAY_MS = float(os.getenv("FAKE_DELAY_MS", "0"))


@bentoml.service(
    name="taxi_duration",
    workers=1,
    traffic={"timeout": 30, "max_concurrency": 256},
)
class TaxiDuration:
    def __init__(self) -> None:
        # Runs once per worker, at start-up: load the model into memory.
        self.predictor = DurationPredictor(source=MODEL_SOURCE).load()

    @bentoml.api(
        batchable=True,  # BentoML may merge concurrent requests along axis 0
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,  # never merge more rows than this
        max_latency_ms=MAX_LATENCY_MS,  # latency budget; beyond it requests get a 503
    )
    def predict(self, trips: np.ndarray, ctx: bentoml.Context) -> np.ndarray:
        """trips: array of shape (n, 3) -> array of n predicted durations (minutes)."""
        frame = features_frame(
            pd.DataFrame(
                {
                    "PULocationID": trips[:, 0],
                    "DOLocationID": trips[:, 1],
                    "trip_distance": trips[:, 2],
                }
            )
        )
        if FAKE_DELAY_MS:
            import time

            time.sleep(FAKE_DELAY_MS / 1000)
        # Tell the caller (and nginx logs) which release answered.
        ctx.response.headers["X-Release"] = RELEASE_LABEL
        ctx.response.headers["X-Model-Version"] = self.predictor.version
        return np.asarray(self.predictor.predict_frame(frame), dtype=float)

    @bentoml.api
    def health(self) -> dict:
        """200 only when the model is loaded (same idea as FastAPI /health)."""
        return {"status": "healthy", "release": RELEASE_LABEL}
