"""Tests for the Redis Streams consumer, using fakeredis (no server needed)."""

import fakeredis
import pytest

from prodml import stream_consumer as sc
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.stream_producer import produce, random_trip


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def predictor(model_dir):
    return DurationPredictor(model_dir).load()


def test_parse_event_validates():
    ok = sc.parse_event(
        {
            "event_id": "e1",
            "PULocationID": "43",
            "DOLocationID": "236",
            "trip_distance": "2.5",
        }
    )
    assert ok["PULocationID"] == 43
    for bad in (
        {
            "event_id": "e",
            "PULocationID": "43",
            "DOLocationID": "236",
            "trip_distance": "abc",
        },
        {
            "event_id": "e",
            "PULocationID": "0",
            "DOLocationID": "236",
            "trip_distance": "2",
        },
        {
            "event_id": "e",
            "PULocationID": "43",
            "DOLocationID": "236",
            "trip_distance": "500",
        },
        {"PULocationID": "43"},
    ):
        with pytest.raises(ValueError):
            sc.parse_event(bad)


def test_good_and_poison_events_end_to_end(r, predictor):
    produce(r, rate=500, duration=0.4, poison_every=10)
    sent = r.xlen(settings.stream_in)
    totals = sc.run_consumer(r, predictor, "w1", stop_when_idle_s=0.3, block_ms=50)
    assert totals["ok"] + totals["dead"] == sent
    assert totals["dead"] == sent // 10
    assert r.xlen(settings.stream_out) == totals["ok"]
    assert r.xlen(settings.stream_dead) == totals["dead"]
    # Everything was acknowledged: nothing is left pending.
    assert r.xpending(settings.stream_in, settings.stream_group)["pending"] == 0
    stats = sc.latency_stats(r)
    assert stats["p50_ms"] >= 0 and stats["dead_letters"] == totals["dead"]


def test_unacked_events_are_recovered_after_crash(r, predictor):
    sc.ensure_group(r)
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(5):
        event = random_trip(rng)
        event["produced_at_ms"] = str(sc.now_ms())
        r.xadd(settings.stream_in, event)
    # A worker reads the events and "crashes" before acknowledging them.
    r.xreadgroup(settings.stream_group, "w1", {settings.stream_in: ">"}, count=10)
    assert r.xpending(settings.stream_in, settings.stream_group)["pending"] == 5
    # The restarted worker with the same name finishes the job.
    totals = sc.run_consumer(r, predictor, "w1", stop_when_idle_s=0.3, block_ms=50)
    assert totals["ok"] == 5
    assert r.xpending(settings.stream_in, settings.stream_group)["pending"] == 0
