"""Mini Project 3, step 6: the PRODUCER, a fake source of trip events.

It puts trip events on the Redis stream `trips:in` at a steady rate, so you can watch
the consumer (stream_consumer.py) score them. Every event carries `produced_at_ms`,
the moment it was created, so the consumer can measure end-to-end latency.

    python -m prodml.stream_producer --rate 200 --duration 30
    python -m prodml.stream_producer --rate 100 --duration 10 --poison-every 25
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid

import numpy as np

from prodml.config import settings
from prodml.logging_conf import setup_logging
from prodml.stream_consumer import connect, ensure_group, now_ms

logger = logging.getLogger("prodml.stream.producer")


def random_trip(rng: np.random.Generator) -> dict[str, str]:
    """One realistic-looking trip event (zones 1-265, skewed short distances)."""
    return {
        "event_id": str(uuid.uuid4()),
        "PULocationID": str(int(rng.integers(1, 266))),
        "DOLocationID": str(int(rng.integers(1, 266))),
        # Log-normal: many short trips, a few long ones, like real taxi data.
        "trip_distance": str(
            round(float(np.clip(rng.lognormal(0.8, 0.8), 0.1, 60)), 2)
        ),
    }


def produce(r, rate: int, duration: float, poison_every: int = 0, seed: int = 0) -> int:
    """Send `rate` events per second for `duration` seconds. Returns how many were sent.

    poison_every=N makes every N-th event unreadable (distance="abc") so you can see
    the dead-letter stream working.
    """
    ensure_group(r)
    rng = np.random.default_rng(seed)
    interval = 1.0 / rate
    sent = 0
    start = time.perf_counter()
    next_time = start
    while time.perf_counter() - start < duration:
        event = random_trip(rng)
        if poison_every and (sent + 1) % poison_every == 0:
            event["trip_distance"] = "abc"  # the poison
        event["produced_at_ms"] = str(now_ms())
        r.xadd(
            settings.stream_in, event, maxlen=settings.stream_maxlen, approximate=True
        )
        sent += 1
        # Sleep until the next slot (keeps the rate steady even if xadd is slow).
        next_time += interval
        delay = next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    return sent


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="prodml.stream_producer")
    parser.add_argument("--rate", type=int, default=100, help="events per second")
    parser.add_argument("--duration", type=float, default=10, help="seconds")
    parser.add_argument("--poison-every", type=int, default=0)
    args = parser.parse_args(argv)
    sent = produce(connect(), args.rate, args.duration, args.poison_every)
    sys.stdout.write(f"sent {sent} events at ~{args.rate}/s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
