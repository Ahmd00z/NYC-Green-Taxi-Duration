"""Mini Project 3, step 6: STREAMING inference with Redis Streams.

Picture a conveyor belt of trip events. Producers put events on the belt (the stream
`trips:in`), workers take events off it, score them, and put predictions on a second
belt (`predictions:out`). This file is the worker (the "consumer").

The four properties that make streaming harder than a web request:

  1. Consumer group   Many workers read the SAME stream, and Redis gives each event
                      to only ONE of them. More workers = more parallelism.
  2. XACK             A worker tells Redis "I finished this event". Until then Redis
                      keeps the event in a pending list for that worker.
  3. At-least-once    We XACK only AFTER the prediction is published. If a worker
                      dies in between, the event is delivered again. You never LOSE
                      an event, but you may see one twice, so every output carries
                      the `event_id` and downstream code can ignore duplicates.
  4. Dead letters     A "poison" event (for example distance = "abc") can never
                      succeed. We move it to `trips:dead` with the reason and
                      acknowledge it, so it cannot block the belt or be retried
                      forever.

We also measure END-TO-END latency (event produced -> prediction published), which is
what a user of the stream really feels, not just the model time.

Run it:
    python -m prodml.stream_consumer run                 # a worker (Ctrl+C to stop)
    python -m prodml.stream_consumer run --name worker-2 # a second worker, same group
    python -m prodml.stream_consumer stats               # latency percentiles so far
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from prodml.config import settings
from prodml.features import build_features
from prodml.logging_conf import setup_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger("prodml.stream")


# ---------------------------------------------------------------- redis plumbing
def connect(url: str | None = None):
    """Open a Redis connection. decode_responses=True gives us str instead of bytes."""
    import redis  # imported here so the rest of the package works without redis

    # socket_timeout must be LONGER than the blocking read (block_ms), or an idle
    # queue looks like a dead connection.
    return redis.Redis.from_url(
        url or settings.redis_url,
        decode_responses=True,
        socket_timeout=30,
        socket_connect_timeout=5,
    )


def ensure_group(r) -> None:
    """Create the consumer group (and the stream itself) if they do not exist yet."""
    import redis

    try:
        # id="0" means a brand new group starts from the beginning of the stream.
        r.xgroup_create(
            settings.stream_in, settings.stream_group, id="0", mkstream=True
        )
    except redis.exceptions.ResponseError as error:
        # BUSYGROUP just means "the group already exists", which is fine.
        if "BUSYGROUP" not in str(error):
            raise


def now_ms() -> float:
    return time.time() * 1000.0


# ---------------------------------------------------------------- event handling
def parse_event(fields: dict[str, str]) -> dict[str, Any]:
    """Check one raw event and turn it into typed values. Raises ValueError if bad.

    The rules are the same as the web API's request schema (api/schemas.py).
    """
    try:
        event = {
            "event_id": fields["event_id"],
            "PULocationID": int(fields["PULocationID"]),
            "DOLocationID": int(fields["DOLocationID"]),
            "trip_distance": float(fields["trip_distance"]),
            "produced_at_ms": float(fields.get("produced_at_ms", "nan")),
        }
    except KeyError as missing:
        raise ValueError(f"missing field {missing}") from None
    except (TypeError, ValueError) as error:
        raise ValueError(f"unreadable field: {error}") from None
    if event["PULocationID"] <= 0 or event["DOLocationID"] <= 0:
        raise ValueError("location ids must be > 0")
    if not 0 < event["trip_distance"] < 200:
        raise ValueError("trip_distance must be between 0 and 200 miles")
    return event


def send_to_dead_letter(
    r, message_id: str, fields: dict[str, str], reason: str
) -> None:
    """Park a poison event in the dead-letter stream, then acknowledge the original."""
    r.xadd(
        settings.stream_dead,
        {
            **fields,
            "error": reason,
            "original_id": message_id,
            "failed_at_ms": now_ms(),
        },
        maxlen=settings.stream_maxlen,
        approximate=True,
    )
    r.xack(settings.stream_in, settings.stream_group, message_id)
    logger.warning("dead-lettered", extra={"id": message_id, "reason": reason})


def delivery_count(r, message_id: str) -> int:
    """How many times Redis has handed this event to a worker (starts at 1)."""
    rows = r.xpending_range(
        settings.stream_in,
        settings.stream_group,
        min=message_id,
        max=message_id,
        count=1,
    )
    return int(rows[0]["times_delivered"]) if rows else 1


def publish_predictions(
    r,
    ids: list[str],
    events: list[dict[str, Any]],
    predictions: np.ndarray,
    version: str,
) -> list[float]:
    """Write predictions to the output stream, THEN acknowledge. Returns e2e
    latencies."""
    latencies: list[float] = []
    pipe = r.pipeline()
    for message_id, event, prediction in zip(ids, events, predictions):
        scored_at = now_ms()
        # End-to-end latency: from "event produced" to "prediction published".
        e2e = (
            scored_at - event["produced_at_ms"]
            if event["produced_at_ms"] == event["produced_at_ms"]
            else float("nan")
        )
        latencies.append(e2e)
        pipe.xadd(
            settings.stream_out,
            {
                "event_id": event["event_id"],
                "prediction": round(float(prediction), 4),
                "model_version": version,
                "produced_at_ms": event["produced_at_ms"],
                "scored_at_ms": scored_at,
                "e2e_ms": e2e,
            },
            maxlen=settings.stream_maxlen,
            approximate=True,
        )
    pipe.execute()  # the predictions are now safely in Redis ...
    r.xack(settings.stream_in, settings.stream_group, *ids)  # ... so we may ACK.
    return latencies


def process_messages(
    r, predictor: DurationPredictor, messages: list[tuple[str, dict[str, str]]]
) -> dict[str, int]:
    """Score a list of (id, fields) messages. Returns counts of ok / dead / retry."""
    counts = {"ok": 0, "dead": 0, "retry": 0}
    good_ids: list[str] = []
    good_events: list[dict[str, Any]] = []

    # Step 1: separate readable events from poison events.
    for message_id, fields in messages:
        try:
            good_events.append(parse_event(fields))
            good_ids.append(message_id)
        except ValueError as error:
            # A bad payload will fail every time, so there is no point retrying it.
            send_to_dead_letter(r, message_id, fields, str(error))
            counts["dead"] += 1
    if not good_events:
        return counts

    # Step 2: score all good events in ONE model call (micro-batch from the stream).
    version = str(predictor.get_metadata().get("model_version", "unknown"))
    try:
        frame = _frame(good_events)
        predictions = predictor.predict_frame(frame)
        publish_predictions(r, good_ids, good_events, predictions, version)
        counts["ok"] += len(good_ids)
        return counts
    except Exception:  # noqa: BLE001 - one bad event must not take the batch down
        logger.exception("batch scoring failed, retrying one by one")

    # Step 3: the batch failed, so score one at a time to find the culprit.
    for message_id, event in zip(good_ids, good_events):
        try:
            prediction = predictor.predict_frame(_frame([event]))
            publish_predictions(r, [message_id], [event], prediction, version)
            counts["ok"] += 1
        except Exception as error:  # noqa: BLE001
            if delivery_count(r, message_id) >= settings.stream_max_deliveries:
                send_to_dead_letter(
                    r, message_id, _as_fields(event), f"scoring: {error}"
                )
                counts["dead"] += 1
            else:
                counts["retry"] += 1  # stays pending; it will be reclaimed and retried
    return counts


def _frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Events -> DataFrame with the model columns, using the SAME build_features()."""
    rows = [
        build_features(e["PULocationID"], e["DOLocationID"], e["trip_distance"])
        for e in events
    ]
    return pd.DataFrame(rows)


def _as_fields(event: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in event.items()}


# ---------------------------------------------------------------- the worker loop
def run_consumer(
    r,
    predictor: DurationPredictor,
    name: str,
    batch_size: int = 100,
    block_ms: int = 1000,
    stop_when_idle_s: float | None = None,
) -> dict[str, int]:
    """Read, score, publish, ACK. Runs until Ctrl+C (or until idle for
    stop_when_idle_s).

    Order of work, every loop:
      a) my own unacked events (left over from a crash)      -> read with id "0"
      b) events other dead workers left unacked for too long -> XAUTOCLAIM
      c) brand new events                                    -> read with id ">"
    """
    from redis.exceptions import TimeoutError as RedisTimeout

    ensure_group(r)
    totals = {"ok": 0, "dead": 0, "retry": 0}
    last_work = time.time()
    while True:
        batches: list[list[tuple[str, dict[str, str]]]] = []

        # a) crash recovery: events delivered to ME before but never acknowledged.
        pending = r.xreadgroup(
            settings.stream_group, name, {settings.stream_in: "0"}, count=batch_size
        )
        batches.append(_flatten(pending))

        # b) steal events that another worker has held for too long without ACK.
        claimed = r.xautoclaim(
            settings.stream_in,
            settings.stream_group,
            name,
            min_idle_time=settings.stream_min_idle_ms,
            start_id="0-0",
            count=batch_size,
        )
        batches.append([(mid, f) for mid, f in claimed[1] if f])

        # c) new events. block_ms waits up to that long for something to arrive.
        try:
            fresh = r.xreadgroup(
                settings.stream_group,
                name,
                {settings.stream_in: ">"},
                count=batch_size,
                block=block_ms,
            )
        except RedisTimeout:  # an idle wait that ran too long is not a crash
            fresh = []
        batches.append(_flatten(fresh))

        did_work = False
        for batch in batches:
            if not batch:
                continue
            did_work = True
            for key, value in process_messages(r, predictor, batch).items():
                totals[key] += value
        if did_work:
            last_work = time.time()
        elif (
            stop_when_idle_s is not None and time.time() - last_work > stop_when_idle_s
        ):
            return totals


def _flatten(response: Any) -> list[tuple[str, dict[str, str]]]:
    """redis returns [[stream, [(id, fields), ...]]]; we only want the (id, fields)."""
    out: list[tuple[str, dict[str, str]]] = []
    for _stream, entries in response or []:
        out.extend((mid, fields) for mid, fields in entries if fields)
    return out


# ---------------------------------------------------------------- statistics
def latency_stats(r, count: int = 100_000) -> dict[str, float]:
    """p50/p95/p99 of end-to-end latency over the latest predictions in the output."""
    entries = r.xrevrange(settings.stream_out, count=count)
    values = [
        float(f["e2e_ms"]) for _, f in entries if f.get("e2e_ms") not in (None, "nan")
    ]
    if not values:
        return {"events": 0}
    return {
        "events": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "dead_letters": int(r.xlen(settings.stream_dead)),
    }


# ---------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="prodml.stream_consumer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="start a worker")
    run.add_argument("--name", default=f"worker-{socket.gethostname()}")
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--source", choices=["registry", "local"], default=None)
    run.add_argument("--stop-when-idle", type=float, default=None, help="seconds")
    sub.add_parser("stats", help="print end-to-end latency percentiles")
    args = parser.parse_args(argv)

    r = connect()
    if args.cmd == "stats":
        sys.stdout.write(json.dumps(latency_stats(r), indent=2) + "\n")
        return 0

    predictor = DurationPredictor(source=args.source).load()
    logger.info("consumer started", extra={"worker": args.name})
    try:
        totals = run_consumer(
            r,
            predictor,
            args.name,
            args.batch_size,
            stop_when_idle_s=args.stop_when_idle,
        )
        sys.stdout.write(json.dumps(totals) + "\n")
    except KeyboardInterrupt:
        logger.info("consumer stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
