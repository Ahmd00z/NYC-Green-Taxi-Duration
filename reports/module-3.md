# Module 3 report: serving, batch, streaming, release

All numbers below were measured in a 2-core cloud sandbox with a SYNTHETIC dataset
(the real TLC download was blocked there). They show how to read the results and
which knob did what, **not** production performance. Re-run the same commands on
your machine and replace the tables with your own numbers.

## 1. Inference-pattern decision (CAT 0)

| Question | Answer for the taxi duration model |
|---|---|
| Who calls the model and when? | The app asks for ONE trip at the moment a ride is requested. |
| Latency target | p95 < 100 ms, user waits for the answer. |
| Freshness of inputs | Seconds. Distance and zones come with the request. |
| Volume | Hundreds of requests per second at peak, millions of rows per night for reports. |
| Pattern | **Online (REST)** for the app, **Batch** for nightly scoring of history, **Streaming** only for the event feed demo. |

Online is the main path; batch is the cheap path for data that is not needed instantly;
streaming is justified only when events already arrive on a bus.

## 2. The four problems of a plain FastAPI service (CAT 1)

| # | Problem | Where it shows in this repo |
|---|---|---|
| 1 | One request = one model call (batch size 1) | `api/main.py` `/predict` calls `predict_one` |
| 2 | Python GIL: one process runs one request's Python at a time | one uvicorn worker, CPU-bound feature code |
| 3 | Eager runtime: pickle + sklearn, no graph optimisation | see section 5, ONNX Runtime is 2.7x faster, OpenVINO 6.6x |
| 4 | No versioning/shadow: a new model replaces the old one at once | solved by section 7 (nginx canary) |

## 3. Load test (Locust, `loadtest/locustfile.py`)

Mix 80% single / 15% small batch / 5% health, 100 users, ramp 20 users/s, 40 s.

| Run | Requests | Failures | req/s | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| FastAPI, think time 1-3 s | 1898 | 0 | 48.6 | 5 | 48 | 70 |
| BentoML, think time 1-3 s | 1887 | 3 | 48.4 | 11 | 37 | 83 |
| FastAPI, saturated (no think time) | 15906 | 0 | 406.7 | 210 | 300 | 350 |
| BentoML, `max_latency_ms=50` (first try) | 18517 | **17052** | 474.2 | 86 | 450 | 2200 |
| BentoML, `max_latency_ms=500` (tuned) | 15525 | 144 | 397.7 | 200 | 350 | 550 |

Raw CSV files: `reports/load/`.

**Bottleneck diagnosis.** With think time the load is only ~48 req/s, so both servers are
idle most of the time and the table says nothing about capacity. In the saturated run the
CPU is the limit: the sandbox has 2 cores and Locust itself uses one, so the server has
roughly one core. The model call is cheap (milliseconds); the time goes into JSON parsing,
feature building and the single Python process (the GIL).

**One tuning change.** BentoML first returned 503 for 92% of requests. Its dispatcher
refuses a request when it expects the answer to arrive later than `max_latency_ms`; with
50 ms and a saturated CPU almost every request was refused. Raising it to 500 ms cut
failures from 17052 to 144. Lesson: `max_latency_ms` is a latency budget, not a wait
time. Adaptive batching did NOT beat FastAPI here, because the model is cheap and the CPU
is shared with the load generator. Expect a gain when a model call is expensive (a
neural network, a GPU) and many small requests arrive together.

## 4. Batch scoring (CAT 3, `prodml.batch`)

1.2 million rows, chunks of 100 000, one Parquet partition `run_date=...`:

| Runtime | Time | Rows/s | Peak RAM | Cost per million rows ($0.10/h machine) |
|---|---|---|---|---|
| pickle (sklearn) | 5.5 s | 217 000 | 312 MB | $0.000128 |
| ONNX Runtime | 37 s | 32 000 | 493 MB | $0.00087 |
| OpenVINO | 36 s | 33 000 | 596 MB | $0.00084 |

Memory stays flat because the file is read chunk by chunk. ONNX/OpenVINO are SLOWER here
because the DictVectorizer turns every row into a ~20 000 column dense vector before the
model sees it; that conversion, not the model, is the cost. Fixing it (a sparse or
embedding-based model) would matter more than any runtime.

## 5. Runtime comparison (CAT 4, `python -m prodml.runtimes benchmark`)

500 rows, same inputs, vectorising done outside the timer:

| Runtime | ms per call | Speed-up | Max difference vs pickle |
|---|---|---|---|
| pickle (eager) | 9.06 | 1.0x | 0 |
| ONNX Runtime (CPU) | 3.38 | 2.7x | 3.9e-6 |
| OpenVINO | 1.38 | 6.6x | 5.4e-6 |

Note: OpenVINO cannot read the `LinearRegressor` op that skl2onnx writes, so
`runtimes.py` writes a plain MatMul+Add ONNX file first (`model_core.onnx`).
Not done here: TensorRT/Triton (needs a GPU, Colab notebook, task 7a) and vLLM
(task 7c, belongs to the RAG final project).

## 6. Streaming (CAT 5, Redis Streams)

`stream_producer` sent 1000 events (every 25th is poison); `stream_consumer` with a
consumer group processed them: **960 scored, 40 dead-lettered, 0 left pending**
(`XACK` after publishing, so delivery is at-least-once). A crashed worker's unacknowledged
events are read again on restart (`tests/test_stream.py`). The reported p95 latency
(~9.9 s) is only because the consumer was started after the producer finished, so the
events waited in the queue. Start both together to see real end-to-end latency.

## 7. Release strategy (CAT 6)

| Strategy | Traffic | Risk to users | Rollback | Use when |
|---|---|---|---|---|
| Blue/Green | 100% switches at once | medium | instant (switch back) | big change, need a clean cut |
| **Canary (this repo)** | 10% -> 25% -> 50% -> 100% | low | seconds (set 0%) | normal model update |
| A/B test | split by user group | low | seconds | measure business impact |
| Shadow (`nginx.shadow.conf`) | copy of traffic, answers discarded | none | n/a | check a model before any user sees it |

Files: `deploy/nginx/nginx.conf` (90/10 `split_clients`), `deploy/canary_promote.sh`,
`deploy/rollback.sh`, `python -m prodml.canary watch` (auto-rollback after 2 bad
10-second intervals: p95 > 300 ms or errors > 5%). The split-rewrite and rollback logic
is unit-tested (`tests/test_canary.py`). NOT run here: nginx itself and the Docker
compose stacks (no Docker registry access in the sandbox). To practise the rollback
start `model-v2` with `FAKE_DELAY_MS=600`, send traffic, and time the rollback.

## 8. Airflow (CAT 7, `pipelines/dags/`)

`train_pipeline` (weekly: FileSensor, validate, prepare, featurize, train, evaluate,
branch, register or skip, notify; `retries=2`, 5 min delay, 45 min training timeout,
`catchup=False`) and `batch_scoring` (daily, idempotent per `run_date`). Verified in a
real Airflow 2.10.5 install: both DAGs import without errors; `batch_scoring` ran
sensor -> score -> notify with `airflow tasks test`. The training DAG was only
import-checked (a full run needs your MLflow server).
