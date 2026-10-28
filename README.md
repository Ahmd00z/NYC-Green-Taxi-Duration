# NYC Green Taxi Trip-Duration — MLOps

[![CI](https://github.com/Ahmd00z/NYC-Green-Taxi-Duration/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmd00z/NYC-Green-Taxi-Duration/actions/workflows/ci.yml)

## Description

An end-to-end **MLOps** project that predicts how long a NYC green taxi trip will take
(in minutes) from the pickup zone, the drop-off zone and the trip distance. It starts
from a notebook and grows, module by module, into a production-style system: a tested
API, reproducible experiments, a model registry, versioned data, automated retraining,
and several ways to serve the same model (online, batch, streaming) with a safe release
process.

The project is built in layers. Each layer solves one real problem you meet when you move
from a prototype to production, and each can be understood on its own.

### How it works

1. **Train**: load trips, clean them, split by time (first 80% train, last 20% validate),
   build features, train a model.
2. **Track**: every run is logged to **MLflow** (parameters, metrics, plots, git commit,
   data version).
3. **Register**: the best run is registered and moves through stages
   `None -> Staging -> Production`. Only a human moves a model to Production.
4. **Serve**: the API loads `models:/ride-duration-predictor/Production`. Changing the
   served model means changing a stage, not code.
5. **Operate**: schedule it (Airflow), score millions of rows (batch), consume an event
   stream (Redis), load test it (Locust), and release new models gradually (nginx canary).

### Modules

| Module | Theme | What was added |
|---|---|---|
| 1 | Packaging | Tested FastAPI service, ONNX export, JSON logs, Docker image, CI |
| 2 | Tracking and automation | MLflow tracking, 3 model families, Optuna sweep, model registry, DVC pipeline, CI quality gate, continuous training |
| 3 | Serving and release | Airflow DAGs, ONNX Runtime and OpenVINO, BentoML, batch scoring, Redis Streams, Locust, nginx canary and rollback |

### Tech stack

| Layer | Technology |
|---|---|
| Language and packaging | Python 3.12, `uv`, `pyproject.toml` + `uv.lock` |
| ML | scikit-learn (linear, random forest), XGBoost, Optuna |
| Tracking and registry | MLflow 3.17 (SQLite backend, local artifacts) |
| Data and pipeline versioning | DVC (local remote folder) |
| API | FastAPI, Pydantic, Uvicorn |
| Fast inference | ONNX, ONNX Runtime, OpenVINO |
| Model server | BentoML (adaptive batching) |
| Orchestration | Apache Airflow |
| Streaming | Redis Streams (consumer groups) |
| Load testing | Locust |
| Release | nginx (`split_clients`, `mirror`), shell scripts, Python watcher |
| Quality | pytest (70% coverage gate), Ruff, Black, pre-commit |
| CI/CD | GitHub Actions (CI and continuous training) |
| Containers | Docker, Docker Compose |

## Architecture

```text
                 raw data ──dvc──▶ prepare ─▶ featurize ─▶ train ─▶ evaluate ─▶ metrics/scores.json
                                                              │
                                                              ▼
                                        MLflow run (params, metrics, plots, tags)
                                                              │ promote_if_better()
                                                              ▼
                      Model Registry: None ─▶ Staging ─▶ (human approval) ─▶ Production
                                                                                  │
   ┌──────────────────────────────────────────────────────────────────────────────┤
   ▼                      ▼                        ▼                             ▼
 FastAPI (online)    BentoML (online,        Batch job (Parquet,          Redis Streams consumer
 /predict            adaptive batching)      chunked, idempotent)         (events in, scores out)
   ▲                      ▲
   └────── nginx canary: 90% stable / 10% new, auto-rollback on bad latency or errors
```

Airflow runs the weekly training DAG and the nightly batch DAG. GitHub Actions runs the CI
gate on every pull request and the continuous-training workflow on a schedule.

## Requirements

- **Python 3.12** (the repo pins it in `.python-version`). Do **not** use 3.14: older
  `werkzeug` releases crash on it.
- `uv` (https://docs.astral.sh/uv/)
- Git
- Docker with Docker Compose (MLflow stack, Airflow, Redis, canary)
- Free host ports: 5000 (MLflow), 8000 (API), 3100 (BentoML in Compose), 8081 (Airflow),
  6380 (Redis in Compose), 8080 (canary).

## Installation

```bash
git clone https://github.com/Ahmd00z/NYC-Green-Taxi-Duration.git
cd NYC-Green-Taxi-Duration

# Pick only the extras you need. `uv sync` REMOVES extras you do not list,
# so always write the full set you want.
uv sync --extra dev --extra train                               # core work
uv sync --extra dev --extra train --extra serving --extra load  # + BentoML + Locust
```

| Extra | Installs | Needed for |
|---|---|---|
| `train` | MLflow, Optuna, matplotlib, DVC | experiments, registry, DVC pipeline |
| `dev` | pytest, ruff, black, redis, fakeredis, onnx, openvino | tests and all code quality checks |
| `serving` | BentoML | `serving/service.py` |
| `runtimes` | onnx, openvino, skl2onnx | ONNX Runtime and OpenVINO conversion |
| `streaming` | redis | Redis Streams producer and consumer |
| `load` | locust | load testing |

Add the dataset: place `green_tripdata_2019-08.parquet` under `data/` (or run
`dvc pull data/green_tripdata_2019-08.parquet.dvc` if you have the DVC remote). Raw data is
not stored in Git.

## Quickstart

```bash
# 1. Start MLflow (no passwords, no .env needed)
docker compose -f docker/docker-compose.yml up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000       # UI: http://localhost:5000

# 2. Train and track 15 runs (3 families + 10-trial sweep + autolog demo)
uv run python -m prodml.experiments

# 3. Promote the best run (get the run id from the MLflow UI)
uv run python -m prodml.registry promote-if-better --run-id <RUN_ID>   # -> Staging
uv run python -m prodml.registry show
uv run python -m prodml.registry promote-production --version 1        # human step

# 4. Serve it
MODEL_SOURCE=registry uv run uvicorn prodml.api.main:app --port 8000
curl http://localhost:8000/health
```

Interactive docs: http://localhost:8000/docs

## Access services

| Service | URL | Started by |
|---|---|---|
| MLflow UI | http://localhost:5000 | `docker compose -f docker/docker-compose.yml up -d mlflow` |
| FastAPI + Swagger | http://localhost:8000/docs | `uvicorn prodml.api.main:app --port 8000` |
| BentoML | http://localhost:3100 (Compose) or the `--port` you choose | `bentoml serve` or `docker-compose.serving.yml` |
| Airflow | http://localhost:8081 | `docker/docker-compose.serving.yml` |
| Redis | `localhost:6380` (Compose) | `docker/docker-compose.serving.yml` |
| Canary entry point (nginx) | http://localhost:8080 | `docker/docker-compose.canary.yml` |

Ports 3000 and 6379 are often used by Grafana and another Redis on a developer machine.
This repo avoids them on purpose (Compose uses 3100 and 6380).

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | 200 only if the model is really loaded |
| GET | `/metadata` | Model version, training info, feature names, model hash |
| POST | `/predict` | One trip |
| POST | `/predict/batch` | Many trips; **all or nothing** (one invalid row rejects the request with 422) |
| POST | `/predict/batch/partial` | Many trips; invalid rows return an error, valid rows return a prediction |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5}'
```

```json
{"prediction": 8.07, "model_version": "ride-duration-predictor/v1",
 "correlation_id": "…", "latency_ms": 1.37}
```

(The numbers are an example of the format. Real values depend on the model.)

Every response carries an `X-Request-ID` header, the same id that appears in the JSON logs.
Validation: zone ids must be > 0 and `0 < trip_distance < 200`.

## Module 1: packaging

- `src/prodml/data.py` loads and cleans trips and splits them in time order.
- `features.py` builds the single feature set used by training **and** serving, so they
  never disagree.
- `train.py` trains the baseline and writes `models/model.pkl`, `models/model.onnx` and
  `models/metadata.json`.
- `export.py` exports to ONNX, checks Pickle/ONNX parity and runs a benchmark.
- The API is in `src/prodml/api/` (`main.py` endpoints and middleware, `schemas.py`
  request/response contract).

```bash
uv run python -m prodml.train --local     # offline training, no MLflow needed
uv run python -m prodml.export
MODEL_SOURCE=local uv run uvicorn prodml.api.main:app --port 8000
```

## Module 2: tracking, registry, data versioning, automation

| Piece | Files | What it does |
|---|---|---|
| Model families | `models.py` | Linear, XGBoost and Random Forest behind one interface |
| Tracking | `tracking.py`, `experiments.py` | Logs params, metrics, plots, `requirements.txt` and tags (`git_commit`, `data_version`, `author`, `framework`) |
| Registry | `registry.py` | Stages, `promote_if_better`, `promote-production`, CLI |
| Serving by stage | `predict.py` | `MODEL_SOURCE=registry` loads `models:/<name>/<stage>` |
| DVC pipeline | `pipelines/dvc.yaml`, `params.yaml`, `pipeline.py` | `prepare -> featurize -> train -> evaluate`, re-runs only what changed |
| Quality gate | `gate.py` | CI fails if MAE is worse than the baseline by more than `GATE_MAX_REGRESSION` |
| CI | `.github/workflows/ci.yml` | lint, tests, model gate, Docker build |
| Continuous training | `.github/workflows/continous-training.yml` | retrain, compare with Production, promote to Staging |

Change the served model with no code change and no rebuild:

```bash
uv run python -m prodml.registry promote-production --version 2
# then restart the API; it reloads whatever is in Production
```

Reproduce a metric:

```bash
dvc pull data/green_tripdata_2019-08.parquet.dvc
dvc repro pipelines/dvc.yaml        # run twice: the second time every stage is cached
dvc metrics show
```

## Module 3: serving, batch, streaming and release

### 1. Faster runtimes (`src/prodml/runtimes.py`)

Converts the model to ONNX Runtime and OpenVINO and compares speed and accuracy.

```bash
uv run python -m prodml.runtimes export       # models/model_core.onnx + models/openvino/
uv run python -m prodml.runtimes benchmark    # pickle vs ONNX Runtime vs OpenVINO
SERVING_FORMAT=openvino MODEL_SOURCE=local uv run uvicorn prodml.api.main:app --port 8000
```

OpenVINO cannot read the `LinearRegressor` operator that `skl2onnx` writes, so the export
first creates a plain MatMul + Add graph (`model_core.onnx`).
Measured on the real data (500 rows per call): pickle 55.7 ms, ONNX Runtime 6.5 ms,
OpenVINO 7.0 ms, with a maximum difference below 1e-5 versus pickle.

### 2. BentoML with adaptive batching (`serving/service.py`)

FastAPI scores one request at a time. BentoML collects the requests that arrive within a
few milliseconds into one array and calls the model once.

```bash
uv run bentoml serve serving.service:TaxiDuration --port 3100
curl -i -X POST localhost:3100/predict -H 'content-type: application/json' \
     -d '{"trips": [[43, 236, 2.5]]}'
```

Each row is `[PULocationID, DOLocationID, trip_distance]`. Settings (environment):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SOURCE` | `local` | `local` (files in `models/`) or `registry` |
| `MAX_BATCH_SIZE` | 64 | Largest batch BentoML builds |
| `MAX_LATENCY_MS` | 500 | Latency budget. If BentoML expects to answer later than this it returns **503** instead of making the client wait |
| `RELEASE_LABEL` | `stable` | Returned as `X-Release`, used by the canary |
| `FAKE_DELAY_MS` | 0 | Adds artificial latency to practise the auto-rollback |

A 503 under heavy load is load shedding, not a model error. Raise `MAX_LATENCY_MS` or add
capacity if you see too many.

### 3. Batch scoring (`src/prodml/batch.py`)

Scores a Parquet file of any size in chunks, so memory stays flat.

```bash
uv run python -m prodml.batch --make-sample data/scoring/big.parquet --rows 200000
uv run python -m prodml.batch --input data/scoring/big.parquet \
    --output data/scoring/output --run-date 2026-10-01 --source local
```

- Output: `data/scoring/output/run_date=2026-10-01/part-00000.parquet` plus `_report.json`
  (rows, rows/second, peak memory, cost per million rows).
- **Idempotent**: running the same `--run-date` again replaces that partition. It writes a
  temporary file and renames it, so a crash never leaves a half-written file.
- The model is scored in slices of 1,000 rows, because the vectorizer creates about 20,000
  columns per row and a larger dense matrix would run out of memory.
- Measured (200k rows, real model): 23 s, about 8,600 rows/s, 540 MB peak.

### 4. Airflow (`pipelines/dags/`, `src/prodml/orchestration.py`)

| DAG | Schedule | Tasks |
|---|---|---|
| `train_pipeline` | weekly | `FileSensor` -> validate -> prepare -> featurize -> train -> evaluate -> **branch** -> register or skip -> notify |
| `batch_scoring` | daily | `FileSensor` (input of that day) -> score -> notify |

Both use `retries=2`, a 5-minute retry delay, `catchup=False` and `max_active_runs=1`.
Only small values (run id, one metric) go through XCom. Heavy work lives in
`orchestration.py` and `batch.py`, which can be unit-tested without Airflow.

```bash
docker compose -f docker/docker-compose.serving.yml up -d --build airflow
# Airflow UI: http://localhost:8081  (the admin password is printed in the container log)
docker compose -f docker/docker-compose.serving.yml exec airflow \
    airflow dags backfill batch_scoring -s 2026-10-01 -e 2026-10-07
```

Put the day's input at `data/scoring/input/trips_<YYYY-MM-DD>.parquet` before you trigger.
The Airflow image is installed with Airflow's own constraints, so it only contains the base
dependencies: `batch_scoring` runs inside it, while the training stages of `train_pipeline`
need the full training environment (run `dvc repro` on your machine).

### 5. Streaming with Redis (`stream_producer.py`, `stream_consumer.py`)

Three streams: `trips:in` (events), `predictions:out` (results), `trips:dead` (dead letters).

```bash
docker compose -f docker/docker-compose.serving.yml up -d redis
export REDIS_URL=redis://localhost:6380/0                     # in every terminal

uv run python -m prodml.stream_consumer run --source local    # terminal 1
uv run python -m prodml.stream_producer --rate 100 --duration 20 --poison-every 25   # terminal 2
uv run python -m prodml.stream_consumer stats                 # latency percentiles
```

- Workers share a **consumer group**, so each event goes to one worker.
- The result is written first and **then** acknowledged (`XACK`): at-least-once delivery,
  an event is never lost.
- A crashed worker's unacknowledged events are read again on restart or claimed by another
  worker after 30 s.
- Unreadable events and events that fail 3 times go to the dead-letter stream.
- Measured (2,000 events, every 25th poison): 1,920 scored, 80 dead-lettered, 0 pending,
  p50 4 ms, p95 64 ms, p99 433 ms.

### 6. Load testing (`loadtest/locustfile.py`)

Traffic mix: 80% single prediction, 15% small batch, 5% health, 1–3 s think time, varied
payloads, ramp to 100 users.

```bash
TARGET=fastapi uv run locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 100 -r 20 -t 1m --csv reports/load/fastapi
TARGET=bento   uv run locust -f loadtest/locustfile.py --host http://localhost:3100 \
    --headless -u 100 -r 20 -t 1m --csv reports/load/bento
```

Use `WAIT_MIN=0 WAIT_MAX=0.05` for a saturation run. Run one server at a time and serve
the **same model** on both, otherwise the comparison is unfair (a gradient-boosted model from
the registry is much slower per request than the linear ONNX model).

### 7. Canary release (`deploy/`, `src/prodml/canary.py`)

```text
client ─▶ nginx :8080 ──90%──▶ model-v1 (stable)
                      └─10%──▶ model-v2 (canary)
```

```bash
docker compose -f docker/docker-compose.canary.yml up -d --build
bash deploy/canary_promote.sh 25      # 10, 25, 50 or 100
bash deploy/rollback.sh               # back to 100% stable, prints how long it took
uv run python -m prodml.canary watch  # auto-rollback watcher
```

- The traffic split is the block between `CANARY_SPLIT_START` and `CANARY_SPLIT_END` in
  `deploy/nginx/nginx.conf`. The scripts rewrite it and reload nginx without downtime.
- The watcher reads the nginx log every 10 s. After **two bad intervals in a row**
  (canary p95 above 300 ms, or error rate above 5%) it sets the canary share to 0%.
- Practise a failure: `FAKE_DELAY_MS=600 docker compose -f docker/docker-compose.canary.yml up -d model-v2`.
- **Shadow mode** (`deploy/nginx/nginx.shadow.conf`, start with `NGINX_CONF=nginx.shadow.conf`):
  users always get the stable answer, a copy of each request goes to the canary and its
  answer is discarded.

| Strategy | Traffic | User risk | Rollback | Use when |
|---|---|---|---|---|
| Blue/Green | 100% switches at once | medium | instant | big change, clean cut |
| Canary | gradual 10 -> 100% | low | seconds | normal model update |
| A/B test | split by user group | low | seconds | measure business impact |
| Shadow | copy of traffic, answers discarded | none | n/a | test before any user sees it |

## Module 4: monitoring and drift

The model keeps answering `200 OK` long after the world has changed, so we measure it.
Everything is small on purpose: no Evidently, no Loki, no extra Airflow.

| Technique | File | Question it answers |
|---|---|---|
| Live metrics | `src/prodml/monitoring/metrics.py`, `GET /metrics` | How is the service doing right now? |
| Serving log | `src/prodml/monitoring/serving_log.py` | What was the API actually asked? |
| Data drift (PSI, KS) | `src/prodml/monitoring/drift.py` | Did the inputs/outputs move away from training? |
| Concept drift (Page-Hinkley) | `src/prodml/monitoring/concept.py` | Did the *error* start to grow? (needs real durations) |
| History | `src/prodml/monitoring/store.py` (SQLite) | Drifting for how many days? |
| Retrain gate | `src/prodml/monitoring/retrain.py` | Should we retrain, and why / why not? |
| Alerts + dashboard | `monitoring/`, `docker/docker-compose.monitoring.yml` | Who gets told, and where do we look? |

```bash
export MODEL_SOURCE=local                        # or run MLflow, see Troubleshooting
uv run python -m prodml.monitoring freeze        # once: freeze 5,000 training rows
uv run uvicorn prodml.api.main:app --port 8000   # serves /metrics and writes the log
uv run python -m prodml.monitoring simulate --shift 1.5   # fake drifted traffic (demo)
uv run python -m prodml.monitoring drift         # PSI/KS table, stored in SQLite
uv run python -m prodml.monitoring drift --fail-on-drift  # exit 1 = CI turns red
uv run python -m prodml.monitoring gate          # RETRAIN or HOLD, with reasons
uv run python -m prodml.monitoring concept --scored data/scoring/<file>.parquet
docker compose -f docker/docker-compose.monitoring.yml up -d   # Prometheus + Grafana
```

PSI rule of thumb: below 0.10 stable, 0.10-0.25 watch, above 0.25 drifted. The drift job
writes its numbers to SQLite and `/metrics` re-exports them as `prodml_drift_psi`, so the
`FeatureDrift` alert and the Grafana panel need no extra pipeline.
Runbook: `docs/RUNBOOK.md`. Report: `reports/module-4.md`.

## Configuration

Environment variables (see `src/prodml/config.py` for all defaults; a `.env` file is read
too):

| Variable | Purpose |
|---|---|
| `MODEL_SOURCE` | `registry` (default) loads `models:/<name>/<stage>` from MLflow; `local` loads files from `MODELS_DIR` |
| `MLFLOW_TRACKING_URI` | MLflow server, e.g. `http://localhost:5000` |
| `REGISTERED_MODEL_NAME` / `MODEL_STAGE` | Registry model and stage to serve (`ride-duration-predictor` / `Production`) |
| `SERVING_FORMAT` | `onnx`, `pickle` or `openvino` (local source only) |
| `DATA_PATH`, `MODELS_DIR` | Dataset and model folders |
| `PROMOTION_MARGIN` | Fraction by which a candidate must beat Production's MAE to reach Staging |
| `GATE_MAX_REGRESSION` | CI fails if MAE is worse than the baseline by more than this fraction (default 0.05) |
| `REDIS_URL` | Redis for streaming (default `redis://localhost:6379/0`) |
| `SERVING_LOG_PATH` | Where the API appends each scored trip (`""` switches it off) |
| `REFERENCE_PATH`, `MONITORING_DB` | Frozen reference sample and the SQLite drift/retrain history |
| `PSI_WARN`, `PSI_DRIFT` | PSI thresholds (default 0.10 and 0.25) |
| `LOG_LEVEL` | Logging verbosity |

## Testing and code quality

```bash
uv run ruff check . && uv run black --check . && uv run pytest
```

pytest enforces **70% coverage**. The suite needs no running MLflow, Redis or Docker: it
trains a tiny model on fake data and uses `fakeredis`.

## Project structure

```text
.
├── README.md
├── pyproject.toml, uv.lock, .python-version
├── params.yaml                      # hyperparameters tracked by DVC
├── pipelines/
│   ├── dvc.yaml                     # prepare -> featurize -> train -> evaluate
│   └── dags/                        # Airflow: train_pipeline.py, batch_scoring.py
├── src/prodml/
│   ├── config.py                    # all settings (env vars)
│   ├── data.py, features.py         # loading, cleaning, shared features
│   ├── train.py, export.py          # Module 1: baseline training and ONNX export
│   ├── models.py, pyfunc_model.py   # Module 2: model families and MLflow wrapper
│   ├── tracking.py, experiments.py  # Module 2: runs, sweep, autolog
│   ├── registry.py, gate.py         # Module 2: stages, promotion, CI gate
│   ├── pipeline.py, params.py       # Module 2: DVC stages
│   ├── predict.py                   # loads the model (registry or local) and scores
│   ├── runtimes.py                  # Module 3: ONNX Runtime, OpenVINO
│   ├── batch.py                     # Module 3: chunked batch scoring
│   ├── orchestration.py             # Module 3: functions the Airflow DAGs call
│   ├── stream_producer.py, stream_consumer.py   # Module 3: Redis Streams
│   ├── canary.py                    # Module 3: traffic split and auto-rollback
│   ├── logging_conf.py, utils.py
│   └── api/                         # FastAPI: main.py, schemas.py
├── serving/service.py               # BentoML service
├── loadtest/locustfile.py           # Locust traffic mix
├── deploy/                          # nginx configs, canary_promote.sh, rollback.sh
├── docker/
│   ├── Dockerfile, mlflow.Dockerfile, docker-compose.yml     # Modules 1–2
│   ├── airflow.Dockerfile, bento.Dockerfile
│   └── docker-compose.serving.yml, docker-compose.canary.yml # Module 3
├── .github/workflows/               # ci.yml, continous-training.yml
├── tests/                           # pytest suite
├── models/                          # model.pkl, model.onnx, metadata.json
├── metrics/                         # scores.json, train_run.json (DVC metrics)
├── reports/                         # module-1.md, module-2.md, module-3.md, load/
├── scripts/make_synthetic_trips.py  # offline stand-in data (smoke tests only)
├── notebooks/                       # baseline notebook
└── data/                            # datasets (git-ignored, tracked by DVC)
```

## Model evaluation

| Model | Validation MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear baseline (Module 1) | 3.8392 min | 5.7529 min | n/a |
| XGBoost in Production (Module 2) | 3.3703 min | 5.1630 min | 0.7927 |

Metrics are written to `metrics/scores.json` by the DVC `evaluate` stage and logged to the
MLflow run that produced the model. They change when the data or parameters change.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `AttributeError: module 'ast' has no attribute 'Str'` when starting MLflow | The venv uses Python 3.14. `echo 3.12 > .python-version && rm -rf .venv && uv sync --python 3.12 --extra dev --extra train` |
| `module 'sqlalchemy' has no attribute 'Engine'` | The lock file holds SQLAlchemy 1.4. Delete `uv.lock`, run `uv lock`, then `uv sync` |
| `Failed to spawn: locust` or `bentoml` | The extra is not installed. Add it to the **same** `uv sync` command (`--extra load`, `--extra serving`) |
| `duplicate key` in `pyproject.toml` | An extra is declared twice under `[project.optional-dependencies]` |
| `Unable to allocate 15 GiB` in batch scoring | Old `predict_frame`. Use the sliced version (1,000 rows per model call) |
| `NOAUTH` / `HELLO must be called with the client already authenticated` | You reached another Redis that has a password. Use the Compose Redis and `export REDIS_URL=redis://localhost:6380/0` |
| `Connection refused` on 6379 or 6380 | Redis is not running. `docker compose -f docker/docker-compose.serving.yml up -d redis` |
| 503 from BentoML under load | Load shedding. Raise `MAX_LATENCY_MS`, lower the load, or add capacity |
| API returns 503 `Model is not loaded` | No model in `Production`. Run `registry show`, promote one, or use `MODEL_SOURCE=local` |
| `404` on `/` | Expected. The API has no home page; open `/docs` |
| Docker "container name already in use" | `docker rm -f <name>` or `docker start <name>` |

## Docker image

```bash
docker build -f docker/Dockerfile -t ahmdooz/prodml-api:0.2.0 .
docker run --rm -p 8000:8000 -e MODEL_SOURCE=local ahmdooz/prodml-api:0.2.0
```

The published image: https://hub.docker.com/r/ahmdooz/prodml-api

## Reports

- `reports/module-1.md`, `reports/module-2.md`: decisions and evidence for each module.
- `reports/module-3.md`: inference-pattern decision, load-test tables, batch and streaming
  numbers, release strategy table. **Replace the sample numbers with the ones you measure.**

- `reports/module-4.md`: metric contract, drift tests, alerts, what was and was not tested.

## License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the project under an open-source license.
