# NYC Green Taxi Trip-Duration Prediction — MLOps

[![CI](https://github.com/Ahmd00z/NYC-Green-Taxi-Duration/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmd00z/NYC-Green-Taxi-Duration/actions/workflows/ci.yml)

An end-to-end MLOps project that predicts NYC green taxi trip duration in minutes. Module 1 turned a notebook into a tested, containerized FastAPI service. **Module 2 makes every model reproducible and auditable and removes the human from the retraining loop:** experiments are tracked in MLflow (three model families), models are promoted through a registry (`None → Staging → Production`), data and pipeline are versioned with DVC, CI gates on model quality, infrastructure is Terraform code, and a continuous-training workflow retrains, compares against Production and promotes to Staging automatically.

## Module 2 at a glance

```text
 raw data ──dvc──▶ prepare ─▶ featurize ─▶ train ─▶ evaluate ─▶ metrics/scores.json
 (DVC remote:       │                        │
  local folder)       │                        ▼
                    │                  MLflow run  (params · metrics · plots · requirements.txt
                    │                   tags: git_commit, data_version, author, framework)
                    │                        │ promote_if_better()
                    ▼                        ▼
              data_version  ─────────▶  Model Registry:  None → Staging → (human approval) → Production
                                                                                      │
                                    API loads  models:/ride-duration-predictor/Production
                                    (swap the model = change a stage, no code change, no rebuild)
```

| Piece | Where |
|---|---|
| MLflow server (SQLite + local volume) | `docker/docker-compose.yml` |
| 3 model families + sweep + autolog | `src/prodml/models.py`, `tracking.py`, `experiments.py` |
| Registry and promotion logic | `src/prodml/registry.py` |
| Serve by stage | `src/prodml/predict.py` (`MODEL_SOURCE=registry`) |
| DVC pipeline | `pipelines/dvc.yaml`, `params.yaml`, `src/prodml/pipeline.py`, `.dvc/config` |
| CI (lint → test → gate → build) | `.github/workflows/ci.yml`, `src/prodml/gate.py` |
| Continuous training | `.github/workflows/continuous-training.yml` |
| Terraform | `infra/` |

### Run the Module 2 stack

```bash
docker compose -f docker/docker-compose.yml up -d mlflow     # no passwords, no .env needed
export MLFLOW_TRACKING_URI=http://localhost:5000     # MLflow UI: http://localhost:5000

uv sync --extra dev --extra train
uv run python -m prodml.experiments                  # 3 families + 10-trial sweep + autolog = 15 runs
uv run python -m prodml.registry promote-if-better --run-id <best-run-id>   # -> Staging
uv run python -m prodml.registry promote-production --version 1             # human step -> Production
docker compose -f docker/docker-compose.yml up -d api                       # serves models:/.../Production
```

Switch the served model with **no code change and no rebuild**:
`uv run python -m prodml.registry promote-production --version 2 && docker compose -f docker/docker-compose.yml restart api`.

### Reproduce a metric

```bash
# once: dvc add data/green_tripdata_2019-08.parquet && dvc push   (storage = ../dvc-storage)
dvc pull data/green_tripdata_2019-08.parquet.dvc
dvc repro pipelines/dvc.yaml            # run again: every stage is skipped (cached)
dvc metrics show                        # same MAE as the MLflow run tagged with this data_version
```

Offline / no MLflow? `MODEL_SOURCE=local` (or `python -m prodml.train --local`) keeps the Module 1 behaviour.

## Features

- **Machine Learning:** Predicts trip duration using pickup location, drop-off location, and trip distance.
- **Feature Engineering:** Shared preprocessing pipeline for training and inference.
- **Model Export:** Supports ONNX inference and Pickle-based model artifacts.
- **FastAPI:** REST API with single-trip and batch prediction endpoints.
- **Validation:** Pydantic request validation and readable error responses.
- **Observability:** JSON logs, request correlation IDs, and latency tracking.
- **Testing:** Automated pytest suite with coverage reporting.
- **Docker:** Multi-stage image with a non-root runtime user.

## Quickstart — Run with Docker

Make sure Docker is installed and running, then execute:

```bash
docker pull ahmdooz/prodml-api:0.1.0

docker run -d \
  --name prodml-api \
  -p 8000:8000 \
  ahmdooz/prodml-api:0.1.0
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

Open the interactive API documentation:

http://localhost:8000/docs

## Make a Prediction

Send a request to the prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PULocationID": 43,
    "DOLocationID": 236,
    "trip_distance": 2.5
  }'
```

Example response:

```json
{
  "prediction": 8.07,
  "model_version": "0.1.0",
  "correlation_id": "example-request-id",
  "latency_ms": 1.37
}
```

The response above illustrates the response format. Actual predictions and latency depend on the model and runtime environment.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks whether the model is loaded and ready |
| GET | `/metadata` | Returns model version, training information, feature details, and model hash |
| POST | `/predict` | Predicts the duration of a single trip |
| POST | `/predict/batch` | Predicts durations for multiple trips |

### Batch Prediction

Send multiple trips in one request:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "PULocationID": 43,
        "DOLocationID": 236,
        "trip_distance": 2.5
      },
      {
        "PULocationID": 161,
        "DOLocationID": 237,
        "trip_distance": 4.2
      }
    ]
  }'
```

Invalid inputs, such as negative trip distances, are rejected by request validation.

## Local Development

### 1. Requirements

- Python 3.11 or a compatible version supported by the project
- `uv`
- Git
- Docker (for containerized deployment)

### 2. Install dependencies

From the project root:

```bash
uv sync --extra dev
```

### 3. Prepare the dataset

Place the NYC green taxi dataset under `data/`.

The training pipeline expects the columns required by the project, including pickup and drop-off timestamps, pickup and drop-off location IDs, and trip distance.

The default dataset path is configured in `src/prodml/config.py`. You can override it using `DATA_PATH`.

Raw data is excluded from Git by default.

### 4. Train the model

```bash
uv run python -m prodml.train
```

The training pipeline loads and cleans the dataset, performs a chronological train/validation split, builds features, trains the model, evaluates its performance, and saves model artifacts under `models/`.

### 5. Run the API locally

```bash
uv run uvicorn prodml.api.main:app --reload --port 8000
```

Then visit:

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Model metadata: http://localhost:8000/metadata

### 6. Export and benchmark ONNX

```bash
uv run python -m prodml.export
```

This command exports the model to ONNX and checks prediction parity between the Pickle and ONNX implementations. It also runs the project's inference benchmark.

## Testing and Code Quality

Run the test suite:

```bash
uv run pytest -v
```

Run linting:

```bash
uv run ruff check src tests
```

Check formatting:

```bash
uv run black --check src tests
```

## Docker Build and Publish

Build the image from the project root:

```bash
docker build \
  -f docker/Dockerfile \
  -t ahmdooz/prodml-api:0.1.0 \
  .
```

Run the image locally:

```bash
docker run --rm -p 8000:8000 ahmdooz/prodml-api:0.1.0
```

Log in to Docker Hub and push the image:

```bash
docker login -u ahmdooz
docker push ahmdooz/prodml-api:0.1.0
```

The published image is available at:

**Docker Hub:** https://hub.docker.com/r/ahmdooz/prodml-api

## Configuration

The service supports the following environment variables:

| Variable | Purpose |
|---|---|
| `DATA_PATH` | Path to the training dataset |
| `MODELS_DIR` | Directory containing model artifacts |
| `SERVING_FORMAT` | Selects `onnx` or `pickle` inference |
| `LOG_LEVEL` | Controls logging verbosity |
| `MODEL_SOURCE` | `registry` (default, Module 2) loads `models:/<name>/<stage>` from MLflow; `local` loads files from `MODELS_DIR` |
| `MLFLOW_TRACKING_URI` | MLflow server (`http://localhost:5000`; inside Compose: `http://mlflow:5000`) |
| `REGISTERED_MODEL_NAME` / `MODEL_STAGE` | Which registry model and stage to serve (`ride-duration-predictor` / `Production`) |
| `PROMOTION_MARGIN` | Fraction by which a candidate must beat Production's MAE to reach Staging (default `0.0`) |
| `GATE_MAX_REGRESSION` | CI fails if MAE is worse than the baseline by more than this fraction (default `0.05`) |

Refer to `src/prodml/config.py` for the actual defaults and supported configuration.

## Project Structure

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── notebooks/
│   └── 00-baseline.ipynb
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       ├── predict.py
│       ├── export.py
│       ├── logging_conf.py
│       ├── utils.py
│       ├── models.py            # linear / xgboost / random_forest behind one interface
│       ├── pyfunc_model.py      # the MLflow wrapper all three share
│       ├── tracking.py          # logs a complete, reproducible MLflow run
│       ├── experiments.py       # 3 families + Optuna sweep + autolog demo
│       ├── registry.py          # stages, promote_if_better, CLI
│       ├── pipeline.py          # the DVC stages
│       ├── gate.py              # CI model-quality gate
│       ├── params.py
│       └── api/
│           ├── main.py
│           └── schemas.py
├── params.yaml                  # hyperparameters tracked by DVC
├── pipelines/dvc.yaml           # prepare -> featurize -> train -> evaluate
├── .github/workflows/           # ci.yml, continuous-training.yml
├── infra/                       # Terraform (docker provider)
├── scripts/make_synthetic_trips.py   # offline stand-in data (smoke tests only)
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── mlflow.Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── model.pkl
│   ├── model.onnx
│   └── metadata.json
├── data/
└── reports/
    └── module-1.md
```

## Model Evaluation

The recorded training run produced the following validation metrics:

| Metric | Value |
|---|---:|
| Training rows | 346,033 |
| Validation rows | 86,509 |
| Validation MAE | 3.8392 minutes |
| Validation RMSE | 5.7529 minutes |

These results describe the recorded run and may change when the model is retrained with different data or settings.

## Technology Stack

- Python
- scikit-learn
- pandas and NumPy
- ONNX and ONNX Runtime
- FastAPI and Pydantic
- pytest
- Ruff and Black
- `uv`
- Docker

## Engineering Goals

This project demonstrates an end-to-end MLOps workflow: reproducible dependency management, model training and evaluation, artifact serialization, model serving, API validation, automated tests, logging, and containerized deployment.

## License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the project under an open-source license.
