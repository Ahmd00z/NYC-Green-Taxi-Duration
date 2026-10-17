# NYC Green Taxi Trip-Duration Prediction — MLOps

An end-to-end MLOps project that predicts NYC green taxi trip duration in minutes using a scikit-learn Linear Regression model. The project includes data processing, feature engineering, model training, ONNX export, automated testing, a FastAPI prediction service, structured logging, and Docker deployment.

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
│       └── api/
│           ├── main.py
│           └── schemas.py
├── tests/
├── docker/
│   ├── Dockerfile
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
