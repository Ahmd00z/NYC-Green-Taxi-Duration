# prodml — NYC Taxi Trip-Duration Service

A scikit-learn model that predicts how long a NYC green-taxi trip will take (in
minutes), packaged as a pip-installable Python package and served by a tested
FastAPI service inside a multi-stage, non-root Docker image.

## Quickstart — from zero to a prediction in 3 commands

```bash
# 1. Start the service (nothing to clone, the model is inside the image)
docker run -d -p 8000:8000 --name prodml-api nourhassan8357/prodml-api:latest

# 2. Check it is healthy (200 only if the model is loaded in memory)
curl -s http://localhost:8000/health

# 3. Ask for a prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 43, "DOLocationID": 236, "trip_distance": 2.5}'
```

Response:

```json
{
  "prediction": 8.07,
  "model_version": "0.1.0",
  "correlation_id": "f2201a26-0833-46a1-91a1-f0333f149682",
  "latency_ms": 1.37
}
```

Interactive docs (Swagger UI): <http://localhost:8000/docs>

## Endpoints

| Method | Path             | What it does                                                  |
|--------|------------------|---------------------------------------------------------------|
| GET    | `/health`        | 200 only if the model is loaded in memory, otherwise 503      |
| GET    | `/metadata`     | version, training date, feature names, framework, SHA-256    |
| POST   | `/predict`       | one trip in, one prediction out                               |
| POST   | `/predict/batch` | `{"inputs": [...]}` in, `{"predictions": [...]}` out          |

Bad input (e.g. `trip_distance: -5`) returns `422` with a readable message.
Every response carries an `X-Request-ID` header, the same id that appears in
every JSON log line of that request.

## Developer workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                                # install the package
ruff check src tests && black --check src tests        # lint
pytest -v                                              # test (fails under 70% coverage)
python -m prodml.train                                 # train -> models/
uvicorn prodml.api.main:app --reload --port 8000       # serve locally
python -m prodml.export                                # pickle vs ONNX parity + latency
```

Training needs one month of NYC green taxi data in `data/`
(`data/green_tripdata_2019-08.csv.gz`, or any `.parquet` file with the same
columns — set `DATA_PATH`).

## Build and publish the image

```bash
python -m prodml.train                                 # the image copies models/
docker build -f docker/Dockerfile -t <user>/prodml-api:0.1.0 -t <user>/prodml-api:latest .
docker run --rm -p 8000:8000 <user>/prodml-api:0.1.0
docker push <user>/prodml-api:0.1.0 && docker push <user>/prodml-api:latest
docker compose -f docker/docker-compose.yml up --build   # or run it with compose
```

## Configuration (environment variables)

| Variable         | Default                              | Meaning                          |
|------------------|--------------------------------------|----------------------------------|
| `DATA_PATH`      | `data/green_tripdata_2019-08.csv.gz` | training data                    |
| `MODELS_DIR`     | `models`                             | where artifacts are saved/loaded |
| `SERVING_FORMAT` | `onnx`                               | `onnx` or `pickle`               |
| `LOG_LEVEL`      | `INFO`                               | `DEBUG` shows feature vectors    |

## Repository layout

```
.
├── README.md
├── pyproject.toml              # package metadata, dependencies, pytest/coverage gate
├── .gitignore  .dockerignore  .pre-commit-config.yaml
├── notebooks/00-baseline.ipynb # the messy "before" picture
├── src/prodml/
│   ├── config.py               # all settings (env vars)
│   ├── logging_conf.py         # JSON logs + correlation id
│   ├── utils.py                # @timed decorator
│   ├── data.py                 # load + split
│   ├── features.py             # feature engineering (shared by train & serve)
│   ├── train.py                # python -m prodml.train
│   ├── predict.py              # DurationPredictor class
│   ├── export.py               # ONNX export + pickle-vs-ONNX benchmark
│   └── api/
│       ├── main.py             # FastAPI app
│       └── schemas.py          # Pydantic request/response models
├── tests/                      # pytest suite
├── docker/                     # Dockerfile + docker-compose.yml
├── models/                     # trained artifacts (model.pkl, model.onnx, metadata.json)
├── data/                       # raw data (not committed)
└── reports/module-1.md         # lab report
```
