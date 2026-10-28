"""The FastAPI service.   uvicorn prodml.api.main:app --reload --port 8000"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.features import build_features
from prodml.logging_conf import correlation_id_var, setup_logging
from prodml.monitoring import metrics, serving_log
from prodml.predict import DurationPredictor

setup_logging()
logger = logging.getLogger("prodml.api")


# ------------------------------------------------------------------ startup
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs ONCE when the server starts: load the model into memory.

    (Loading inside the request handler would reload it on every request.)
    """
    app.state.predictor = None
    try:
        app.state.predictor = DurationPredictor().load()
        logger.info("model loaded", extra={"version": app.state.predictor.version})
        metrics.set_model_info(app.state.predictor.version)
    except Exception:
        logger.exception("model load failure")  # ERROR + traceback in the log
    metrics.register_drift_collector()  # drift numbers appear on /metrics too
    yield
    app.state.predictor = None


app = FastAPI(title="NYC Taxi Duration API", version="0.1.0", lifespan=lifespan)


def get_predictor(request: Request) -> DurationPredictor:
    """Dependency: hands the loaded model to an endpoint, or 503 if it isn't loaded."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return predictor


# --------------------------------------------------------------- middleware
@app.middleware("http")
async def correlation_middleware(request: Request, call_next: Any) -> Any:
    """Give every request a uuid, log it, return it as X-Request-ID.
    Also turns any unexpected crash into a clean 500 (traceback goes to the
    LOG, never to the client)."""
    request_id = str(uuid.uuid4())
    token = correlation_id_var.set(request_id)
    start = time.perf_counter()
    metrics.INFLIGHT.inc()
    logger.info(
        "request received", extra={"method": request.method, "path": request.url.path}
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error")
        response = JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )
    metrics.INFLIGHT.dec()
    # route TEMPLATE ("/predict"), never the raw URL: bounded label values
    route = getattr(request.scope.get("route"), "path", "unmatched")
    metrics.HTTP_REQUESTS.labels(route, str(response.status_code)).inc()
    metrics.HTTP_LATENCY.labels(route).observe(time.perf_counter() - start)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - start) * 1000, 3),
        },
    )
    correlation_id_var.reset(token)
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> Any:
    """Bad payload -> 422 with a readable message instead of a stack trace."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    logger.error("validation rejection", extra={"errors": errors})
    return JSONResponse(status_code=422, content={"detail": errors})


def _observe(version: str, trips: list, predictions: list[float]) -> None:
    """Update the live metrics and append to the serving log (never raises)."""
    metrics.record_predictions(version, predictions, [t.trip_distance for t in trips])
    serving_log.append(
        {
            "PULocationID": t.PULocationID,
            "DOLocationID": t.DOLocationID,
            "trip_distance": t.trip_distance,
            "prediction": p,
        }
        for t, p in zip(trips, predictions)
    )


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health(predictor: DurationPredictor = Depends(get_predictor)) -> dict:
    """200 only if the model is really in memory (not just 'process is alive')."""
    return {"status": "healthy", "model_loaded": True}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Scraped by Prometheus every few seconds."""
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/metadata")
def metadata(predictor: DurationPredictor = Depends(get_predictor)) -> dict:
    return predictor.get_metadata()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: PredictionRequest, predictor: DurationPredictor = Depends(get_predictor)
) -> PredictionResponse:
    if req.trip_distance > 100:
        logger.warning(
            "trip_distance outside training range",
            extra={"trip_distance": req.trip_distance},
        )

    features = build_features(req.PULocationID, req.DOLocationID, req.trip_distance)
    logger.debug("feature vector", extra={"features": features})

    start = time.perf_counter()
    prediction = predictor.predict_one(features)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "prediction served",
        extra={"prediction": prediction, "latency_ms": round(latency_ms, 3)},
    )
    _observe(predictor.version, [req], [prediction])
    return PredictionResponse(
        prediction=prediction,
        model_version=predictor.version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    req: BatchPredictionRequest,
    predictor: DurationPredictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    rows = [
        build_features(t.PULocationID, t.DOLocationID, t.trip_distance)
        for t in req.inputs
    ]

    start = time.perf_counter()
    predictions = predictor.predict_batch(rows)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "batch prediction served",
        extra={"batch_size": len(rows), "latency_ms": round(latency_ms, 3)},
    )
    _observe(predictor.version, req.inputs, predictions)
    return BatchPredictionResponse(
        predictions=predictions,
        model_version=predictor.version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch/partial")
def predict_batch_partial(
    payload: dict[str, Any],
    predictor: DurationPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    """Like /predict/batch, but one bad row does not reject the whole request.

    Every row is checked on its own. Good rows get a prediction; bad rows get an
    error message. The answer is always 200, and `results` keeps the input order.
    """
    items = payload.get("inputs")
    if not isinstance(items, list) or not 1 <= len(items) <= 1000:
        raise HTTPException(status_code=422, detail="inputs must be a list of 1-1000")

    results: list[dict[str, Any]] = [{} for _ in items]
    good_index: list[int] = []
    good_rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        try:
            trip = PredictionRequest.model_validate(item)  # same rules as /predict
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(str(p) for p in first["loc"])
            results[i] = {"index": i, "error": f"{field}: {first['msg']}"}
            continue
        good_index.append(i)
        good_rows.append(
            build_features(trip.PULocationID, trip.DOLocationID, trip.trip_distance)
        )

    if good_rows:  # ONE model call for all the valid rows
        for i, value in zip(good_index, predictor.predict_batch(good_rows)):
            results[i] = {"index": i, "prediction": value}
        _observe(
            predictor.version,
            [PredictionRequest.model_validate(items[i]) for i in good_index],
            [results[i]["prediction"] for i in good_index],
        )

    return {
        "results": results,
        "ok": len(good_rows),
        "failed": len(items) - len(good_rows),
        "model_version": predictor.version,
        "correlation_id": correlation_id_var.get(),
    }
