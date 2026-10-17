"""The FastAPI service.   uvicorn prodml.api.main:app --reload --port 8000"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.features import build_features
from prodml.logging_conf import correlation_id_var, setup_logging
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
    except Exception:
        logger.exception("model load failure")  # ERROR + traceback in the log
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


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health(predictor: DurationPredictor = Depends(get_predictor)) -> dict:
    """200 only if the model is really in memory (not just 'process is alive')."""
    return {"status": "healthy", "model_loaded": True}


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
    return BatchPredictionResponse(
        predictions=predictions,
        model_version=predictor.version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 3),
    )
