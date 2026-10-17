"""Structured JSON logging + a correlation ID that follows a request everywhere."""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

from prodml.config import settings

# Holds the id of the request currently being handled ("-" outside a request).
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")

# Attributes every LogRecord has. Anything else came from `extra={...}`.
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Turns a log record into one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        # logger.info("hi", extra={"latency_ms": 3}) -> adds "latency_ms" to the JSON
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                log[key] = value
        if record.exc_info:
            log["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def setup_logging(level: str | None = None) -> None:
    """Send ALL logs (ours + uvicorn's) through one JSON handler. Safe to call twice."""
    root = logging.getLogger()
    root.setLevel((level or settings.log_level).upper())

    if not any(getattr(h, "_prodml", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._prodml = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # Make uvicorn use the root JSON handler instead of its own plain-text one.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
