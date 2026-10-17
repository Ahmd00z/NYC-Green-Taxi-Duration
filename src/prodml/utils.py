"""Small helpers. Contains our own decorator, @timed."""

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("prodml.utils")


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: log how long the wrapped function took (DEBUG level)."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "function timed",
            extra={"function": func.__name__, "duration_ms": round(duration_ms, 3)},
        )
        return result

    return wrapper
