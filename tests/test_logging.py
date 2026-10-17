import json
import logging

from prodml.logging_conf import JsonFormatter, correlation_id_var
from prodml.utils import timed


def test_json_formatter_has_required_keys():
    token = correlation_id_var.set("abc-123")
    try:
        record = logging.LogRecord("x", logging.INFO, "f.py", 1, "hello", (), None)
        record.latency_ms = 3.5  # what `extra={...}` does
        line = json.loads(JsonFormatter().format(record))
    finally:
        correlation_id_var.reset(token)
    for key in ("timestamp", "level", "logger", "message", "correlation_id"):
        assert key in line
    assert line["correlation_id"] == "abc-123"
    assert line["latency_ms"] == 3.5


def test_timed_decorator_returns_value_and_keeps_name():
    @timed
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add.__name__ == "add"
