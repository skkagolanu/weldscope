"""Bounded, sanitized JSONL events for request correlation and operational diagnosis."""

import json
import logging
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock

_LOCK = RLock()


@lru_cache(maxsize=8)
def _sink(path):
    with _LOCK:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"weldscope.events.{destination.resolve()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = RotatingFileHandler(destination, maxBytes=5_000_000, backupCount=3)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console)
        return logger


def record_event(path, event):
    """Only accept a fixed schema; questions, answers and exception text never enter logs."""
    allowed = {
        "timestamp",
        "request_id",
        "trace_id",
        "status",
        "route",
        "error_type",
        "timings",
        "retrieved_count",
        "reranked_count",
        "model",
        "cold",
        "token_usage",
        "answer_check_status",
    }
    safe = {key: value for key, value in event.items() if key in allowed}
    if isinstance(safe.get("token_usage"), dict):
        safe["token_usage"] = {
            key: value
            for key, value in safe["token_usage"].items()
            if key
            in {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "queue_time",
                "prompt_time",
                "completion_time",
                "total_time",
            }
            and isinstance(value, (int, float))
        }
    if path:
        try:
            _sink(path).info(json.dumps(safe, allow_nan=False))
        except (OSError, ValueError, TypeError):
            logging.getLogger(__name__).warning("Operational event could not be persisted")
