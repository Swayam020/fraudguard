"""Centralized logging setup for FraudGuard.

Call `setup_logging()` once at application startup. Subsequent
`logging.getLogger(__name__)` calls across the codebase pick up
the configured handlers automatically.
"""

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Each record becomes one JSON object, suitable for log aggregators
    (Loki, Datadog, CloudWatch) that parse structured logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        # The minimum useful fields. Add more (e.g. trace_id) as needed.
        log_dict: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # If the caller did logger.error("boom", exc_info=True), include
        # the traceback string in the JSON.
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger.

    Args:
        level: minimum level to emit ("DEBUG", "INFO", "WARNING", ...)
        json_format: if True, emit one JSON object per line; else
            emit human-readable text.

    Idempotent: calling twice doesn't double-attach handlers.
    """

    # The root logger sits at the top of the logger hierarchy.
    # All `logging.getLogger(name)` calls inherit its handlers by default.
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Wipe any handlers attached on a previous call — prevents duplicates
    # when setup_logging() is invoked twice (e.g. in tests).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # StreamHandler writes to stderr by default; we redirect to stdout
    # so logs interleave properly with print() in containerized envs.
    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        # %(asctime)s = timestamp, %(name)s = logger name, etc.
        # These %-style placeholders are Python logging's own mini-template language.
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)
