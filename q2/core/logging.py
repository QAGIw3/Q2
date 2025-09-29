from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Mapping
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": f"{time.time():.6f}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k in ("request_id", "trace_id"):
            v = getattr(record, k, None)
            if v:
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind(record: logging.LogRecord, fields: Mapping[str, Any]) -> None:
    for k, v in fields.items():
        setattr(record, k, v)


def with_request_id(extra: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    base = {"request_id": new_request_id()}
    if extra:
        base.update(extra)
    return base
