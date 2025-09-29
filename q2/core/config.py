from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    device: str = os.getenv("Q2_DEVICE", "auto")
    mixed_precision: bool = _as_bool(os.getenv("Q2_MIXED_PRECISION"), True)
    torch_compile: bool = _as_bool(os.getenv("Q2_TORCH_COMPILE"), False)
    log_level: str = os.getenv("Q2_LOG_LEVEL", "INFO")
    prometheus_enabled: bool = _as_bool(os.getenv("Q2_PROMETHEUS_ENABLED"), True)


SETTINGS = Settings()
