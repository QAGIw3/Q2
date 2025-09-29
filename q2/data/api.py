from __future__ import annotations

from pydantic import BaseModel, Field


class QvnnTrainRequest(BaseModel):
    n: int = Field(4096, ge=1)
    in_dim: int = Field(64, ge=1)
    hidden: int = Field(128, ge=1)
    out_dim: int = Field(1, ge=1)
    steps: int = Field(200, ge=1)
    lr: float = Field(1e-3, gt=0)
    amp: bool = True


class QvnnTrainResponse(BaseModel):
    loss: float
    wall_time_ms: float
    device: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    trace_id: str | None = None