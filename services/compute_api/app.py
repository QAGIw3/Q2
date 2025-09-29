from __future__ import annotations

import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, generate_latest

from q2.core.config import SETTINGS
from q2.core.logging import configure_logging, new_request_id
from q2.core.telemetry import setup_tracing
from q2.compute.device import seed_all, torch_setup
from .routes.qvnn import router as qvnn_router

configure_logging(SETTINGS.log_level)
setup_tracing()
seed_all()
torch_setup()

app = FastAPI(title="Q2 Compute API", version="v1")

# --- Metrics
REGISTRY = CollectorRegistry()
REQUESTS = Counter("q2_requests_total", "Total HTTP requests", ["path", "method", "code"], registry=REGISTRY)


@app.middleware("http")
async def _metrics_and_request_id(request: Request, call_next: Callable):
    request_id = new_request_id()
    try:
        response: Response = await call_next(request)
        code = str(response.status_code)
    except Exception as exc:  # noqa: BLE001
        code = "500"
        logging.getLogger("q2").exception("unhandled error", extra={"request_id": request_id})
        return JSONResponse(status_code=500, content={"code": "internal_error", "message": "internal server error", "trace_id": request_id})
    finally:
        try:
            REQUESTS.labels(path=request.url.path, method=request.method, code=code).inc()
        except Exception:
            pass
    response.headers["x-request-id"] = request_id
    return response


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    if not SETTINGS.prometheus_enabled:
        return PlainTextResponse("disabled", status_code=404)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# --- Routes
app.include_router(qvnn_router, prefix="/v1/compute")