import logging
import os
import time

from fastapi import FastAPI, Request
from prometheus_client import REGISTRY, Counter, Histogram, start_http_server
from prometheus_client.exposition import generate_latest

logger = logging.getLogger(__name__)

# --- Prometheus Metrics Definitions ---

# A counter to track the total number of HTTP requests
REQUESTS = Counter("http_requests_total", "Total number of HTTP requests.", ["method", "path", "status_code"])

# A histogram to track the latency of HTTP requests
LATENCY = Histogram("http_request_latency_seconds", "HTTP request latency in seconds.", ["method", "path"])

# --- Standard Metrics ---

# --- Workflow Metrics ---
WORKFLOW_COMPLETED_COUNTER = Counter(
    "workflow_completed_total", "Total number of completed workflows", ["status"]  # e.g., 'COMPLETED', 'FAILED'
)

WORKFLOW_DURATION_HISTOGRAM = Histogram("workflow_duration_seconds", "Histogram of workflow execution time in seconds")

TASK_COMPLETED_COUNTER = Counter(
    "task_completed_total", "Total number of completed tasks", ["status"]  # e.g., 'COMPLETED', 'FAILED', 'CANCELLED'
)

AGENT_TASK_PROCESSED_COUNTER = Counter(
    "agent_task_processed_total",
    "Total number of tasks processed by the agent",
    ["agent_id", "personality", "status"],  # e.g., 'COMPLETED', 'FAILED'
)

_METRICS_STARTED = False


def setup_metrics(app: FastAPI, app_name: str):
    """Initialize Prometheus metrics and middleware (idempotent & reload-safe).

    During uvicorn --reload two processes are spawned (a monitor + the worker).
    Both would attempt to bind the metrics port causing: OSError: [Errno 98] Address already in use.
    We guard with a module-level flag and also catch OSError so a second attempt
    becomes a no-op instead of crashing the service.
    """
    global _METRICS_STARTED

    metrics_port = int(os.environ.get("METRICS_PORT", 8000))

    if not _METRICS_STARTED:
        try:
            start_http_server(metrics_port)
            _METRICS_STARTED = True
            logger.info(f"Prometheus metrics server started for {app_name} on port {metrics_port}")
        except OSError as e:
            # Likely already started by the parent reloader process; log & continue
            logger.debug(f"Metrics server already running on port {metrics_port}: {e}")
            _METRICS_STARTED = True  # Avoid retry storms

    @app.middleware("http")
    async def track_metrics(request: Request, call_next):
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # After the request is processed, record the metrics
        latency = time.time() - start_time
        path = request.url.path

        LATENCY.labels(method=request.method, path=path).observe(latency)
        REQUESTS.labels(method=request.method, path=path, status_code=response.status_code).inc()

        return response
