"""
Middleware patterns for Q2 Platform APIs.

Provides standardized middleware for error handling, logging, and correlation tracking.
"""

import time
import uuid
from typing import Callable, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from ..error_handling.exceptions import Q2Exception
from ..observability.enhanced_logging import get_enhanced_logger, set_correlation_id, set_request_id, get_correlation_id
from .responses import create_error_response_from_exception, get_http_status_for_exception


logger = get_enhanced_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracking."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation ID
        correlation_id = (
            request.headers.get("x-correlation-id") or request.headers.get("correlation-id") or str(uuid.uuid4())
        )

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Set in context
        set_correlation_id(correlation_id)
        set_request_id(request_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["x-correlation-id"] = correlation_id
        response.headers["x-request-id"] = request_id

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling across all endpoints."""

    def __init__(self, app, service_name: str = "unknown"):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_enhanced_logger(__name__, service_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return await self._handle_exception(request, e)

    async def _handle_exception(self, request: Request, exception: Exception) -> JSONResponse:
        """Handle exceptions and return standardized error responses."""

        # Log the exception with context
        self.logger.error(
            f"Unhandled exception in {request.method} {request.url.path}",
            exception=exception,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
        )

        # Create error response
        error_response = create_error_response_from_exception(exception)

        # Get appropriate HTTP status code
        status_code = get_http_status_for_exception(exception)

        return JSONResponse(
            content=error_response.dict(),
            status_code=status_code,
            headers={"x-correlation-id": get_correlation_id() or "unknown", "content-type": "application/json"},
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""

    def __init__(self, app, service_name: str = "unknown", log_bodies: bool = False):
        super().__init__(app)
        self.service_name = service_name
        self.log_bodies = log_bodies
        self.logger = get_enhanced_logger(__name__, service_name)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        await self._log_request(request)

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        await self._log_response(request, response, duration)

        return response

    async def _log_request(self, request: Request) -> None:
        """Log incoming request details."""
        request_body = None
        if self.log_bodies and request.method in ["POST", "PUT", "PATCH"]:
            try:
                # This is a simplified approach - in practice, you'd need to handle
                # different content types and streaming bodies more carefully
                request_body = await request.body()
                if request_body:
                    request_body = request_body.decode("utf-8")[:1000]  # Limit size
            except Exception:
                request_body = "<unable to read body>"

        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers),
            body=request_body,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

    async def _log_response(self, request: Request, response: Response, duration: float) -> None:
        """Log outgoing response details."""
        response_body = None
        if self.log_bodies and hasattr(response, "body"):
            try:
                if isinstance(response.body, bytes):
                    response_body = response.body.decode("utf-8")[:1000]  # Limit size
            except Exception:
                response_body = "<unable to read body>"

        self.logger.info(
            f"Outgoing response: {request.method} {request.url.path} - {response.status_code}",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response_body,
            duration_seconds=duration,
            duration_ms=duration * 1000,
        )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware (in-memory implementation)."""

    def __init__(self, app, requests_per_minute: int = 60, burst_requests: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_requests = burst_requests
        self.request_counts = {}  # In production, use Redis or similar
        self.logger = get_enhanced_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        if await self._is_rate_limited(client_ip):
            self.logger.warning(
                f"Rate limit exceeded for client {client_ip}",
                client_ip=client_ip,
                method=request.method,
                path=request.url.path,
            )

            from ..error_handling.exceptions import Q2RateLimitError

            error = Q2RateLimitError(
                f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                context={"client_ip": client_ip},
            )

            error_response = create_error_response_from_exception(error)
            return JSONResponse(
                content=error_response.dict(),
                status_code=429,
                headers={
                    "retry-after": "60",
                    "x-ratelimit-limit": str(self.requests_per_minute),
                    "x-ratelimit-remaining": "0",
                },
            )

        # Process request
        return await call_next(request)

    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        # This is a simplified in-memory implementation
        # In production, you'd use Redis with sliding window or token bucket
        current_time = time.time()

        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # Clean old requests (older than 1 minute)
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip] if current_time - req_time < 60
        ]

        # Check if limit exceeded
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return True

        # Add current request
        self.request_counts[client_ip].append(current_time)
        return False


# Convenience functions for setting up middleware


def setup_standard_middleware(
    app,
    service_name: str,
    enable_rate_limiting: bool = True,
    enable_request_logging: bool = True,
    requests_per_minute: int = 60,
):
    """
    Set up standard middleware stack for Q2 Platform services.

    Args:
        app: FastAPI or Starlette application
        service_name: Name of the service
        enable_rate_limiting: Whether to enable rate limiting
        enable_request_logging: Whether to enable request logging
        requests_per_minute: Rate limit threshold
    """

    # Add middleware in reverse order (they're applied as a stack)

    if enable_rate_limiting:
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=requests_per_minute)

    if enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware, service_name=service_name)

    app.add_middleware(ErrorHandlingMiddleware, service_name=service_name)

    app.add_middleware(CorrelationIdMiddleware)


# Health check middleware


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware to handle health check endpoints."""

    def __init__(self, app, health_check_path: str = "/health"):
        super().__init__(app)
        self.health_check_path = health_check_path

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path == self.health_check_path:
            return JSONResponse({"status": "healthy", "timestamp": time.time(), "service": "q2-platform"})

        return await call_next(request)
