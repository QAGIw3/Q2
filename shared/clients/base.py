"""
Base client classes for Q2 Platform services.

Provides standardized client patterns with error handling, retry logic,
circuit breaker support, and comprehensive monitoring.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TypeVar, Generic, List
from datetime import datetime

from ..error_handling import (
    Q2Exception,
    Q2ExternalServiceError,
    Q2TimeoutError,
    Q2NetworkError,
    ExponentialBackoffRetry,
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from ..observability.enhanced_logging import get_enhanced_logger, get_correlation_id


T = TypeVar("T")


@dataclass
class ClientConfig:
    """Base configuration for Q2 Platform clients."""

    service_name: str
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    enable_metrics: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClientMetrics:
    """Metrics for client operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: Optional[float] = None
    max_response_time: Optional[float] = None
    last_request_time: Optional[datetime] = None
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    def record_request(self, duration: float, success: bool, error_type: Optional[str] = None):
        """Record a request in metrics."""
        self.total_requests += 1
        self.last_request_time = datetime.utcnow()

        if success:
            self.successful_requests += 1
            self.total_response_time += duration

            if self.min_response_time is None or duration < self.min_response_time:
                self.min_response_time = duration
            if self.max_response_time is None or duration > self.max_response_time:
                self.max_response_time = duration
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "min_response_time": self.min_response_time,
            "max_response_time": self.max_response_time,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "error_counts": self.error_counts,
        }


class ClientResponse(Generic[T]):
    """Standardized client response wrapper."""

    def __init__(
        self,
        data: T,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        response_time: Optional[float] = None,
    ):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.request_id = request_id
        self.correlation_id = correlation_id or get_correlation_id()
        self.response_time = response_time

    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "data": self.data,
            "status_code": self.status_code,
            "headers": self.headers,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "response_time": self.response_time,
        }


class StreamingResponse:
    """Response wrapper for streaming data."""

    def __init__(self, stream, headers: Optional[Dict[str, str]] = None):
        self.stream = stream
        self.headers = headers or {}

    async def __aiter__(self):
        """Async iterator for streaming data."""
        async for chunk in self.stream:
            yield chunk

    async def collect(self) -> List[Any]:
        """Collect all streaming data into a list."""
        data = []
        async for chunk in self.stream:
            data.append(chunk)
        return data


class BaseQ2Client(ABC):
    """
    Base client class for all Q2 Platform services.

    Provides standardized patterns for:
    - Error handling and exceptions
    - Retry logic with exponential backoff
    - Circuit breaker support
    - Request/response logging
    - Metrics collection
    - Correlation ID tracking
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = get_enhanced_logger(f"{config.service_name}_client", config.service_name)
        self.metrics = ClientMetrics()

        # Set up retry strategy
        self.retry_strategy = ExponentialBackoffRetry(
            max_attempts=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            multiplier=config.retry_backoff_multiplier,
        )

        # Set up circuit breaker
        if config.enable_circuit_breaker:
            breaker_config = CircuitBreakerConfig(
                failure_threshold=config.circuit_breaker_failure_threshold,
                recovery_timeout=config.circuit_breaker_recovery_timeout,
                timeout=config.timeout,
            )
            self.circuit_breaker = get_circuit_breaker(config.service_name, breaker_config)
        else:
            self.circuit_breaker = None

        self.logger.info(
            f"Client initialized for {config.service_name}",
            base_url=config.base_url,
            timeout=config.timeout,
            circuit_breaker_enabled=config.enable_circuit_breaker,
        )

    @abstractmethod
    async def _make_request(
        self, method: str, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """
        Make a request to the service.

        This method must be implemented by subclasses to handle the actual
        communication protocol (HTTP, gRPC, etc.).
        """
        pass

    async def request(
        self, method: str, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """
        Make a request with comprehensive error handling and monitoring.

        Args:
            method: HTTP method or operation type
            path: Endpoint path or operation identifier
            data: Request data
            params: Query parameters or additional options
            **kwargs: Additional arguments for the request

        Returns:
            ClientResponse with the result

        Raises:
            Q2Exception: Various Q2 exceptions based on failure type
        """
        start_time = time.time()

        try:
            # Use circuit breaker if enabled
            if self.circuit_breaker:
                response = await self.circuit_breaker.async_call(
                    self._make_request_with_retry, method, path, data, params, **kwargs
                )
            else:
                response = await self._make_request_with_retry(method, path, data, params, **kwargs)

            # Record success metrics
            duration = time.time() - start_time
            self.metrics.record_request(duration, True)

            self.logger.debug(
                f"Request completed successfully: {method} {path}",
                method=method,
                path=path,
                status_code=response.status_code,
                response_time=duration,
            )

            return response

        except Q2Exception as e:
            # Q2 exceptions are already well-formed
            duration = time.time() - start_time
            self.metrics.record_request(duration, False, type(e).__name__)

            self.logger.error(
                f"Request failed: {method} {path}",
                exception=e,
                method=method,
                path=path,
                response_time=duration,
            )

            raise

        except Exception as e:
            # Wrap unexpected exceptions
            duration = time.time() - start_time
            self.metrics.record_request(duration, False, type(e).__name__)

            wrapped_exception = Q2ExternalServiceError(
                f"Unexpected error calling {self.config.service_name}",
                context={
                    "service": self.config.service_name,
                    "method": method,
                    "path": path,
                    "original_error": str(e),
                },
                cause=e,
            )

            self.logger.error(
                f"Unexpected error in request: {method} {path}",
                exception=wrapped_exception,
                method=method,
                path=path,
                response_time=duration,
            )

            raise wrapped_exception

    async def _make_request_with_retry(
        self, method: str, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """Make request with retry logic."""
        last_exception = None

        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            try:
                self.logger.debug(
                    f"Making request attempt {attempt}: {method} {path}",
                    method=method,
                    path=path,
                    attempt=attempt,
                )

                response = await self._make_request(method, path, data, params, **kwargs)

                if attempt > 1:
                    self.logger.info(
                        f"Request succeeded on attempt {attempt}: {method} {path}",
                        method=method,
                        path=path,
                        attempt=attempt,
                    )

                return response

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.retry_strategy.should_retry(e, attempt):
                    break

                # Calculate delay
                delay = self.retry_strategy.get_delay(attempt)

                self.logger.warning(
                    f"Request failed on attempt {attempt}, retrying in {delay:.2f}s: {method} {path}",
                    method=method,
                    path=path,
                    attempt=attempt,
                    delay=delay,
                    error=str(e),
                )

                # Wait before retry
                if attempt < self.retry_strategy.max_attempts:
                    import asyncio

                    await asyncio.sleep(delay)

        # All retries exhausted, raise the last exception
        if isinstance(last_exception, Q2Exception):
            raise last_exception
        else:
            raise Q2ExternalServiceError(
                f"Request failed after {self.retry_strategy.max_attempts} attempts",
                context={
                    "service": self.config.service_name,
                    "method": method,
                    "path": path,
                    "attempts": self.retry_strategy.max_attempts,
                    "original_error": str(last_exception),
                },
                cause=last_exception,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        base_metrics = self.metrics.to_dict()

        if self.circuit_breaker:
            base_metrics["circuit_breaker"] = self.circuit_breaker.get_metrics()

        return base_metrics

    def reset_metrics(self) -> None:
        """Reset client metrics."""
        self.metrics = ClientMetrics()

    async def health_check(self) -> bool:
        """
        Perform a health check against the service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Default implementation - subclasses should override with specific health check
            await self.request("GET", "/health")
            return True
        except Exception as e:
            self.logger.warning(f"Health check failed for {self.config.service_name}", exception=e)
            return False

    async def close(self) -> None:
        """Clean up client resources."""
        self.logger.info(f"Closing client for {self.config.service_name}")
        # Subclasses should override to clean up specific resources

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
