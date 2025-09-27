"""
Circuit breaker pattern implementation for Q2 Platform.

Provides circuit breaker functionality to prevent cascading failures in
distributed systems by temporarily stopping calls to failing services.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar
from enum import Enum
from dataclasses import dataclass, field

from .exceptions import Q2Exception, Q2ExternalServiceError


logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service is recovered


class CircuitBreakerError(Q2Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            f"Circuit breaker is open for service '{service_name}'. Service temporarily unavailable.",
            error_code="CIRCUIT_BREAKER_OPEN",
            context={"service": service_name},
            suggestions=[
                "Wait for the circuit breaker to reset automatically",
                "Check service health and resolve underlying issues",
                "Consider implementing fallback mechanisms",
            ],
            **kwargs,
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to trip the breaker
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    success_threshold: int = 1  # Successes needed to close breaker from half-open
    timeout: Optional[float] = 30.0  # Call timeout
    expected_exception: type = Exception  # Exception type that counts as failure


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changed_time: float = field(default_factory=time.time)
    total_requests: int = 0
    failed_requests: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting external service calls.

    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests are blocked
    - HALF_OPEN: Limited requests allowed to test service recovery
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()

        logger.info(
            f"Circuit breaker '{name}' initialized", extra={"circuit_breaker": name, "config": self.config.__dict__}
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.metrics.last_failure_time is None:
            return True

        return (time.time() - self.metrics.last_failure_time) >= self.config.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.metrics.success_count += 1
            self.metrics.total_requests += 1
            self.metrics.last_success_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.success_count >= self.config.success_threshold:
                    self._close_breaker()

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                if self.metrics.failure_count >= self.config.failure_threshold:
                    self._open_breaker()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._open_breaker()

    def _open_breaker(self) -> None:
        """Open the circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        self.metrics.state_changed_time = time.time()
        self.metrics.failure_count = 0  # Reset for next cycle

        logger.warning(
            f"Circuit breaker '{self.name}' opened",
            extra={"circuit_breaker": self.name, "state": self.state.value, "metrics": self.metrics.__dict__},
        )

    def _close_breaker(self) -> None:
        """Close the circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.state_changed_time = time.time()
        self.metrics.failure_count = 0
        self.metrics.success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' closed",
            extra={"circuit_breaker": self.name, "state": self.state.value, "metrics": self.metrics.__dict__},
        )

    def _half_open_breaker(self) -> None:
        """Set circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.state_changed_time = time.time()
        self.metrics.success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' half-opened",
            extra={"circuit_breaker": self.name, "state": self.state.value, "metrics": self.metrics.__dict__},
        )

    def _can_execute(self) -> bool:
        """Check if a request can be executed."""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._half_open_breaker()
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True

        return False

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit breaker is open
            Original exception: If function fails and breaker allows it
        """
        if not self._can_execute():
            raise CircuitBreakerError(self.name)

        try:
            start_time = time.time()

            # Apply timeout if configured
            if self.config.timeout:
                # For sync functions, we can't easily apply timeout
                # This would require threading or signal handling
                pass

            result = func(*args, **kwargs)

            execution_time = time.time() - start_time
            self._record_success()

            logger.debug(
                f"Circuit breaker '{self.name}' call succeeded",
                extra={"circuit_breaker": self.name, "execution_time": execution_time, "state": self.state.value},
            )

            return result

        except Exception as e:
            # Only count specific exceptions as failures
            if isinstance(e, self.config.expected_exception):
                self._record_failure(e)

                logger.warning(
                    f"Circuit breaker '{self.name}' call failed",
                    extra={
                        "circuit_breaker": self.name,
                        "error": str(e),
                        "state": self.state.value,
                        "failure_count": self.metrics.failure_count,
                    },
                )

            raise

    async def async_call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit breaker is open
            Original exception: If function fails and breaker allows it
        """
        if not self._can_execute():
            raise CircuitBreakerError(self.name)

        try:
            start_time = time.time()

            # Apply timeout if configured
            if self.config.timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                result = await func(*args, **kwargs)

            execution_time = time.time() - start_time
            self._record_success()

            logger.debug(
                f"Circuit breaker '{self.name}' async call succeeded",
                extra={"circuit_breaker": self.name, "execution_time": execution_time, "state": self.state.value},
            )

            return result

        except Exception as e:
            # Only count specific exceptions as failures
            if isinstance(e, self.config.expected_exception):
                self._record_failure(e)

                logger.warning(
                    f"Circuit breaker '{self.name}' async call failed",
                    extra={
                        "circuit_breaker": self.name,
                        "error": str(e),
                        "state": self.state.value,
                        "failure_count": self.metrics.failure_count,
                    },
                )

            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._close_breaker()
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.metrics.failure_count,
                "success_count": self.metrics.success_count,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "state_changed_time": self.metrics.state_changed_time,
                "uptime": time.time() - self.metrics.state_changed_time,
                "config": self.config.__dict__,
            }


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.

    Args:
        name: Circuit breaker name
        config: Configuration (only used for new circuit breakers)

    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all registered circuit breakers."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all registered circuit breakers."""
    with _registry_lock:
        return {name: breaker.get_metrics() for name, breaker in _circuit_breakers.items()}
