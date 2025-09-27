"""
Retry patterns and mechanisms for Q2 Platform.

This module provides configurable retry strategies with exponential backoff,
jitter, and timeout support for improving resilience in distributed systems.
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Type, Union, TypeVar
from enum import Enum

from .exceptions import Q2Exception, Q2TimeoutError, Q2NetworkError, Q2ExternalServiceError


logger = logging.getLogger(__name__)

# Type variable for generic retry decorators
T = TypeVar("T")


class BackoffStrategy(Enum):
    """Backoff strategy options."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: Optional[float] = None,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.retryable_exceptions = retryable_exceptions or [
            Q2NetworkError,
            Q2ExternalServiceError,
            Q2TimeoutError,
            ConnectionError,
            TimeoutError,
        ]
        self.jitter = jitter

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        pass

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False

        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)

    def add_jitter(self, delay: float) -> float:
        """Add jitter to delay to prevent thundering herd."""
        if not self.jitter:
            return delay

        # Add up to 10% jitter
        jitter_amount = delay * 0.1 * random.random()
        return delay + jitter_amount


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(self, multiplier: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (self.multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)
        return self.add_jitter(delay)


class LinearBackoffRetry(RetryStrategy):
    """Linear backoff retry strategy."""

    def get_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        delay = self.base_delay * attempt
        delay = min(delay, self.max_delay)
        return self.add_jitter(delay)


class FixedIntervalRetry(RetryStrategy):
    """Fixed interval retry strategy."""

    def get_delay(self, attempt: int) -> float:
        """Return fixed delay."""
        return self.add_jitter(self.base_delay)


def with_retry(
    strategy: Optional[RetryStrategy] = None,
    log_failures: bool = True,
    correlation_id: Optional[str] = None,
) -> Callable:
    """
    Decorator for adding retry logic to synchronous functions.

    Args:
        strategy: Retry strategy to use (defaults to ExponentialBackoffRetry)
        log_failures: Whether to log retry attempts
        correlation_id: Correlation ID for tracking requests
    """
    if strategy is None:
        strategy = ExponentialBackoffRetry()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            last_exception = None

            for attempt in range(1, strategy.max_attempts + 1):
                try:
                    # Check timeout
                    if strategy.timeout and (time.time() - start_time) > strategy.timeout:
                        raise Q2TimeoutError(
                            f"Operation timed out after {strategy.timeout}s",
                            context={"function": func.__name__, "attempt": attempt},
                            correlation_id=correlation_id,
                        )

                    result = func(*args, **kwargs)

                    # Log successful retry if we had failures
                    if attempt > 1 and log_failures:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "correlation_id": correlation_id,
                            },
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not strategy.should_retry(e, attempt):
                        break

                    # Calculate delay
                    delay = strategy.get_delay(attempt)

                    # Log retry attempt
                    if log_failures:
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "delay": delay,
                                "error": str(e),
                                "correlation_id": correlation_id,
                            },
                        )

                    # Wait before retry
                    if attempt < strategy.max_attempts:
                        time.sleep(delay)

            # All retries exhausted
            if log_failures:
                logger.error(
                    f"Function {func.__name__} failed after {strategy.max_attempts} attempts",
                    extra={
                        "function": func.__name__,
                        "max_attempts": strategy.max_attempts,
                        "error": str(last_exception),
                        "correlation_id": correlation_id,
                    },
                )

            # Re-raise the last exception
            if isinstance(last_exception, Q2Exception):
                raise last_exception
            else:
                # Wrap in Q2Exception for consistency
                raise Q2ExternalServiceError(
                    f"Function {func.__name__} failed after {strategy.max_attempts} attempts",
                    context={
                        "function": func.__name__,
                        "attempts": strategy.max_attempts,
                        "original_error": str(last_exception),
                    },
                    cause=last_exception,
                    correlation_id=correlation_id,
                )

        return wrapper

    return decorator


def async_with_retry(
    strategy: Optional[RetryStrategy] = None,
    log_failures: bool = True,
    correlation_id: Optional[str] = None,
) -> Callable:
    """
    Decorator for adding retry logic to asynchronous functions.

    Args:
        strategy: Retry strategy to use (defaults to ExponentialBackoffRetry)
        log_failures: Whether to log retry attempts
        correlation_id: Correlation ID for tracking requests
    """
    if strategy is None:
        strategy = ExponentialBackoffRetry()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            last_exception = None

            for attempt in range(1, strategy.max_attempts + 1):
                try:
                    # Check timeout
                    if strategy.timeout and (time.time() - start_time) > strategy.timeout:
                        raise Q2TimeoutError(
                            f"Operation timed out after {strategy.timeout}s",
                            context={"function": func.__name__, "attempt": attempt},
                            correlation_id=correlation_id,
                        )

                    result = await func(*args, **kwargs)

                    # Log successful retry if we had failures
                    if attempt > 1 and log_failures:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "correlation_id": correlation_id,
                            },
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not strategy.should_retry(e, attempt):
                        break

                    # Calculate delay
                    delay = strategy.get_delay(attempt)

                    # Log retry attempt
                    if log_failures:
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "delay": delay,
                                "error": str(e),
                                "correlation_id": correlation_id,
                            },
                        )

                    # Wait before retry
                    if attempt < strategy.max_attempts:
                        await asyncio.sleep(delay)

            # All retries exhausted
            if log_failures:
                logger.error(
                    f"Function {func.__name__} failed after {strategy.max_attempts} attempts",
                    extra={
                        "function": func.__name__,
                        "max_attempts": strategy.max_attempts,
                        "error": str(last_exception),
                        "correlation_id": correlation_id,
                    },
                )

            # Re-raise the last exception
            if isinstance(last_exception, Q2Exception):
                raise last_exception
            else:
                # Wrap in Q2Exception for consistency
                raise Q2ExternalServiceError(
                    f"Function {func.__name__} failed after {strategy.max_attempts} attempts",
                    context={
                        "function": func.__name__,
                        "attempts": strategy.max_attempts,
                        "original_error": str(last_exception),
                    },
                    cause=last_exception,
                    correlation_id=correlation_id,
                )

        return wrapper

    return decorator
