"""
Q2 Platform shared error handling and retry mechanisms.
"""

from .exceptions import *
from .retry_patterns import *
from .circuit_breaker import *

__all__ = [
    # Exceptions
    "Q2Exception",
    "Q2ServiceError",
    "Q2ConfigurationError",
    "Q2ValidationError",
    "Q2NetworkError",
    "Q2TimeoutError",
    "Q2AuthenticationError",
    "Q2AuthorizationError",
    "Q2ResourceNotFoundError",
    "Q2ResourceConflictError",
    "Q2RateLimitError",
    "Q2ExternalServiceError",
    "ErrorSeverity",
    "ErrorCategory",
    "ConfigError",  # Legacy compatibility
    # Retry patterns
    "RetryStrategy",
    "ExponentialBackoffRetry",
    "LinearBackoffRetry",
    "FixedIntervalRetry",
    "with_retry",
    "async_with_retry",
    "BackoffStrategy",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerError",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    "get_all_circuit_breaker_metrics",
]
