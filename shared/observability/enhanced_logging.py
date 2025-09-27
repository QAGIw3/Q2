"""
Enhanced logging patterns for Q2 Platform.

Provides structured logging with error context, correlation IDs,
and integration with the error handling system.
"""

import contextvars
import functools
import logging
import uuid
from typing import Any, Dict, Optional, Callable
from datetime import datetime

import structlog

from ..error_handling.exceptions import Q2Exception, ErrorSeverity


# Context variables for correlation tracking
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)


class EnhancedLogger:
    """Enhanced logger with context awareness and error handling integration."""

    def __init__(self, name: str, service_name: Optional[str] = None):
        self.name = name
        self.service_name = service_name
        self._logger = structlog.get_logger(name)

    def _get_context(self, extra_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get logging context with correlation IDs and service info."""
        context = {
            "service": self.service_name,
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Remove None values
        context = {k: v for k, v in context.items() if v is not None}

        if extra_context:
            context.update(extra_context)

        return context

    def debug(self, message: str, **kwargs):
        """Log debug message with enhanced context."""
        context = self._get_context(kwargs)
        self._logger.debug(message, **context)

    def info(self, message: str, **kwargs):
        """Log info message with enhanced context."""
        context = self._get_context(kwargs)
        self._logger.info(message, **context)

    def warning(self, message: str, **kwargs):
        """Log warning message with enhanced context."""
        context = self._get_context(kwargs)
        self._logger.warning(message, **context)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with enhanced context and exception details."""
        context = self._get_context(kwargs)

        if exception:
            context["exception_type"] = type(exception).__name__
            context["exception_message"] = str(exception)

            if isinstance(exception, Q2Exception):
                context["error_id"] = exception.error_id
                context["error_code"] = exception.error_code
                context["error_category"] = exception.category.value
                context["error_severity"] = exception.severity.value
                context["error_context"] = exception.context
                context["error_suggestions"] = exception.suggestions

        self._logger.error(message, **context)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with enhanced context."""
        context = self._get_context(kwargs)

        if exception:
            context["exception_type"] = type(exception).__name__
            context["exception_message"] = str(exception)

            if isinstance(exception, Q2Exception):
                context["error_id"] = exception.error_id
                context["error_code"] = exception.error_code
                context["error_category"] = exception.category.value
                context["error_severity"] = exception.severity.value
                context["error_context"] = exception.context
                context["error_suggestions"] = exception.suggestions

        self._logger.critical(message, **context)

    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation."""
        context = self._get_context(kwargs)
        context["operation"] = operation
        context["operation_status"] = "started"
        self._logger.info(f"Operation '{operation}' started", **context)

    def log_operation_success(self, operation: str, duration: Optional[float] = None, **kwargs):
        """Log successful completion of an operation."""
        context = self._get_context(kwargs)
        context["operation"] = operation
        context["operation_status"] = "completed"
        if duration is not None:
            context["duration_seconds"] = duration
        self._logger.info(f"Operation '{operation}' completed successfully", **context)

    def log_operation_failure(self, operation: str, exception: Exception, duration: Optional[float] = None, **kwargs):
        """Log failure of an operation."""
        context = self._get_context(kwargs)
        context["operation"] = operation
        context["operation_status"] = "failed"
        if duration is not None:
            context["duration_seconds"] = duration

        self.error(f"Operation '{operation}' failed", exception=exception, **context)


def get_enhanced_logger(name: str, service_name: Optional[str] = None) -> EnhancedLogger:
    """Get an enhanced logger instance."""
    return EnhancedLogger(name, service_name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    """Set request ID for current context."""
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return request_id_var.get()


def set_user_id(user_id: str) -> None:
    """Set user ID for current context."""
    user_id_var.set(user_id)


def get_user_id() -> Optional[str]:
    """Get current user ID."""
    return user_id_var.get()


def with_correlation_id(correlation_id: Optional[str] = None) -> Callable:
    """
    Decorator to automatically set correlation ID for a function.

    Args:
        correlation_id: Specific correlation ID to use, or None to generate one
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate correlation ID if not provided
            corr_id = correlation_id or generate_correlation_id()

            # Set in context
            token = correlation_id_var.set(corr_id)

            try:
                return func(*args, **kwargs)
            finally:
                # Reset context
                correlation_id_var.reset(token)

        return wrapper

    return decorator


def with_async_correlation_id(correlation_id: Optional[str] = None) -> Callable:
    """
    Decorator to automatically set correlation ID for an async function.

    Args:
        correlation_id: Specific correlation ID to use, or None to generate one
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate correlation ID if not provided
            corr_id = correlation_id or generate_correlation_id()

            # Set in context
            token = correlation_id_var.set(corr_id)

            try:
                return await func(*args, **kwargs)
            finally:
                # Reset context
                correlation_id_var.reset(token)

        return wrapper

    return decorator


def log_exception_with_context(
    logger: EnhancedLogger, exception: Exception, operation: str, additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an exception with full context information.

    Args:
        logger: Enhanced logger instance
        exception: Exception to log
        operation: Operation that failed
        additional_context: Additional context to include
    """
    context = additional_context or {}

    # Determine log level based on exception severity
    if isinstance(exception, Q2Exception):
        severity = exception.severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical failure in {operation}", exception=exception, **context)
        elif severity in (ErrorSeverity.HIGH, ErrorSeverity.MEDIUM):
            logger.error(f"Error in {operation}", exception=exception, **context)
        else:
            logger.warning(f"Issue in {operation}", exception=exception, **context)
    else:
        logger.error(f"Unexpected error in {operation}", exception=exception, **context)
