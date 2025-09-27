"""
Q2 Platform standardized exception hierarchy.

This module provides a comprehensive set of exception classes for consistent
error handling across all Q2 Platform services. Each exception includes
structured error information for better debugging and monitoring.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification"""

    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"


class Q2Exception(Exception):
    """
    Base exception class for all Q2 Platform errors.

    Provides structured error information including:
    - Unique error ID for tracing
    - Error category and severity
    - Contextual information for debugging
    - Support for error chaining
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
        service_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message)

        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.suggestions = suggestions or []
        self.cause = cause
        self.service_name = service_name
        self.correlation_id = correlation_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "suggestions": self.suggestions,
            "service_name": self.service_name,
            "correlation_id": self.correlation_id,
            "cause": str(self.cause) if self.cause else None,
            "type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg


class Q2ServiceError(Q2Exception):
    """Base class for service-level errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.SYSTEM, severity=ErrorSeverity.HIGH, **kwargs)


class Q2ConfigurationError(Q2Exception):
    """Raised when there are configuration-related issues."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, severity=ErrorSeverity.HIGH, **kwargs)


class Q2ValidationError(Q2Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, severity=ErrorSeverity.MEDIUM, **kwargs)


class Q2NetworkError(Q2Exception):
    """Raised when network-related issues occur."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, severity=ErrorSeverity.HIGH, **kwargs)


class Q2TimeoutError(Q2Exception):
    """Raised when operations timeout."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, severity=ErrorSeverity.HIGH, **kwargs)


class Q2AuthenticationError(Q2Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHENTICATION, severity=ErrorSeverity.HIGH, **kwargs)


class Q2AuthorizationError(Q2Exception):
    """Raised when authorization fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHORIZATION, severity=ErrorSeverity.HIGH, **kwargs)


class Q2ResourceNotFoundError(Q2Exception):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, severity=ErrorSeverity.MEDIUM, **kwargs)


class Q2ResourceConflictError(Q2Exception):
    """Raised when there's a conflict with an existing resource."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, severity=ErrorSeverity.MEDIUM, **kwargs)


class Q2RateLimitError(Q2Exception):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_SERVICE, severity=ErrorSeverity.MEDIUM, **kwargs)


class Q2ExternalServiceError(Q2Exception):
    """Raised when external service calls fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_SERVICE, severity=ErrorSeverity.HIGH, **kwargs)


# Legacy compatibility - map old ConfigError to new system
ConfigError = Q2ConfigurationError
