"""
Standardized API patterns for Q2 Platform.
"""

from .responses import *
from .validation import *
from .middleware import *

__all__ = [
    # Response patterns
    "APIResponse",
    "SuccessResponse",
    "ErrorResponse",
    "ValidationErrorResponse",
    "PaginatedResponse",
    "create_success_response",
    "create_error_response",
    "create_validation_error_response",
    "create_paginated_response",
    # Validation patterns
    "validate_request",
    "RequestValidator",
    # Middleware
    "error_handling_middleware",
    "correlation_id_middleware",
    "request_logging_middleware",
]
