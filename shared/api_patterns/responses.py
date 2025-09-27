"""
Standardized API response patterns for Q2 Platform.

Provides consistent response structures across all services.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from pydantic import BaseModel, Field

from ..error_handling.exceptions import Q2Exception, ErrorSeverity
from ..observability.enhanced_logging import get_correlation_id


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Base API response model."""

    success: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(default_factory=get_correlation_id)
    data: Optional[T] = None
    errors: Optional[List[Dict[str, Any]]] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SuccessResponse(APIResponse[T]):
    """Success response model."""

    success: bool = True
    message: Optional[str] = None


class ErrorResponse(APIResponse[None]):
    """Error response model."""

    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


class ValidationErrorResponse(ErrorResponse):
    """Validation error response model."""

    error_code: str = "VALIDATION_ERROR"
    field_errors: Optional[Dict[str, List[str]]] = None


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class PaginatedResponse(SuccessResponse[List[T]]):
    """Paginated response model."""

    pagination: PaginationMeta


# Response creation helpers


def create_success_response(
    data: T, message: Optional[str] = None, correlation_id: Optional[str] = None
) -> SuccessResponse[T]:
    """Create a success response."""
    return SuccessResponse(data=data, message=message, correlation_id=correlation_id or get_correlation_id())


def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
) -> ErrorResponse:
    """Create an error response."""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        suggestions=suggestions,
        correlation_id=correlation_id or get_correlation_id(),
    )


def create_error_response_from_exception(exception: Exception) -> ErrorResponse:
    """Create an error response from an exception."""
    if isinstance(exception, Q2Exception):
        return ErrorResponse(
            error_code=exception.error_code,
            message=exception.message,
            details=exception.context,
            suggestions=exception.suggestions,
            correlation_id=exception.correlation_id or get_correlation_id(),
        )
    else:
        return ErrorResponse(error_code="INTERNAL_ERROR", message=str(exception), correlation_id=get_correlation_id())


def create_validation_error_response(
    message: str = "Validation failed",
    field_errors: Optional[Dict[str, List[str]]] = None,
    correlation_id: Optional[str] = None,
) -> ValidationErrorResponse:
    """Create a validation error response."""
    return ValidationErrorResponse(
        message=message, field_errors=field_errors, correlation_id=correlation_id or get_correlation_id()
    )


def create_paginated_response(
    data: List[T],
    page: int,
    page_size: int,
    total_items: int,
    message: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> PaginatedResponse[T]:
    """Create a paginated response."""
    total_pages = (total_items + page_size - 1) // page_size

    pagination = PaginationMeta(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1,
    )

    return PaginatedResponse(
        data=data, pagination=pagination, message=message, correlation_id=correlation_id or get_correlation_id()
    )


# HTTP status code mappings for different error types


def get_http_status_for_exception(exception: Exception) -> int:
    """Get appropriate HTTP status code for an exception."""
    if isinstance(exception, Q2Exception):
        from ..error_handling.exceptions import (
            Q2ValidationError,
            Q2AuthenticationError,
            Q2AuthorizationError,
            Q2ResourceNotFoundError,
            Q2ResourceConflictError,
            Q2RateLimitError,
            Q2TimeoutError,
        )

        if isinstance(exception, Q2ValidationError):
            return 400  # Bad Request
        elif isinstance(exception, Q2AuthenticationError):
            return 401  # Unauthorized
        elif isinstance(exception, Q2AuthorizationError):
            return 403  # Forbidden
        elif isinstance(exception, Q2ResourceNotFoundError):
            return 404  # Not Found
        elif isinstance(exception, Q2ResourceConflictError):
            return 409  # Conflict
        elif isinstance(exception, Q2RateLimitError):
            return 429  # Too Many Requests
        elif isinstance(exception, Q2TimeoutError):
            return 408  # Request Timeout
        elif exception.severity == ErrorSeverity.CRITICAL:
            return 503  # Service Unavailable
        else:
            return 500  # Internal Server Error
    else:
        return 500  # Internal Server Error
