"""
Request validation patterns for Q2 Platform APIs.

Provides standardized validation decorators and utilities.
"""

import functools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

from ..error_handling.exceptions import Q2ValidationError
from ..observability.enhanced_logging import get_enhanced_logger


logger = get_enhanced_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class RequestValidator:
    """Request validator with comprehensive error handling."""

    def __init__(self, model_class: Type[T]):
        self.model_class = model_class

    def validate(self, data: Dict[str, Any]) -> T:
        """
        Validate request data against the model.

        Args:
            data: Request data to validate

        Returns:
            Validated model instance

        Raises:
            Q2ValidationError: If validation fails
        """
        try:
            return self.model_class(**data)
        except ValidationError as e:
            # Convert Pydantic validation errors to our format
            field_errors = self._format_validation_errors(e.errors())

            raise Q2ValidationError(
                "Request validation failed",
                context={
                    "model": self.model_class.__name__,
                    "field_errors": field_errors,
                    "error_count": len(e.errors()),
                },
                suggestions=[
                    "Check the request payload format",
                    "Ensure all required fields are provided",
                    "Verify field types and constraints",
                ],
            )

    def _format_validation_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Format Pydantic validation errors into field-based structure."""
        field_errors = {}

        for error in errors:
            field_path = ".".join(str(loc) for loc in error["loc"])
            error_msg = error["msg"]
            error_type = error["type"]

            # Create a more user-friendly error message
            if error_type == "missing":
                error_msg = "This field is required"
            elif error_type == "value_error.missing":
                error_msg = "This field is required"
            elif error_type.startswith("type_error"):
                expected_type = error_type.split(".")[-1]
                error_msg = f"Expected {expected_type}, got {error.get('input_type', 'unknown')}"

            if field_path not in field_errors:
                field_errors[field_path] = []

            field_errors[field_path].append(error_msg)

        return field_errors


def validate_request(model_class: Type[T], data_source: str = "json") -> Callable:
    """
    Decorator for automatic request validation.

    Args:
        model_class: Pydantic model class for validation
        data_source: Source of data ("json", "query", "form")

    Returns:
        Decorated function
    """
    validator = RequestValidator(model_class)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract request data based on framework
            request_data = await _extract_request_data(args, kwargs, data_source)

            # Validate the data
            validated_data = validator.validate(request_data)

            # Inject validated data into kwargs
            kwargs["validated_data"] = validated_data

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract request data based on framework
            request_data = _extract_request_data_sync(args, kwargs, data_source)

            # Validate the data
            validated_data = validator.validate(request_data)

            # Inject validated data into kwargs
            kwargs["validated_data"] = validated_data

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _extract_request_data(args, kwargs, data_source: str) -> Dict[str, Any]:
    """Extract request data from FastAPI request."""
    # This is a simplified version - in practice, you'd need to handle
    # different web frameworks (FastAPI, Flask, etc.)

    # For FastAPI, typically the request object is passed as an argument
    for arg in args:
        if hasattr(arg, "json") and data_source == "json":
            return await arg.json()
        elif hasattr(arg, "query_params") and data_source == "query":
            return dict(arg.query_params)
        elif hasattr(arg, "form") and data_source == "form":
            return await arg.form()

    # Fallback to kwargs
    return kwargs.get("data", {})


def _extract_request_data_sync(args, kwargs, data_source: str) -> Dict[str, Any]:
    """Extract request data for synchronous functions."""
    # This would need to be adapted based on the web framework
    return kwargs.get("data", {})


# Validation utilities


def validate_pagination_params(
    page: Optional[int] = None, page_size: Optional[int] = None, max_page_size: int = 100
) -> Dict[str, int]:
    """
    Validate and normalize pagination parameters.

    Args:
        page: Page number (1-based)
        page_size: Items per page
        max_page_size: Maximum allowed page size

    Returns:
        Normalized pagination parameters

    Raises:
        Q2ValidationError: If parameters are invalid
    """
    errors = []

    # Validate page
    if page is None:
        page = 1
    elif page < 1:
        errors.append("Page number must be greater than 0")

    # Validate page_size
    if page_size is None:
        page_size = 20  # Default page size
    elif page_size < 1:
        errors.append("Page size must be greater than 0")
    elif page_size > max_page_size:
        errors.append(f"Page size cannot exceed {max_page_size}")

    if errors:
        raise Q2ValidationError(
            "Invalid pagination parameters",
            context={"errors": errors},
            suggestions=["Use valid page and page_size values"],
        )

    return {"page": page, "page_size": page_size}


def validate_sort_params(
    sort_by: Optional[str] = None, sort_order: Optional[str] = None, allowed_fields: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Validate and normalize sorting parameters.

    Args:
        sort_by: Field to sort by
        sort_order: Sort order ('asc' or 'desc')
        allowed_fields: List of allowed sort fields

    Returns:
        Normalized sort parameters

    Raises:
        Q2ValidationError: If parameters are invalid
    """
    errors = []

    # Validate sort_by
    if sort_by and allowed_fields and sort_by not in allowed_fields:
        errors.append(f"Sort field '{sort_by}' is not allowed. Allowed fields: {allowed_fields}")

    # Validate sort_order
    if sort_order is None:
        sort_order = "asc"
    elif sort_order.lower() not in ["asc", "desc"]:
        errors.append("Sort order must be 'asc' or 'desc'")
    else:
        sort_order = sort_order.lower()

    if errors:
        raise Q2ValidationError(
            "Invalid sort parameters",
            context={"errors": errors},
            suggestions=["Use valid sort_by and sort_order values"],
        )

    return {"sort_by": sort_by, "sort_order": sort_order}


def validate_date_range(
    start_date: Optional[str] = None, end_date: Optional[str] = None, date_format: str = "%Y-%m-%d"
) -> Dict[str, Optional[str]]:
    """
    Validate date range parameters.

    Args:
        start_date: Start date string
        end_date: End date string
        date_format: Expected date format

    Returns:
        Validated date parameters

    Raises:
        Q2ValidationError: If dates are invalid
    """
    from datetime import datetime

    errors = []
    parsed_start = None
    parsed_end = None

    # Validate start_date
    if start_date:
        try:
            parsed_start = datetime.strptime(start_date, date_format)
        except ValueError:
            errors.append(f"Invalid start_date format. Expected: {date_format}")

    # Validate end_date
    if end_date:
        try:
            parsed_end = datetime.strptime(end_date, date_format)
        except ValueError:
            errors.append(f"Invalid end_date format. Expected: {date_format}")

    # Validate date range
    if parsed_start and parsed_end and parsed_start > parsed_end:
        errors.append("start_date must be before or equal to end_date")

    if errors:
        raise Q2ValidationError(
            "Invalid date range parameters",
            context={"errors": errors},
            suggestions=[f"Use date format: {date_format}", "Ensure start_date <= end_date"],
        )

    return {"start_date": start_date, "end_date": end_date}


# Import asyncio here to avoid circular import issues
import asyncio
