"""
Observability utilities for Q2 Platform.
"""

import logging
import structlog
from opentelemetry import trace

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return structlog.get_logger(name)

def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)

# Re-export for compatibility
__all__ = ["get_logger", "get_tracer"]