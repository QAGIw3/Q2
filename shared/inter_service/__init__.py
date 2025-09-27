"""
Inter-service communication patterns for Q2 Platform.
"""

from .registry import *

__all__ = [
    # Service registry
    "ServiceRegistry",
    "ServiceInfo",
    "ServiceStatus",
    "register_service",
    "discover_service",
    "get_service_registry",
]
