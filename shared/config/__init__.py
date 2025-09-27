"""
Shared configuration management for Q2 Platform.

This module provides centralized configuration management across all services.
"""

from .base import BaseConfig, ConfigError

# Import loader and validator only when explicitly needed to avoid dependency issues
def get_config_loader():
    """Get ConfigLoader class (lazy import to avoid dependency issues)."""
    from .loader import ConfigLoader
    return ConfigLoader

def get_config_validator():
    """Get ConfigValidator class (lazy import to avoid dependency issues)."""
    from .validator import ConfigValidator
    return ConfigValidator

__all__ = ["BaseConfig", "ConfigError", "get_config_loader", "get_config_validator"]