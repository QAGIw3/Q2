"""
Connection pool managers for Q2 Platform performance optimization.

Provides pooled connections for:
- Apache Pulsar (message brokers)
- Apache Ignite (in-memory data grid)  
- HTTP clients
"""

from .pulsar_pool import PulsarConnectionPool

__all__ = [
    'PulsarConnectionPool'
]
