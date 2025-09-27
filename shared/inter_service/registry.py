"""
Service registry for Q2 Platform inter-service communication.

Provides centralized service discovery and health monitoring.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import threading

from ..error_handling import Q2Exception, Q2ResourceNotFoundError, Q2ExternalServiceError
from ..observability.enhanced_logging import get_enhanced_logger


logger = get_enhanced_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInfo:
    """Information about a registered service."""

    name: str
    base_url: str
    version: str = "1.0.0"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Health check configuration
    health_check_endpoint: str = "/health"
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0

    # Registration tracking
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0

    # Service capabilities
    capabilities: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)

    def is_healthy(self) -> bool:
        """Check if service is considered healthy."""
        return self.status == ServiceStatus.HEALTHY

    def is_stale(self, max_age: float = 300.0) -> bool:
        """Check if service registration is stale."""
        if not self.last_heartbeat:
            age = (datetime.utcnow() - self.registered_at).total_seconds()
        else:
            age = (datetime.utcnow() - self.last_heartbeat).total_seconds()

        return age > max_age

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "version": self.version,
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "health_check_endpoint": self.health_check_endpoint,
            "health_check_interval": self.health_check_interval,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_check_failures": self.health_check_failures,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
        }


class ServiceRegistry:
    """
    Centralized service registry for Q2 Platform.

    Provides:
    - Service registration and discovery
    - Health monitoring
    - Load balancing support
    - Service dependency tracking
    """

    def __init__(self, cleanup_interval: float = 60.0, stale_threshold: float = 300.0):
        self.services: Dict[str, ServiceInfo] = {}
        self.cleanup_interval = cleanup_interval
        self.stale_threshold = stale_threshold
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._event_handlers: List[Callable[[str, ServiceInfo, str], None]] = []

        logger.info("Service registry initialized")

    def register_service(
        self,
        name: str,
        base_url: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        capabilities: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        enable_health_checks: bool = False,
        **kwargs,
    ) -> ServiceInfo:
        """
        Register a service in the registry.

        Args:
            name: Service name (must be unique)
            base_url: Base URL for the service
            version: Service version
            metadata: Additional service metadata
            tags: Service tags for categorization
            capabilities: Service capabilities/features
            dependencies: Services this service depends on
            enable_health_checks: Whether to start health check tasks (requires async context)
            **kwargs: Additional ServiceInfo parameters

        Returns:
            ServiceInfo object for the registered service

        Raises:
            Q2Exception: If registration fails
        """
        with self._lock:
            service_info = ServiceInfo(
                name=name,
                base_url=base_url,
                version=version,
                metadata=metadata or {},
                tags=tags or set(),
                capabilities=capabilities or set(),
                dependencies=dependencies or set(),
                **kwargs,
            )

            # Check for existing service
            existing = self.services.get(name)
            if existing:
                logger.info(
                    f"Updating existing service registration: {name}",
                    old_url=existing.base_url,
                    new_url=base_url,
                )
                # Cancel existing health check task
                if name in self._health_check_tasks:
                    self._health_check_tasks[name].cancel()
            else:
                logger.info(f"Registering new service: {name}", base_url=base_url)

            self.services[name] = service_info

            # Start health check task only if requested and in async context
            if enable_health_checks:
                try:
                    self._start_health_check_task(service_info)
                except RuntimeError:
                    logger.warning(f"Cannot start health checks for {name}: no async context")

            # Notify event handlers
            self._notify_event_handlers(name, service_info, "registered")

            return service_info

    def deregister_service(self, name: str) -> bool:
        """
        Deregister a service from the registry.

        Args:
            name: Service name to deregister

        Returns:
            True if service was deregistered, False if not found
        """
        with self._lock:
            service_info = self.services.pop(name, None)

            if service_info:
                logger.info(f"Deregistering service: {name}")

                # Cancel health check task
                if name in self._health_check_tasks:
                    self._health_check_tasks[name].cancel()
                    del self._health_check_tasks[name]

                # Notify event handlers
                self._notify_event_handlers(name, service_info, "deregistered")

                return True

            return False

    def discover_service(self, name: str) -> Optional[ServiceInfo]:
        """
        Discover a service by name.

        Args:
            name: Service name to discover

        Returns:
            ServiceInfo if found, None otherwise
        """
        with self._lock:
            return self.services.get(name)

    def discover_services(
        self,
        tags: Optional[Set[str]] = None,
        capabilities: Optional[Set[str]] = None,
        status: Optional[ServiceStatus] = None,
        healthy_only: bool = False,
    ) -> List[ServiceInfo]:
        """
        Discover services by criteria.

        Args:
            tags: Required tags (all must match)
            capabilities: Required capabilities (all must match)
            status: Required status
            healthy_only: Only return healthy services

        Returns:
            List of matching ServiceInfo objects
        """
        with self._lock:
            services = list(self.services.values())

        # Apply filters
        if tags:
            services = [s for s in services if tags.issubset(s.tags)]

        if capabilities:
            services = [s for s in services if capabilities.issubset(s.capabilities)]

        if status:
            services = [s for s in services if s.status == status]

        if healthy_only:
            services = [s for s in services if s.is_healthy()]

        return services

    def get_service_url(self, name: str) -> str:
        """
        Get service URL by name.

        Args:
            name: Service name

        Returns:
            Service base URL

        Raises:
            Q2ResourceNotFoundError: If service not found
        """
        service = self.discover_service(name)

        if not service:
            raise Q2ResourceNotFoundError(
                f"Service '{name}' not found in registry",
                context={"service_name": name},
                suggestions=["Check service name", "Ensure service is registered"],
            )

        if not service.is_healthy():
            logger.warning(f"Returning URL for unhealthy service: {name}")

        return service.base_url

    def heartbeat(self, name: str) -> bool:
        """
        Record a heartbeat for a service.

        Args:
            name: Service name

        Returns:
            True if heartbeat recorded, False if service not found
        """
        with self._lock:
            service = self.services.get(name)

            if service:
                service.last_heartbeat = datetime.utcnow()
                logger.debug(f"Heartbeat recorded for service: {name}")
                return True

            return False

    def update_service_status(self, name: str, status: ServiceStatus) -> bool:
        """
        Update service status.

        Args:
            name: Service name
            status: New status

        Returns:
            True if status updated, False if service not found
        """
        with self._lock:
            service = self.services.get(name)

            if service:
                old_status = service.status
                service.status = status

                logger.info(f"Service status updated: {name}", old_status=old_status.value, new_status=status.value)

                # Notify event handlers
                self._notify_event_handlers(name, service, f"status_changed:{status.value}")

                return True

            return False

    def get_all_services(self) -> Dict[str, ServiceInfo]:
        """Get all registered services."""
        with self._lock:
            return self.services.copy()

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            services = list(self.services.values())

            status_counts = {}
            for status in ServiceStatus:
                status_counts[status.value] = sum(1 for s in services if s.status == status)

            return {
                "total_services": len(services),
                "healthy_services": sum(1 for s in services if s.is_healthy()),
                "stale_services": sum(1 for s in services if s.is_stale(self.stale_threshold)),
                "status_counts": status_counts,
                "active_health_checks": len(self._health_check_tasks),
            }

    def add_event_handler(self, handler: Callable[[str, ServiceInfo, str], None]) -> None:
        """
        Add event handler for service events.

        Args:
            handler: Function called with (service_name, service_info, event_type)
        """
        self._event_handlers.append(handler)

    def _notify_event_handlers(self, name: str, service_info: ServiceInfo, event_type: str) -> None:
        """Notify all event handlers of a service event."""
        for handler in self._event_handlers:
            try:
                handler(name, service_info, event_type)
            except Exception as e:
                logger.error(f"Error in service event handler: {e}", exception=e)

    def _start_health_check_task(self, service_info: ServiceInfo) -> None:
        """Start health check task for a service."""

        async def health_check_loop():
            from ..clients.http import create_http_client

            client = None
            try:
                client = create_http_client(
                    service_name=service_info.name,
                    base_url=service_info.base_url,
                    timeout=service_info.health_check_timeout,
                )

                while True:
                    try:
                        # Perform health check
                        response = await client.get(service_info.health_check_endpoint)

                        with self._lock:
                            service_info.last_health_check = datetime.utcnow()
                            service_info.health_check_failures = 0

                            new_status = ServiceStatus.HEALTHY if response.is_success else ServiceStatus.UNHEALTHY

                            if service_info.status != new_status:
                                self.update_service_status(service_info.name, new_status)

                        logger.debug(f"Health check passed for service: {service_info.name}")

                    except Exception as e:
                        with self._lock:
                            service_info.health_check_failures += 1
                            service_info.last_health_check = datetime.utcnow()

                            if service_info.status != ServiceStatus.UNHEALTHY:
                                self.update_service_status(service_info.name, ServiceStatus.UNHEALTHY)

                        logger.warning(
                            f"Health check failed for service: {service_info.name}",
                            exception=e,
                            failure_count=service_info.health_check_failures,
                        )

                    # Wait for next check
                    await asyncio.sleep(service_info.health_check_interval)

            except asyncio.CancelledError:
                logger.debug(f"Health check task cancelled for service: {service_info.name}")
            except Exception as e:
                logger.error(f"Health check task error for service: {service_info.name}", exception=e)
            finally:
                if client:
                    await client.close()

        # Create and store the task
        task = asyncio.create_task(health_check_loop())
        self._health_check_tasks[service_info.name] = task

    async def start_cleanup_task(self) -> None:
        """Start the cleanup task for removing stale services."""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)

                    stale_services = []
                    with self._lock:
                        for name, service in self.services.items():
                            if service.is_stale(self.stale_threshold):
                                stale_services.append(name)

                    for name in stale_services:
                        logger.warning(f"Removing stale service: {name}")
                        self.deregister_service(name)

                except asyncio.CancelledError:
                    logger.debug("Service registry cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error("Error in service registry cleanup task", exception=e)

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop(self) -> None:
        """Stop the service registry and cleanup tasks."""
        logger.info("Stopping service registry")

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._health_check_tasks:
            await asyncio.gather(*self._health_check_tasks.values(), return_exceptions=True)

        self._health_check_tasks.clear()


# Global service registry instance
_global_registry: Optional[ServiceRegistry] = None
_registry_lock = threading.Lock()


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry instance."""
    global _global_registry

    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = ServiceRegistry()

    return _global_registry


def register_service(name: str, base_url: str, version: str = "1.0.0", **kwargs) -> ServiceInfo:
    """Register a service with the global registry."""
    registry = get_service_registry()
    return registry.register_service(name, base_url, version, **kwargs)


def discover_service(name: str) -> Optional[ServiceInfo]:
    """Discover a service by name from the global registry."""
    registry = get_service_registry()
    return registry.discover_service(name)
