"""
Tenant Management System for Q2 Platform.

Provides comprehensive tenant lifecycle management including:
- Tenant registration and provisioning
- Subscription and plan management
- Tenant status and health monitoring
- Multi-level tenant hierarchies
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    """Tenant lifecycle status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    PENDING = "pending"
    TRIAL = "trial"


class TenantPlan(Enum):
    """Tenant subscription plans."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class TenantMetrics:
    """Tenant usage and performance metrics."""
    api_requests_count: int = 0
    storage_usage_bytes: int = 0
    compute_hours: float = 0.0
    active_users: int = 0
    workflows_executed: int = 0
    models_deployed: int = 0
    data_processed_bytes: int = 0
    last_activity: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Tenant:
    """Complete tenant information."""
    id: str
    name: str
    domain: Optional[str] = None
    status: TenantStatus = TenantStatus.PENDING
    plan: TenantPlan = TenantPlan.FREE
    parent_tenant_id: Optional[str] = None  # For hierarchical tenants
    contact_email: str = ""
    billing_email: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: Optional[datetime] = None
    trial_expires_at: Optional[datetime] = None
    subscription_expires_at: Optional[datetime] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metrics: TenantMetrics = field(default_factory=TenantMetrics)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Validate tenant after initialization."""
        if not self.name:
            raise ValueError("Tenant name is required")
        if not self.contact_email:
            raise ValueError("Contact email is required")


class TenantManagerError(Exception):
    """Base exception for tenant management errors."""
    pass


class TenantNotFoundError(TenantManagerError):
    """Raised when a tenant is not found."""
    pass


class TenantManager:
    """
    Comprehensive tenant management system.
    
    Features:
    - Tenant lifecycle management
    - Subscription and plan management
    - Hierarchical tenant support
    - Usage tracking and metrics
    - Feature flag management
    - Tenant health monitoring
    """
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._domain_map: Dict[str, str] = {}  # domain -> tenant_id
        self._parent_children: Dict[str, Set[str]] = {}  # parent_id -> {child_ids}
        self._lock = asyncio.Lock()
        
        # Default resource limits by plan
        self._default_limits = {
            TenantPlan.FREE: {
                "max_users": 5,
                "max_workflows": 10,
                "max_models": 2,
                "storage_gb": 1,
                "compute_hours_month": 10,
                "api_requests_month": 1000
            },
            TenantPlan.BASIC: {
                "max_users": 25,
                "max_workflows": 100,
                "max_models": 10,
                "storage_gb": 10,
                "compute_hours_month": 100,
                "api_requests_month": 10000
            },
            TenantPlan.PROFESSIONAL: {
                "max_users": 100,
                "max_workflows": 500,
                "max_models": 50,
                "storage_gb": 100,
                "compute_hours_month": 500,
                "api_requests_month": 100000
            },
            TenantPlan.ENTERPRISE: {
                "max_users": -1,  # Unlimited
                "max_workflows": -1,
                "max_models": -1,
                "storage_gb": 1000,
                "compute_hours_month": -1,
                "api_requests_month": -1
            }
        }
    
    async def create_tenant(self, name: str, contact_email: str,
                          plan: TenantPlan = TenantPlan.FREE,
                          domain: Optional[str] = None,
                          parent_tenant_id: Optional[str] = None,
                          custom_attributes: Optional[Dict[str, Any]] = None) -> str:
        """Create a new tenant."""
        async with self._lock:
            tenant_id = str(uuid4())
            
            # Check domain uniqueness
            if domain and domain in self._domain_map:
                raise TenantManagerError(f"Domain {domain} is already in use")
            
            # Validate parent tenant
            if parent_tenant_id and parent_tenant_id not in self._tenants:
                raise TenantNotFoundError(f"Parent tenant {parent_tenant_id} not found")
            
            # Set trial expiry for new tenants
            trial_expires_at = None
            if plan == TenantPlan.FREE:
                trial_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
            
            # Create tenant
            tenant = Tenant(
                id=tenant_id,
                name=name,
                domain=domain,
                contact_email=contact_email,
                plan=plan,
                parent_tenant_id=parent_tenant_id,
                trial_expires_at=trial_expires_at,
                custom_attributes=custom_attributes or {},
                resource_limits=self._default_limits.get(plan, {}).copy()
            )
            
            self._tenants[tenant_id] = tenant
            
            # Update domain mapping
            if domain:
                self._domain_map[domain] = tenant_id
            
            # Update parent-child relationships
            if parent_tenant_id:
                if parent_tenant_id not in self._parent_children:
                    self._parent_children[parent_tenant_id] = set()
                self._parent_children[parent_tenant_id].add(tenant_id)
            
            logger.info(f"Created tenant: {name} ({tenant_id})")
            return tenant_id
    
    async def get_tenant(self, tenant_id: str) -> Tenant:
        """Get tenant by ID."""
        if tenant_id not in self._tenants:
            raise TenantNotFoundError(f"Tenant {tenant_id} not found")
        
        return self._tenants[tenant_id]
    
    async def get_tenant_by_domain(self, domain: str) -> Tenant:
        """Get tenant by domain."""
        if domain not in self._domain_map:
            raise TenantNotFoundError(f"No tenant found for domain {domain}")
        
        tenant_id = self._domain_map[domain]
        return await self.get_tenant(tenant_id)
    
    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> Tenant:
        """Update tenant information."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            # Handle domain changes
            if "domain" in updates:
                new_domain = updates["domain"]
                if new_domain and new_domain in self._domain_map and self._domain_map[new_domain] != tenant_id:
                    raise TenantManagerError(f"Domain {new_domain} is already in use")
                
                # Remove old domain mapping
                if tenant.domain:
                    self._domain_map.pop(tenant.domain, None)
                
                # Add new domain mapping
                if new_domain:
                    self._domain_map[new_domain] = tenant_id
                
                tenant.domain = new_domain
            
            # Update other fields
            for field, value in updates.items():
                if hasattr(tenant, field) and field != "id":
                    setattr(tenant, field, value)
            
            logger.info(f"Updated tenant: {tenant_id}")
            return tenant
    
    async def activate_tenant(self, tenant_id: str) -> None:
        """Activate a tenant."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            if tenant.status != TenantStatus.PENDING:
                raise TenantManagerError(f"Cannot activate tenant in {tenant.status.value} status")
            
            tenant.status = TenantStatus.ACTIVE
            tenant.activated_at = datetime.now(timezone.utc)
            
            logger.info(f"Activated tenant: {tenant_id}")
    
    async def suspend_tenant(self, tenant_id: str, reason: str = "") -> None:
        """Suspend a tenant."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            if tenant.status == TenantStatus.DEACTIVATED:
                raise TenantManagerError("Cannot suspend deactivated tenant")
            
            tenant.status = TenantStatus.SUSPENDED
            tenant.custom_attributes["suspension_reason"] = reason
            tenant.custom_attributes["suspended_at"] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Suspended tenant: {tenant_id} (reason: {reason})")
    
    async def deactivate_tenant(self, tenant_id: str) -> None:
        """Deactivate a tenant (soft delete)."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            tenant.status = TenantStatus.DEACTIVATED
            tenant.custom_attributes["deactivated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Remove domain mapping
            if tenant.domain:
                self._domain_map.pop(tenant.domain, None)
            
            logger.info(f"Deactivated tenant: {tenant_id}")
    
    async def change_plan(self, tenant_id: str, new_plan: TenantPlan) -> None:
        """Change tenant subscription plan."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            old_plan = tenant.plan
            tenant.plan = new_plan
            
            # Update resource limits based on new plan
            tenant.resource_limits.update(self._default_limits.get(new_plan, {}))
            
            # Update subscription expiry if upgrading from free
            if old_plan == TenantPlan.FREE and new_plan != TenantPlan.FREE:
                tenant.subscription_expires_at = datetime.now(timezone.utc) + timedelta(days=365)
                tenant.trial_expires_at = None  # Clear trial expiry
            
            tenant.custom_attributes["plan_changed_at"] = datetime.now(timezone.utc).isoformat()
            tenant.custom_attributes["previous_plan"] = old_plan.value
            
            logger.info(f"Changed plan for tenant {tenant_id}: {old_plan.value} -> {new_plan.value}")
    
    async def list_tenants(self, status: Optional[TenantStatus] = None,
                         plan: Optional[TenantPlan] = None,
                         parent_tenant_id: Optional[str] = None) -> List[Tenant]:
        """List tenants with optional filtering."""
        tenants = list(self._tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        if plan:
            tenants = [t for t in tenants if t.plan == plan]
        
        if parent_tenant_id:
            tenants = [t for t in tenants if t.parent_tenant_id == parent_tenant_id]
        
        return tenants
    
    async def get_child_tenants(self, parent_tenant_id: str) -> List[Tenant]:
        """Get all child tenants of a parent."""
        if parent_tenant_id not in self._parent_children:
            return []
        
        child_ids = self._parent_children[parent_tenant_id]
        return [self._tenants[child_id] for child_id in child_ids if child_id in self._tenants]
    
    async def update_metrics(self, tenant_id: str, metrics_update: Dict[str, Any]) -> None:
        """Update tenant usage metrics."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            
            for metric_name, value in metrics_update.items():
                if hasattr(tenant.metrics, metric_name):
                    setattr(tenant.metrics, metric_name, value)
            
            tenant.metrics.last_activity = datetime.now(timezone.utc)
            tenant.metrics.last_updated = datetime.now(timezone.utc)
    
    async def check_resource_limits(self, tenant_id: str, resource_type: str, 
                                  requested_amount: int = 1) -> bool:
        """Check if tenant can use more resources."""
        tenant = await self.get_tenant(tenant_id)
        
        if resource_type not in tenant.resource_limits:
            return True  # No limit defined
        
        limit = tenant.resource_limits[resource_type]
        if limit == -1:  # Unlimited
            return True
        
        # Get current usage based on resource type
        current_usage = 0
        if resource_type == "max_users":
            current_usage = tenant.metrics.active_users
        elif resource_type == "max_workflows":
            current_usage = tenant.metrics.workflows_executed
        elif resource_type == "max_models":
            current_usage = tenant.metrics.models_deployed
        elif resource_type == "storage_gb":
            current_usage = tenant.metrics.storage_usage_bytes // (1024**3)  # Convert to GB
        elif resource_type == "api_requests_month":
            current_usage = tenant.metrics.api_requests_count
        elif resource_type == "compute_hours_month":
            current_usage = int(tenant.metrics.compute_hours)
        
        return (current_usage + requested_amount) <= limit
    
    async def set_feature_flag(self, tenant_id: str, feature_name: str, enabled: bool) -> None:
        """Set a feature flag for a tenant."""
        async with self._lock:
            tenant = await self.get_tenant(tenant_id)
            tenant.feature_flags[feature_name] = enabled
            
            logger.info(f"Set feature flag for tenant {tenant_id}: {feature_name} = {enabled}")
    
    async def is_feature_enabled(self, tenant_id: str, feature_name: str) -> bool:
        """Check if a feature is enabled for a tenant."""
        tenant = await self.get_tenant(tenant_id)
        return tenant.feature_flags.get(feature_name, False)
    
    async def get_tenant_health(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant health status and metrics."""
        tenant = await self.get_tenant(tenant_id)
        
        # Calculate health indicators
        now = datetime.now(timezone.utc)
        
        # Check if trial/subscription is expiring
        trial_expired = (tenant.trial_expires_at and 
                        tenant.trial_expires_at < now)
        subscription_expired = (tenant.subscription_expires_at and 
                              tenant.subscription_expires_at < now)
        
        # Check resource usage vs limits
        resource_warnings = []
        for resource, limit in tenant.resource_limits.items():
            if limit > 0:  # Skip unlimited resources
                usage_pct = await self._calculate_resource_usage_percentage(tenant, resource)
                if usage_pct > 80:  # 80% threshold
                    resource_warnings.append({
                        "resource": resource,
                        "usage_percentage": usage_pct,
                        "limit": limit
                    })
        
        # Overall health score (0-100)
        health_score = 100
        if trial_expired or subscription_expired:
            health_score -= 50
        if resource_warnings:
            health_score -= min(30, len(resource_warnings) * 10)
        if tenant.status != TenantStatus.ACTIVE:
            health_score -= 20
        
        return {
            "tenant_id": tenant_id,
            "status": tenant.status.value,
            "plan": tenant.plan.value,
            "health_score": max(0, health_score),
            "trial_expired": trial_expired,
            "subscription_expired": subscription_expired,
            "resource_warnings": resource_warnings,
            "last_activity": tenant.metrics.last_activity.isoformat() if tenant.metrics.last_activity else None,
            "metrics": {
                "api_requests": tenant.metrics.api_requests_count,
                "storage_usage_gb": tenant.metrics.storage_usage_bytes // (1024**3),
                "compute_hours": tenant.metrics.compute_hours,
                "active_users": tenant.metrics.active_users,
                "workflows_executed": tenant.metrics.workflows_executed,
                "models_deployed": tenant.metrics.models_deployed
            }
        }
    
    async def _calculate_resource_usage_percentage(self, tenant: Tenant, resource_type: str) -> float:
        """Calculate resource usage percentage."""
        limit = tenant.resource_limits.get(resource_type, 0)
        if limit <= 0:
            return 0.0
        
        current_usage = 0
        if resource_type == "max_users":
            current_usage = tenant.metrics.active_users
        elif resource_type == "max_workflows":
            current_usage = tenant.metrics.workflows_executed
        elif resource_type == "max_models":
            current_usage = tenant.metrics.models_deployed
        elif resource_type == "storage_gb":
            current_usage = tenant.metrics.storage_usage_bytes // (1024**3)
        elif resource_type == "api_requests_month":
            current_usage = tenant.metrics.api_requests_count
        elif resource_type == "compute_hours_month":
            current_usage = int(tenant.metrics.compute_hours)
        
        return (current_usage / limit) * 100
    
    async def cleanup_expired_trials(self) -> List[str]:
        """Cleanup expired trial tenants."""
        expired_tenants = []
        now = datetime.now(timezone.utc)
        
        async with self._lock:
            for tenant_id, tenant in self._tenants.items():
                if (tenant.status == TenantStatus.TRIAL and 
                    tenant.trial_expires_at and 
                    tenant.trial_expires_at < now):
                    
                    tenant.status = TenantStatus.SUSPENDED
                    tenant.custom_attributes["suspension_reason"] = "Trial expired"
                    expired_tenants.append(tenant_id)
        
        if expired_tenants:
            logger.info(f"Suspended {len(expired_tenants)} expired trial tenants")
        
        return expired_tenants
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide tenant statistics."""
        total_tenants = len(self._tenants)
        
        status_counts = {}
        plan_counts = {}
        
        for tenant in self._tenants.values():
            status_counts[tenant.status.value] = status_counts.get(tenant.status.value, 0) + 1
            plan_counts[tenant.plan.value] = plan_counts.get(tenant.plan.value, 0) + 1
        
        total_metrics = TenantMetrics()
        for tenant in self._tenants.values():
            total_metrics.api_requests_count += tenant.metrics.api_requests_count
            total_metrics.storage_usage_bytes += tenant.metrics.storage_usage_bytes
            total_metrics.compute_hours += tenant.metrics.compute_hours
            total_metrics.active_users += tenant.metrics.active_users
            total_metrics.workflows_executed += tenant.metrics.workflows_executed
            total_metrics.models_deployed += tenant.metrics.models_deployed
            total_metrics.data_processed_bytes += tenant.metrics.data_processed_bytes
        
        return {
            "total_tenants": total_tenants,
            "status_distribution": status_counts,
            "plan_distribution": plan_counts,
            "aggregate_metrics": {
                "total_api_requests": total_metrics.api_requests_count,
                "total_storage_gb": total_metrics.storage_usage_bytes // (1024**3),
                "total_compute_hours": total_metrics.compute_hours,
                "total_active_users": total_metrics.active_users,
                "total_workflows_executed": total_metrics.workflows_executed,
                "total_models_deployed": total_metrics.models_deployed,
                "total_data_processed_gb": total_metrics.data_processed_bytes // (1024**3)
            }
        }


# Global instance
tenant_manager = TenantManager()