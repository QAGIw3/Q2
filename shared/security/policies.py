"""
Security Policies and Enforcement for Q2 Platform

Implements comprehensive security policies including:
- Role-Based Access Control (RBAC)
- Data classification and protection
- Security compliance checks
- Policy enforcement mechanisms
"""

import os
import json
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
from pydantic import BaseModel, Field

from shared.security_config import get_security_audit_logger


class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Permission(Enum):
    """System permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"


class Role(BaseModel):
    """Role definition with permissions"""
    name: str
    description: str
    permissions: Set[Permission]
    allowed_resources: List[str] = Field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    max_session_duration_hours: int = 8


class SecurityPolicy(BaseModel):
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: str  # rbac, data_protection, compliance, etc.
    rules: List[Dict[str, Any]]
    enforcement_level: str = "strict"  # strict, warn, audit
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    enabled: bool = True


class DataClassification(BaseModel):
    """Data classification metadata"""
    classification: SecurityLevel
    retention_days: int
    encryption_required: bool = True
    access_logging_required: bool = True
    restricted_countries: List[str] = Field(default_factory=list)


class SecurityPolicyEngine:
    """Enforces security policies across the platform"""
    
    def __init__(self):
        self.audit_logger = get_security_audit_logger()
        
        # Built-in roles
        self.roles: Dict[str, Role] = {
            "user": Role(
                name="user",
                description="Standard user with basic permissions",
                permissions={Permission.READ},
                security_level=SecurityLevel.INTERNAL
            ),
            "developer": Role(
                name="developer",
                description="Developer with code access",
                permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE},
                allowed_resources=["agentQ", "workflows", "integrations"],
                security_level=SecurityLevel.INTERNAL
            ),
            "admin": Role(
                name="admin", 
                description="System administrator",
                permissions={Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
                security_level=SecurityLevel.CONFIDENTIAL,
                max_session_duration_hours=4
            ),
            "auditor": Role(
                name="auditor",
                description="Security auditor with read-only access",
                permissions={Permission.READ, Permission.AUDIT},
                security_level=SecurityLevel.RESTRICTED
            )
        }
        
        # Security policies
        self.policies: Dict[str, SecurityPolicy] = {}
        self._load_default_policies()
        
        # Data classifications
        self.data_classifications: Dict[str, DataClassification] = {
            "user_data": DataClassification(
                classification=SecurityLevel.CONFIDENTIAL,
                retention_days=2555,  # 7 years
                encryption_required=True,
                access_logging_required=True
            ),
            "system_logs": DataClassification(
                classification=SecurityLevel.INTERNAL,
                retention_days=365,
                encryption_required=False,
                access_logging_required=True
            ),
            "security_events": DataClassification(
                classification=SecurityLevel.RESTRICTED,
                retention_days=2555,
                encryption_required=True,
                access_logging_required=True
            ),
            "agent_configurations": DataClassification(
                classification=SecurityLevel.CONFIDENTIAL,
                retention_days=1095,  # 3 years
                encryption_required=True
            )
        }
    
    def _load_default_policies(self):
        """Load default security policies"""
        
        # RBAC Policy
        self.policies["rbac_enforcement"] = SecurityPolicy(
            policy_id="rbac_enforcement",
            name="Role-Based Access Control",
            description="Enforce role-based access control for all resources",
            policy_type="rbac",
            rules=[
                {
                    "condition": "always",
                    "action": "require_authentication",
                    "resources": ["*"]
                },
                {
                    "condition": "resource_type == 'admin'",
                    "action": "require_role",
                    "required_roles": ["admin"],
                    "resources": ["/admin/*", "/system/*"]
                },
                {
                    "condition": "resource_type == 'audit'",
                    "action": "require_role", 
                    "required_roles": ["admin", "auditor"],
                    "resources": ["/audit/*", "/logs/*"]
                }
            ]
        )
        
        # Data Protection Policy
        self.policies["data_protection"] = SecurityPolicy(
            policy_id="data_protection",
            name="Data Protection and Privacy",
            description="Protect sensitive data according to classification",
            policy_type="data_protection",
            rules=[
                {
                    "condition": "data_classification == 'confidential'",
                    "action": "require_encryption",
                    "encryption_algorithm": "AES-256"
                },
                {
                    "condition": "data_classification == 'restricted'",
                    "action": "require_audit_logging",
                    "log_level": "INFO"
                },
                {
                    "condition": "data_contains_pii == true",
                    "action": "apply_retention_policy",
                    "retention_days": 2555
                }
            ]
        )
        
        # Session Security Policy
        self.policies["session_security"] = SecurityPolicy(
            policy_id="session_security",
            name="Session Security",
            description="Secure session management",
            policy_type="session",
            rules=[
                {
                    "condition": "role == 'admin'",
                    "action": "limit_session_duration",
                    "max_duration_hours": 4
                },
                {
                    "condition": "always",
                    "action": "require_secure_cookies",
                    "secure": True,
                    "httponly": True,
                    "samesite": "strict"
                },
                {
                    "condition": "failed_login_attempts >= 5",
                    "action": "lockout_account",
                    "lockout_duration_minutes": 15
                }
            ]
        )
    
    def check_access_permission(
        self, 
        user_roles: List[str], 
        resource: str, 
        action: Permission,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if user has permission to perform action on resource"""
        
        context = context or {}
        
        # Check if user has any valid role
        user_role_objects = [self.roles.get(role) for role in user_roles if role in self.roles]
        if not user_role_objects:
            self.audit_logger.log_authorization_event(
                context.get("user_id", "unknown"),
                resource,
                action.value,
                False
            )
            return False
        
        # Check permissions
        for role in user_role_objects:
            if action in role.permissions or Permission.ADMIN in role.permissions:
                # Check resource restrictions
                if role.allowed_resources and resource not in role.allowed_resources:
                    if not any(resource.startswith(allowed) for allowed in role.allowed_resources):
                        continue
                
                self.audit_logger.log_authorization_event(
                    context.get("user_id", "unknown"),
                    resource,
                    action.value,
                    True
                )
                return True
        
        # No permission found
        self.audit_logger.log_authorization_event(
            context.get("user_id", "unknown"),
            resource,
            action.value,
            False
        )
        return False
    
    def classify_data(self, data: Dict[str, Any], data_type: str) -> DataClassification:
        """Classify data according to security policies"""
        
        # Check for PII indicators
        pii_fields = ['email', 'phone', 'ssn', 'credit_card', 'passport', 'address']
        contains_pii = any(
            field in str(data).lower() for field in pii_fields
        )
        
        # Default classification
        if data_type in self.data_classifications:
            classification = self.data_classifications[data_type]
        else:
            classification = DataClassification(
                classification=SecurityLevel.INTERNAL,
                retention_days=365,
                encryption_required=contains_pii
            )
        
        # Upgrade classification if PII is detected
        if contains_pii:
            classification.classification = SecurityLevel.CONFIDENTIAL
            classification.encryption_required = True
            classification.access_logging_required = True
        
        return classification
    
    def enforce_policy(
        self, 
        policy_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enforce a specific security policy"""
        
        if policy_id not in self.policies:
            return {"allowed": False, "reason": "Policy not found"}
        
        policy = self.policies[policy_id]
        
        if not policy.enabled:
            return {"allowed": True, "reason": "Policy disabled"}
        
        violations = []
        actions_required = []
        
        # Evaluate each rule in the policy
        for rule in policy.rules:
            if self._evaluate_condition(rule.get("condition", ""), context):
                action = rule.get("action")
                
                if action == "require_authentication":
                    if not context.get("authenticated", False):
                        violations.append("Authentication required")
                
                elif action == "require_role":
                    required_roles = rule.get("required_roles", [])
                    user_roles = context.get("user_roles", [])
                    if not any(role in user_roles for role in required_roles):
                        violations.append(f"Required role: {required_roles}")
                
                elif action == "require_encryption":
                    if not context.get("encrypted", False):
                        actions_required.append("encrypt_data")
                
                elif action == "require_audit_logging":
                    actions_required.append("audit_log")
                
                elif action == "limit_session_duration":
                    max_duration = rule.get("max_duration_hours", 8)
                    session_start = context.get("session_start")
                    if session_start:
                        session_age = (datetime.now(timezone.utc) - session_start).total_seconds() / 3600
                        if session_age > max_duration:
                            violations.append("Session expired")
        
        # Determine result based on enforcement level
        if violations:
            if policy.enforcement_level == "strict":
                self.audit_logger.log_security_event(
                    "policy_violation",
                    "ERROR",
                    {
                        "policy_id": policy_id,
                        "violations": violations,
                        "context": context
                    }
                )
                return {"allowed": False, "violations": violations}
            
            elif policy.enforcement_level == "warn":
                self.audit_logger.log_security_event(
                    "policy_warning",
                    "WARNING", 
                    {
                        "policy_id": policy_id,
                        "violations": violations,
                        "context": context
                    }
                )
                return {"allowed": True, "warnings": violations, "actions_required": actions_required}
        
        return {"allowed": True, "actions_required": actions_required}
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a policy condition"""
        if condition == "always":
            return True
        
        # Simple condition evaluation (in production, use a proper expression parser)
        try:
            # Replace context variables
            for key, value in context.items():
                condition = condition.replace(key, f"'{value}'" if isinstance(value, str) else str(value))
            
            # Safe evaluation of simple conditions
            allowed_names = {
                "__builtins__": {},
                "True": True,
                "False": False,
                "None": None
            }
            
            return eval(condition, allowed_names)
            
        except Exception:
            return False
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report"""
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "policies": {
                "total": len(self.policies),
                "enabled": sum(1 for p in self.policies.values() if p.enabled),
                "disabled": sum(1 for p in self.policies.values() if not p.enabled)
            },
            "roles": {
                "total": len(self.roles),
                "details": [
                    {
                        "name": role.name,
                        "permissions": [p.value for p in role.permissions],
                        "security_level": role.security_level.value
                    }
                    for role in self.roles.values()
                ]
            },
            "data_classifications": {
                "total": len(self.data_classifications),
                "by_level": {}
            }
        }
        
        # Count classifications by level
        for classification in self.data_classifications.values():
            level = classification.classification.value
            report["data_classifications"]["by_level"][level] = (
                report["data_classifications"]["by_level"].get(level, 0) + 1
            )
        
        return report
    
    def add_custom_role(self, role: Role):
        """Add a custom role to the system"""
        self.roles[role.name] = role
        
        self.audit_logger.log_security_event(
            "role_added",
            "INFO",
            {
                "role_name": role.name,
                "permissions": [p.value for p in role.permissions],
                "security_level": role.security_level.value
            }
        )
    
    def add_custom_policy(self, policy: SecurityPolicy):
        """Add a custom security policy"""
        self.policies[policy.policy_id] = policy
        
        self.audit_logger.log_security_event(
            "policy_added",
            "INFO",
            {
                "policy_id": policy.policy_id,
                "policy_type": policy.policy_type,
                "enforcement_level": policy.enforcement_level
            }
        )


# Global policy engine instance
_policy_engine: Optional[SecurityPolicyEngine] = None


def get_security_policy_engine() -> SecurityPolicyEngine:
    """Get the global security policy engine"""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = SecurityPolicyEngine()
    return _policy_engine


# Utility functions for common policy checks
def check_user_access(user_roles: List[str], resource: str, action: str, context: Dict[str, Any] = None) -> bool:
    """Check if user has access to perform action on resource"""
    engine = get_security_policy_engine()
    
    try:
        permission = Permission(action)
    except ValueError:
        return False
    
    return engine.check_access_permission(user_roles, resource, permission, context)


def enforce_data_protection(data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
    """Enforce data protection policies on data"""
    engine = get_security_policy_engine()
    
    classification = engine.classify_data(data, data_type)
    
    context = {
        "data_classification": classification.classification.value,
        "data_contains_pii": any(
            field in str(data).lower() 
            for field in ['email', 'phone', 'ssn', 'credit_card']
        )
    }
    
    policy_result = engine.enforce_policy("data_protection", context)
    
    return {
        "classification": classification,
        "policy_result": policy_result,
        "actions_required": policy_result.get("actions_required", [])
    }