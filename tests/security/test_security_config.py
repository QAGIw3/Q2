"""
Security Configuration Tests

Tests for the security configuration and policy enforcement.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from shared.security_config import (
    SecurityConfig, 
    get_security_config,
    EncryptionManager,
    JWTManager,
    SecurityAuditLogger
)
from shared.security.policies import (
    SecurityPolicyEngine,
    Role,
    Permission,
    SecurityLevel,
    check_user_access
)
from shared.security.auth_utils import (
    validate_jwt_token,
    check_permissions,
    SecurityContext
)


class TestSecurityConfig:
    """Test security configuration"""
    
    def test_default_config(self):
        """Test default security configuration"""
        config = SecurityConfig()
        
        assert config.password_salt_rounds == 12
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_access_token_expire_minutes == 30
        assert config.session_cookie_secure is True
        assert config.session_cookie_httponly is True
        assert config.failed_login_attempts_limit == 5
    
    def test_config_from_env(self):
        """Test loading configuration from environment"""
        with patch.dict(os.environ, {
            'JWT_ACCESS_TOKEN_EXPIRE_MINUTES': '60',
            'FAILED_LOGIN_ATTEMPTS_LIMIT': '10',
            'CORS_ORIGINS': 'https://app.example.com,https://api.example.com'
        }):
            config = SecurityConfig.from_env()
            
            assert config.jwt_access_token_expire_minutes == 60
            assert config.failed_login_attempts_limit == 10
            assert 'https://app.example.com' in config.cors_origins


class TestEncryptionManager:
    """Test encryption functionality"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"
        
        hashed = EncryptionManager.hash_password(password)
        assert EncryptionManager.verify_password(password, hashed)
        assert not EncryptionManager.verify_password("wrong_password", hashed)
    
    def test_data_encryption(self):
        """Test data encryption and decryption"""
        manager = EncryptionManager()
        data = "sensitive_data_123"
        
        encrypted = manager.encrypt(data)
        assert encrypted != data
        
        decrypted = manager.decrypt(encrypted)
        assert decrypted == data


class TestJWTManager:
    """Test JWT token management"""
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        manager = JWTManager()
        
        payload = {"user_id": "123", "roles": ["user"]}
        
        # Create access token
        token = manager.create_access_token(payload)
        assert token is not None
        
        # Verify token
        decoded = manager.verify_token(token)
        assert decoded["user_id"] == "123"
        assert decoded["roles"] == ["user"]
        assert decoded["type"] == "access"
    
    def test_refresh_token(self):
        """Test refresh token creation"""
        manager = JWTManager()
        
        payload = {"user_id": "123"}
        token = manager.create_refresh_token(payload)
        
        decoded = manager.verify_token(token)
        assert decoded["user_id"] == "123"
        assert decoded["type"] == "refresh"
    
    def test_expired_token(self):
        """Test expired token handling"""
        # Create manager with very short expiration
        config = SecurityConfig()
        config.jwt_access_token_expire_minutes = 0  # Immediate expiration
        
        manager = JWTManager(config)
        
        payload = {"user_id": "123"}
        token = manager.create_access_token(payload)
        
        with pytest.raises(ValueError, match="Token has expired"):
            manager.verify_token(token)


class TestSecurityPolicyEngine:
    """Test security policy engine"""
    
    def test_role_permissions(self):
        """Test role-based permissions"""
        engine = SecurityPolicyEngine()
        
        # Test user role
        result = engine.check_access_permission(
            ["user"], "resource", Permission.READ
        )
        assert result is True
        
        result = engine.check_access_permission(
            ["user"], "resource", Permission.DELETE
        )
        assert result is False
        
        # Test admin role
        result = engine.check_access_permission(
            ["admin"], "resource", Permission.DELETE
        )
        assert result is True
    
    def test_policy_enforcement(self):
        """Test security policy enforcement"""
        engine = SecurityPolicyEngine() 
        
        # Test RBAC policy
        context = {
            "authenticated": True,
            "user_roles": ["admin"]
        }
        
        result = engine.enforce_policy("rbac_enforcement", context)
        assert result["allowed"] is True
        
        # Test without authentication
        context = {"authenticated": False}
        result = engine.enforce_policy("rbac_enforcement", context)
        assert result["allowed"] is False
    
    def test_data_classification(self):
        """Test data classification"""
        engine = SecurityPolicyEngine()
        
        # Test PII data
        data = {"email": "user@example.com", "phone": "123-456-7890"}
        classification = engine.classify_data(data, "user_data")
        
        assert classification.classification == SecurityLevel.CONFIDENTIAL
        assert classification.encryption_required is True
        
        # Test non-PII data
        data = {"count": 42, "status": "active"}
        classification = engine.classify_data(data, "metrics")
        
        assert classification.encryption_required is False
    
    def test_custom_role_creation(self):
        """Test creating custom roles"""
        engine = SecurityPolicyEngine()
        
        custom_role = Role(
            name="analyst",
            description="Data analyst role",
            permissions={Permission.READ, Permission.EXECUTE},
            security_level=SecurityLevel.INTERNAL
        )
        
        engine.add_custom_role(custom_role)
        assert "analyst" in engine.roles
        
        # Test permission check
        result = engine.check_access_permission(
            ["analyst"], "data", Permission.READ
        )
        assert result is True


class TestAuthUtils:
    """Test authentication utilities"""
    
    def test_jwt_validation(self):
        """Test JWT token validation"""
        # Create a valid token
        manager = JWTManager()
        payload = {"user_id": "123", "roles": ["user"]}
        token = manager.create_access_token(payload)
        
        # Validate token
        decoded = validate_jwt_token(token)
        assert decoded["user_id"] == "123"
    
    def test_invalid_jwt(self):
        """Test invalid JWT handling"""
        with pytest.raises(Exception):
            validate_jwt_token("invalid.token.here")
    
    def test_permission_checking(self):
        """Test permission checking logic"""
        from shared.q_auth_parser.models import UserClaims
        
        user_claims = UserClaims(
            user_id="123",
            roles=["developer"],
            preferred_username="testuser"
        )
        
        # Test valid permission
        result = check_permissions(user_claims, ["developer", "admin"])
        assert result is True
        
        # Test invalid permission
        result = check_permissions(user_claims, ["admin"])
        assert result is False


class TestSecurityContext:
    """Test security context functionality"""
    
    def test_security_context_creation(self):
        """Test security context creation"""
        from shared.q_auth_parser.models import UserClaims
        
        user_claims = UserClaims(
            user_id="123",
            roles=["admin"],
            preferred_username="testuser"
        )
        
        context = SecurityContext(
            user_claims=user_claims,
            session_id="session_123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0",
            security_level="elevated",
            mfa_verified=True
        )
        
        assert context.user_claims.user_id == "123"
        assert context.security_level == "elevated"
        assert context.mfa_verified is True


@pytest.mark.asyncio
class TestSecurityAuditLogger:
    """Test security audit logging"""
    
    async def test_authentication_logging(self):
        """Test authentication event logging"""
        with patch('shared.observability.audit.audit_log') as mock_audit:
            logger = SecurityAuditLogger()
            
            logger.log_authentication_event(
                "user123",
                "login",
                True,
                {"ip_address": "192.168.1.1"}
            )
            
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            
            assert call_args[1]["action"] == "auth.login"
            assert call_args[1]["user"] == "user123"
            assert call_args[1]["details"]["success"] is True
    
    async def test_authorization_logging(self):
        """Test authorization event logging"""
        with patch('shared.observability.audit.audit_log') as mock_audit:
            logger = SecurityAuditLogger()
            
            logger.log_authorization_event(
                "user123",
                "/admin/users",
                "read",
                True
            )
            
            mock_audit.assert_called_once()
            call_args = mock_audit.call_args
            
            assert call_args[1]["action"] == "auth.authorization"
            assert call_args[1]["details"]["resource"] == "/admin/users"
            assert call_args[1]["details"]["granted"] is True


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_end_to_end_auth_flow(self):
        """Test complete authentication flow"""
        # 1. Create user credentials
        password = "secure_password_123"
        hashed_password = EncryptionManager.hash_password(password)
        
        # 2. Verify password
        assert EncryptionManager.verify_password(password, hashed_password)
        
        # 3. Create JWT token
        manager = JWTManager()
        payload = {"user_id": "123", "roles": ["user"]}
        token = manager.create_access_token(payload)
        
        # 4. Validate token
        decoded = validate_jwt_token(token)
        assert decoded["user_id"] == "123"
        
        # 5. Check permissions
        from shared.q_auth_parser.models import UserClaims
        user_claims = UserClaims(
            user_id=decoded["user_id"],
            roles=decoded["roles"],
            preferred_username="testuser"
        )
        
        assert check_permissions(user_claims, ["user"])
    
    def test_policy_enforcement_integration(self):
        """Test policy enforcement with real context"""
        engine = SecurityPolicyEngine()
        
        # Test complete RBAC enforcement
        context = {
            "authenticated": True,
            "user_roles": ["admin"],
            "resource_type": "admin",
            "user_id": "admin_user"
        }
        
        result = engine.enforce_policy("rbac_enforcement", context)
        assert result["allowed"] is True
        
        # Test with insufficient permissions
        context["user_roles"] = ["user"]
        result = engine.enforce_policy("rbac_enforcement", context)
        assert result["allowed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])