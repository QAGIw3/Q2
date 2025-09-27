"""
Security Configuration for Q2 Platform

Centralized security settings and utilities to address Phase 4 requirements.
"""

import os
import secrets
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone


class SecurityConfig(BaseModel):
    """Centralized security configuration"""
    
    # Encryption settings
    encryption_key: Optional[str] = Field(default=None, description="Master encryption key")
    password_salt_rounds: int = Field(default=12, description="BCrypt salt rounds")
    
    # JWT settings  
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30)
    jwt_refresh_token_expire_days: int = Field(default=7)
    
    # API Security
    api_rate_limit: int = Field(default=100, description="Requests per minute")
    cors_origins: List[str] = Field(default=["https://localhost", "https://127.0.0.1"])
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"])
    
    # Session Security
    session_cookie_secure: bool = Field(default=True)
    session_cookie_httponly: bool = Field(default=True) 
    session_cookie_samesite: str = Field(default="strict")
    
    # Audit & Compliance
    audit_log_retention_days: int = Field(default=365)
    security_event_log_level: str = Field(default="INFO")
    failed_login_attempts_limit: int = Field(default=5)
    account_lockout_duration_minutes: int = Field(default=15)
    
    # Vault Integration
    vault_enable_encryption: bool = Field(default=True)
    vault_secret_rotation_days: int = Field(default=30)
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Load security configuration from environment variables"""
        return cls(
            encryption_key=os.getenv('SECURITY_ENCRYPTION_KEY'),
            jwt_secret_key=os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32)),
            jwt_access_token_expire_minutes=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', '30')),
            cors_origins=os.getenv('CORS_ORIGINS', 'https://localhost,https://127.0.0.1').split(','),
            allowed_hosts=os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            session_cookie_secure=os.getenv('SESSION_COOKIE_SECURE', 'true').lower() == 'true',
            audit_log_retention_days=int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '365')),
            failed_login_attempts_limit=int(os.getenv('FAILED_LOGIN_ATTEMPTS_LIMIT', '5')),
        )


# Global security configuration instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get the global security configuration instance"""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig.from_env()
    return _security_config


class EncryptionManager:
    """Handles data encryption and decryption"""
    
    def __init__(self, key: Optional[str] = None):
        if key:
            self.key = key.encode()
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using BCrypt"""
        salt = bcrypt.gensalt(rounds=get_security_config().password_salt_rounds)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod 
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


class JWTManager:
    """Handles JWT token operations"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or get_security_config()
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.config.jwt_access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=self.config.jwt_refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")


class SecurityAuditLogger:
    """Enhanced audit logging for security events"""
    
    def __init__(self):
        from shared.observability.audit import audit_log
        self.audit_log = audit_log
        self.config = get_security_config()
    
    def log_authentication_event(self, user_id: str, event_type: str, success: bool, details: Dict[str, Any] = None):
        """Log authentication events"""
        self.audit_log(
            action=f"auth.{event_type}",
            user=user_id,
            details={
                "success": success,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "INFO" if success else "WARNING",
                **(details or {})
            }
        )
    
    def log_authorization_event(self, user_id: str, resource: str, action: str, granted: bool):
        """Log authorization events"""
        self.audit_log(
            action="auth.authorization",
            user=user_id,
            details={
                "resource": resource,
                "action": action,
                "granted": granted,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "INFO" if granted else "WARNING"
            }
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log general security events"""
        self.audit_log(
            action=f"security.{event_type}",
            user="system",
            details={
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details
            }
        )


# Global instances
_encryption_manager: Optional[EncryptionManager] = None
_jwt_manager: Optional[JWTManager] = None
_audit_logger: Optional[SecurityAuditLogger] = None


def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance"""
    global _encryption_manager
    if _encryption_manager is None:
        config = get_security_config()
        _encryption_manager = EncryptionManager(config.encryption_key)
    return _encryption_manager


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


def get_security_audit_logger() -> SecurityAuditLogger:
    """Get the global security audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


# Utility functions for backward compatibility
def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive data"""
    return get_encryption_manager().encrypt(data)


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data"""
    return get_encryption_manager().decrypt(encrypted_data)


def hash_password(password: str) -> str:
    """Hash a password"""
    return EncryptionManager.hash_password(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password"""
    return EncryptionManager.verify_password(password, hashed)


def generate_secure_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)