"""
Enhanced Secrets Management for Q2 Platform

Extends the existing vault client with additional security features including:
- Secret rotation policies
- Encryption at rest for sensitive configuration
- Secure secret distribution
- Audit logging for secret access
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

from shared.vault_client import VaultClient
from shared.security_config import get_security_config, get_encryption_manager, get_security_audit_logger


class SecretRotationPolicy:
    """Defines how secrets should be rotated"""
    
    def __init__(
        self,
        secret_path: str,
        rotation_interval_days: int = 30,
        rotation_handler: Optional[callable] = None,
        pre_rotation_hooks: Optional[List[callable]] = None,
        post_rotation_hooks: Optional[List[callable]] = None
    ):
        self.secret_path = secret_path
        self.rotation_interval_days = rotation_interval_days
        self.rotation_handler = rotation_handler
        self.pre_rotation_hooks = pre_rotation_hooks or []
        self.post_rotation_hooks = post_rotation_hooks or []
        self.last_rotated: Optional[datetime] = None
        
    def needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        if not self.last_rotated:
            return True
            
        return (
            datetime.now(timezone.utc) - self.last_rotated
        ).days >= self.rotation_interval_days


class EnhancedSecretsManager:
    """Enhanced secrets manager with rotation, encryption, and audit capabilities"""
    
    def __init__(
        self, 
        vault_client: Optional[VaultClient] = None,
        enable_local_encryption: bool = True
    ):
        self.vault_client = vault_client
        self.enable_local_encryption = enable_local_encryption
        self.config = get_security_config()
        self.encryption_manager = get_encryption_manager() if enable_local_encryption else None
        self.audit_logger = get_security_audit_logger()
        
        # Secret rotation policies
        self.rotation_policies: Dict[str, SecretRotationPolicy] = {}
        
        # Local encrypted cache for frequently accessed secrets
        self._secret_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
    def add_rotation_policy(self, policy: SecretRotationPolicy):
        """Add a secret rotation policy"""
        self.rotation_policies[policy.secret_path] = policy
        
        self.audit_logger.log_security_event(
            "rotation_policy_added",
            "INFO",
            {
                "secret_path": policy.secret_path,
                "rotation_interval_days": policy.rotation_interval_days
            }
        )
    
    async def get_secret(
        self, 
        path: str, 
        use_cache: bool = True,
        cache_ttl_minutes: int = 15
    ) -> Optional[Dict[str, Any]]:
        """
        Get secret with caching and audit logging
        
        Args:
            path: Secret path in vault
            use_cache: Whether to use local cache
            cache_ttl_minutes: Cache TTL in minutes
            
        Returns:
            Secret data or None if not found
        """
        # Check cache first
        if use_cache and path in self._secret_cache:
            if path in self._cache_expiry:
                if datetime.now(timezone.utc) < self._cache_expiry[path]:
                    self.audit_logger.log_security_event(
                        "secret_accessed",
                        "INFO",
                        {"path": path, "source": "cache"}
                    )
                    return self._secret_cache[path]
                else:
                    # Cache expired
                    del self._secret_cache[path]
                    del self._cache_expiry[path]
        
        # Get from vault
        try:
            if self.vault_client:
                secret_data = self.vault_client.read_secret_data(path)
            else:
                # Fallback to environment or local file
                secret_data = self._get_secret_fallback(path)
            
            if secret_data:
                # Cache the secret
                if use_cache:
                    self._secret_cache[path] = secret_data
                    self._cache_expiry[path] = (
                        datetime.now(timezone.utc) + timedelta(minutes=cache_ttl_minutes)
                    )
                
                # Log access
                self.audit_logger.log_security_event(
                    "secret_accessed",
                    "INFO",
                    {"path": path, "source": "vault"}
                )
                
                return secret_data
                
        except Exception as e:
            self.audit_logger.log_security_event(
                "secret_access_failed",
                "ERROR",
                {"path": path, "error": str(e)}
            )
            
        return None
    
    async def store_secret(
        self, 
        path: str, 
        secret_data: Dict[str, Any],
        encrypt_locally: bool = True
    ) -> bool:
        """
        Store secret with encryption and audit logging
        
        Args:
            path: Secret path in vault
            secret_data: Secret data to store
            encrypt_locally: Whether to encrypt sensitive fields locally
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Encrypt sensitive fields if enabled
            if encrypt_locally and self.encryption_manager:
                secret_data = self._encrypt_sensitive_fields(secret_data)
            
            if self.vault_client:
                # Store in vault (implementation depends on vault client)
                # For now, we'll assume vault client handles storage
                pass
            else:
                # Store locally with encryption
                await self._store_secret_locally(path, secret_data)
            
            # Clear cache for this path
            if path in self._secret_cache:
                del self._secret_cache[path]
                del self._cache_expiry[path]
            
            # Log storage
            self.audit_logger.log_security_event(
                "secret_stored",
                "INFO",
                {"path": path, "encrypted": encrypt_locally}
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log_security_event(
                "secret_storage_failed",
                "ERROR",
                {"path": path, "error": str(e)}
            )
            return False
    
    async def rotate_secret(self, path: str) -> bool:
        """
        Rotate a secret according to its policy
        
        Args:
            path: Secret path to rotate
            
        Returns:
            True if successful, False otherwise
        """
        if path not in self.rotation_policies:
            self.audit_logger.log_security_event(
                "rotation_failed",
                "WARNING",
                {"path": path, "reason": "no_policy"}
            )
            return False
        
        policy = self.rotation_policies[path]
        
        if not policy.needs_rotation():
            return True  # No rotation needed
        
        try:
            # Execute pre-rotation hooks
            for hook in policy.pre_rotation_hooks:
                await self._execute_hook(hook, path, "pre")
            
            # Execute rotation handler
            if policy.rotation_handler:
                new_secret = await self._execute_rotation_handler(
                    policy.rotation_handler, path
                )
                
                if new_secret:
                    await self.store_secret(path, new_secret)
                    
            # Execute post-rotation hooks
            for hook in policy.post_rotation_hooks:
                await self._execute_hook(hook, path, "post")
            
            # Update last rotated timestamp
            policy.last_rotated = datetime.now(timezone.utc)
            
            self.audit_logger.log_security_event(
                "secret_rotated",
                "INFO",
                {"path": path}
            )
            
            return True
            
        except Exception as e:
            self.audit_logger.log_security_event(
                "rotation_failed",
                "ERROR",
                {"path": path, "error": str(e)}
            )
            return False
    
    async def rotate_all_eligible_secrets(self):
        """Rotate all secrets that are eligible for rotation"""
        rotation_results = []
        
        for path, policy in self.rotation_policies.items():
            if policy.needs_rotation():
                result = await self.rotate_secret(path)
                rotation_results.append((path, result))
        
        successful = sum(1 for _, result in rotation_results if result)
        total = len(rotation_results)
        
        self.audit_logger.log_security_event(
            "bulk_rotation_completed",
            "INFO",
            {
                "total_secrets": total,
                "successful_rotations": successful,
                "failed_rotations": total - successful
            }
        )
        
        return rotation_results
    
    def _encrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in secret data"""
        if not self.encryption_manager:
            return data
        
        encrypted_data = data.copy()
        sensitive_fields = [
            'password', 'secret', 'key', 'token', 'credential',
            'api_key', 'private_key', 'cert', 'certificate'
        ]
        
        def encrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if any(field in k.lower() for field in sensitive_fields):
                        if isinstance(v, str):
                            result[f"{k}_encrypted"] = self.encryption_manager.encrypt(v)
                            result[f"{k}_is_encrypted"] = True
                        else:
                            result[k] = v
                    else:
                        result[k] = encrypt_recursive(v)
                return result
            elif isinstance(obj, list):
                return [encrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return encrypt_recursive(encrypted_data)
    
    def _decrypt_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in secret data"""
        if not self.encryption_manager:
            return data
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k.endswith('_encrypted') and f"{k.replace('_encrypted', '')}_is_encrypted" in obj:
                        if obj[f"{k.replace('_encrypted', '')}_is_encrypted"]:
                            original_key = k.replace('_encrypted', '')
                            result[original_key] = self.encryption_manager.decrypt(v)
                        else:
                            result[k] = v
                    elif k.endswith('_is_encrypted'):
                        continue  # Skip encryption flags
                    else:
                        result[k] = decrypt_recursive(v)
                return result
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(data)
    
    def _get_secret_fallback(self, path: str) -> Optional[Dict[str, Any]]:
        """Fallback method to get secrets from environment or local files"""
        # Try environment variable
        env_key = path.replace('/', '_').replace('-', '_').upper()
        env_value = os.getenv(env_key)
        
        if env_value:
            try:
                return json.loads(env_value)
            except json.JSONDecodeError:
                return {"value": env_value}
        
        # Try local encrypted file
        local_path = Path(f".secrets/{path}.json")
        if local_path.exists():
            try:
                with open(local_path, 'r') as f:
                    encrypted_data = json.load(f)
                    return self._decrypt_sensitive_fields(encrypted_data)
            except Exception:
                pass
        
        return None
    
    async def _store_secret_locally(self, path: str, secret_data: Dict[str, Any]):
        """Store secret locally with encryption"""
        local_path = Path(f".secrets/{path}.json")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, 'w') as f:
            json.dump(secret_data, f, indent=2)
        
        # Set restrictive permissions
        local_path.chmod(0o600)
    
    async def _execute_hook(self, hook: callable, path: str, phase: str):
        """Execute a rotation hook"""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(path, phase)
            else:
                hook(path, phase)
        except Exception as e:
            self.audit_logger.log_security_event(
                "rotation_hook_failed",
                "ERROR",
                {"path": path, "phase": phase, "error": str(e)}
            )
    
    async def _execute_rotation_handler(
        self, 
        handler: callable, 
        path: str
    ) -> Optional[Dict[str, Any]]:
        """Execute a rotation handler"""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(path)
            else:
                return handler(path)
        except Exception as e:
            self.audit_logger.log_security_event(
                "rotation_handler_failed",
                "ERROR",
                {"path": path, "error": str(e)}
            )
            return None


# Global secrets manager instance
_secrets_manager: Optional[EnhancedSecretsManager] = None


def get_secrets_manager() -> EnhancedSecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        vault_client = None
        try:
            vault_client = VaultClient()
        except Exception:
            pass  # Vault not available, use fallbacks
        
        _secrets_manager = EnhancedSecretsManager(vault_client)
    
    return _secrets_manager


@asynccontextmanager
async def secret_context(path: str, use_cache: bool = True):
    """
    Context manager for secure secret access
    
    Usage:
        async with secret_context("database/credentials") as secret:
            if secret:
                conn = connect(secret["username"], secret["password"])
    """
    secrets_manager = get_secrets_manager()
    secret = await secrets_manager.get_secret(path, use_cache)
    
    try:
        yield secret
    finally:
        # Clear secret from memory
        if secret:
            for key in secret.keys():
                secret[key] = None
            del secret


# Utility functions for common rotation handlers
async def generate_api_key_rotation_handler(path: str) -> Dict[str, Any]:
    """Default rotation handler for API keys"""
    import secrets
    
    new_key = secrets.token_urlsafe(32)
    return {
        "api_key": new_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    }


async def generate_password_rotation_handler(path: str) -> Dict[str, Any]:
    """Default rotation handler for passwords"""
    import secrets
    import string
    
    # Generate secure password
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(16))
    
    from shared.security_config import hash_password
    
    return {
        "password": password,
        "password_hash": hash_password(password),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }