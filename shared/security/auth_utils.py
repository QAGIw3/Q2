"""
Enhanced Authentication Utilities for Q2 Platform

Extends the existing q_auth_parser with additional security features.
"""

import os
import jwt
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from functools import wraps
from fastapi import HTTPException, Request
from pydantic import BaseModel

from shared.q_auth_parser.models import UserClaims
from shared.security_config import get_security_config, get_security_audit_logger


class SecurityContext(BaseModel):
    """Enhanced security context with additional security information"""
    user_claims: UserClaims
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    last_activity: Optional[datetime] = None
    security_level: str = "standard"  # standard, elevated, admin
    mfa_verified: bool = False
    

class PermissionDeniedError(Exception):
    """Raised when a user lacks required permissions"""
    pass


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


def validate_jwt_token(token: str, verify_signature: bool = True) -> Dict[str, Any]:
    """
    Validate and decode a JWT token with enhanced security checks
    
    Args:
        token: JWT token to validate
        verify_signature: Whether to verify the token signature
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    config = get_security_config()
    audit_logger = get_security_audit_logger()
    
    try:
        # Decode the token
        if verify_signature:
            payload = jwt.decode(
                token, 
                config.jwt_secret_key, 
                algorithms=[config.jwt_algorithm]
            )
        else:
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
        
        # Additional security validations
        now = datetime.now(timezone.utc)
        
        # Check if token is expired
        if 'exp' in payload:
            exp = datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
            if now > exp:
                audit_logger.log_security_event(
                    "token_expired",
                    "WARNING", 
                    {"token_type": payload.get("type", "unknown")}
                )
                raise AuthenticationError("Token has expired")
        
        # Check if token is issued in the future (clock skew protection)
        if 'iat' in payload:
            iat = datetime.fromtimestamp(payload['iat'], tz=timezone.utc)
            if iat > now:
                audit_logger.log_security_event(
                    "token_future_issued",
                    "ERROR",
                    {"issued_at": iat.isoformat()}
                )
                raise AuthenticationError("Token issued in the future")
        
        # Log successful token validation
        audit_logger.log_authentication_event(
            payload.get("sub", "unknown"),
            "token_validated",
            True,
            {"token_type": payload.get("type", "access")}
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        audit_logger.log_security_event("token_expired", "WARNING", {"reason": "signature_expired"})
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        audit_logger.log_security_event("token_invalid", "ERROR", {"error": str(e)})
        raise AuthenticationError(f"Invalid token: {str(e)}")


def extract_user_claims(request: Request) -> Optional[UserClaims]:
    """
    Extract user claims from request with enhanced security validation
    
    Args:
        request: FastAPI request object
        
    Returns:
        UserClaims if valid, None otherwise
    """
    from shared.q_auth_parser import get_user_claims_from_http
    
    audit_logger = get_security_audit_logger()
    
    try:
        # Try to get claims using existing parser
        claims = get_user_claims_from_http(request)
        
        if claims:
            # Additional security validations
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            
            # Log successful claim extraction
            audit_logger.log_authentication_event(
                claims.user_id,
                "claims_extracted",
                True,
                {
                    "ip_address": client_ip,
                    "user_agent": user_agent,
                    "roles": claims.roles
                }
            )
            
        return claims
        
    except Exception as e:
        # Log failed claim extraction
        audit_logger.log_security_event(
            "claims_extraction_failed",
            "ERROR",
            {"error": str(e)}
        )
        return None


def check_permissions(
    user_claims: UserClaims, 
    required_permissions: Union[str, List[str]],
    resource: Optional[str] = None
) -> bool:
    """
    Check if user has required permissions
    
    Args:
        user_claims: User claims from JWT
        required_permissions: Required permission(s)
        resource: Specific resource being accessed
        
    Returns:
        True if user has permissions, False otherwise
    """
    if isinstance(required_permissions, str):
        required_permissions = [required_permissions]
    
    audit_logger = get_security_audit_logger()
    
    # Check if user has any of the required permissions
    user_permissions = set(user_claims.roles)  # In this system, roles act as permissions
    required_permissions_set = set(required_permissions)
    
    has_permission = bool(user_permissions.intersection(required_permissions_set))
    
    # Special case: admin role has all permissions
    if "admin" in user_permissions:
        has_permission = True
    
    # Log authorization attempt
    audit_logger.log_authorization_event(
        user_claims.user_id,
        resource or "unknown",
        ",".join(required_permissions),
        has_permission
    )
    
    return has_permission


def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions for endpoint access
    
    Args:
        required_permissions: Required permission(s)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Extract user claims
            user_claims = extract_user_claims(request)
            
            if not user_claims:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            # Check permissions
            if not check_permissions(user_claims, list(required_permissions)):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
            
            # Add user claims to kwargs for convenience
            kwargs['user_claims'] = user_claims
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


def create_security_context(request: Request) -> Optional[SecurityContext]:
    """
    Create a comprehensive security context from request
    
    Args:
        request: FastAPI request object
        
    Returns:
        SecurityContext if valid, None otherwise
    """
    user_claims = extract_user_claims(request)
    
    if not user_claims:
        return None
    
    return SecurityContext(
        user_claims=user_claims,
        session_id=request.headers.get("x-session-id"),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        last_activity=datetime.now(timezone.utc),
        security_level="elevated" if "admin" in user_claims.roles else "standard",
        mfa_verified=request.headers.get("x-mfa-verified", "false").lower() == "true"
    )


def validate_service_to_service_auth(request: Request) -> bool:
    """
    Validate service-to-service authentication using mTLS or service tokens
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if valid service authentication, False otherwise
    """
    audit_logger = get_security_audit_logger()
    
    # Check for service token in headers
    service_token = request.headers.get("x-service-token")
    if service_token:
        try:
            # Validate service token
            payload = validate_jwt_token(service_token)
            
            # Check if it's a service token
            if payload.get("type") == "service":
                service_name = payload.get("service_name")
                
                audit_logger.log_authentication_event(
                    service_name or "unknown_service",
                    "service_auth_success",
                    True,
                    {"auth_method": "service_token"}
                )
                
                return True
                
        except AuthenticationError:
            audit_logger.log_security_event(
                "service_auth_failed",
                "ERROR",
                {"auth_method": "service_token", "reason": "invalid_token"}
            )
            return False
    
    # Check for mTLS client certificate
    # In production, this would check the client certificate from the request
    client_cert = request.headers.get("x-client-cert-cn")  # Common Name from cert
    if client_cert and client_cert.startswith("service."):
        audit_logger.log_authentication_event(
            client_cert,
            "service_auth_success", 
            True,
            {"auth_method": "mtls"}
        )
        return True
    
    # No valid service authentication found
    audit_logger.log_security_event(
        "service_auth_failed",
        "WARNING",
        {"reason": "no_valid_auth"}
    )
    
    return False


class RateLimiter:
    """Simple in-memory rate limiter for security"""
    
    def __init__(self):
        self._requests: Dict[str, List[datetime]] = {}
        self.config = get_security_config()
    
    def is_allowed(self, identifier: str, limit: Optional[int] = None) -> bool:
        """
        Check if request is allowed based on rate limiting
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Rate limit override (defaults to config)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.now(timezone.utc)
        limit = limit or self.config.api_rate_limit
        
        # Clean old requests (older than 1 minute)
        cutoff = now - datetime.timedelta(minutes=1)
        
        if identifier in self._requests:
            self._requests[identifier] = [
                req_time for req_time in self._requests[identifier]
                if req_time > cutoff
            ]
        else:
            self._requests[identifier] = []
        
        # Check if under limit
        if len(self._requests[identifier]) >= limit:
            return False
        
        # Add current request
        self._requests[identifier].append(now)
        return True


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter