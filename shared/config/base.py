"""
Base configuration classes for Q2 Platform services.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pydantic import BaseSettings, Field, validator


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass


@dataclass
class DatabaseConfig:
    """Base database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "q2"
    username: str = "q2user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    def get_url(self, driver: str = "postgresql") -> str:
        """Get database URL."""
        if self.password:
            return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"{driver}://{self.username}@{self.host}:{self.port}/{self.database}"


@dataclass
class PulsarConfig:
    """Apache Pulsar configuration."""
    service_url: str = "pulsar://localhost:6650"
    admin_url: str = "http://localhost:8080"
    tenant: str = "public"
    namespace: str = "default"
    connection_timeout: int = 30
    operation_timeout: int = 30
    
    def get_topic_name(self, topic: str) -> str:
        """Get fully qualified topic name."""
        return f"persistent://{self.tenant}/{self.namespace}/{topic}"


@dataclass
class IgniteConfig:
    """Apache Ignite configuration."""
    addresses: list = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30

    def __post_init__(self):
        if self.addresses is None:
            self.addresses = ["127.0.0.1:10800"]


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    logging_level: str = "INFO"
    otlp_endpoint: Optional[str] = "http://localhost:4317"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    service_name: str = ""


@dataclass
class VaultConfig:
    """HashiCorp Vault configuration."""
    url: str = "http://localhost:8200"
    role: str = ""
    mount_point: str = "secret"
    timeout: int = 30


class BaseConfig(BaseSettings, ABC):
    """
    Base configuration class for all Q2 Platform services.
    
    Inherits from Pydantic BaseSettings for automatic environment variable loading.
    """
    
    # Service identification
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(default="1.0.0", description="Service version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Common configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pulsar: PulsarConfig = Field(default_factory=PulsarConfig)
    ignite: IgniteConfig = Field(default_factory=IgniteConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    
    # API Configuration (common to most services)
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API path prefix")
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    
    # Service URLs (for inter-service communication)
    services: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        allow_population_by_field_name = True
        
        # Support for nested configuration via environment variables
        # e.g., DATABASE__HOST=localhost sets database.host
        env_nested_delimiter = "__"
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v
    
    @validator('observability', pre=True, always=True)
    def set_service_name_in_observability(cls, v, values):
        """Set service name in observability config."""
        if isinstance(v, dict):
            v = ObservabilityConfig(**v)
        if not v.service_name and 'service_name' in values:
            v.service_name = values['service_name']
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get URL for another service."""
        return self.services.get(service_name)
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate service-specific configuration.
        
        Should raise ConfigError if configuration is invalid.
        """
        pass
    
    def dict_for_logging(self) -> Dict[str, Any]:
        """
        Get configuration dictionary suitable for logging.
        
        Excludes sensitive information like passwords.
        """
        config_dict = self.dict()
        
        # Remove sensitive fields
        if 'database' in config_dict and 'password' in config_dict['database']:
            config_dict['database']['password'] = "***"
        
        if 'vault' in config_dict and 'token' in config_dict['vault']:
            config_dict['vault']['token'] = "***"
            
        return config_dict