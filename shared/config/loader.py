"""
Configuration loader for Q2 Platform services.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path

from .base import BaseConfig, ConfigError
from shared.vault_client import VaultClient

T = TypeVar('T', bound=BaseConfig)


class ConfigLoader:
    """
    Centralized configuration loader for Q2 Platform services.
    
    Supports loading configuration from:
    - HashiCorp Vault (primary)
    - Environment variables
    - Configuration files
    - Default values
    """
    
    def __init__(self, service_name: str, vault_role: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            service_name: Name of the service
            vault_role: Vault role for authentication (defaults to f"{service_name}-role")
        """
        self.service_name = service_name
        self.vault_role = vault_role or f"{service_name}-role"
        self._vault_client: Optional[VaultClient] = None
    
    def load_config(self, config_class: Type[T], vault_path: Optional[str] = None) -> T:
        """
        Load configuration for a service.
        
        Args:
            config_class: Configuration class to instantiate
            vault_path: Vault path (defaults to f"secret/data/{service_name}/config")
            
        Returns:
            Configured instance of config_class
            
        Raises:
            ConfigError: If configuration loading fails
        """
        if vault_path is None:
            vault_path = f"secret/data/{self.service_name}/config"
        
        config_data = {}
        
        # 1. Load from Vault (if available)
        vault_data = self._load_from_vault(vault_path)
        if vault_data:
            config_data.update(vault_data)
        
        # 2. Load from environment file
        env_data = self._load_from_env_file()
        if env_data:
            config_data.update(env_data)
        
        # 3. Load from configuration file
        file_data = self._load_from_config_file()
        if file_data:
            config_data.update(file_data)
        
        # 4. Set service name if not provided
        if 'service_name' not in config_data:
            config_data['service_name'] = self.service_name
        
        try:
            # 5. Create configuration instance (environment variables are handled by Pydantic)
            config = config_class(**config_data)
            
            # 6. Validate service-specific configuration
            config.validate_config()
            
            return config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration for {self.service_name}: {e}") from e
    
    def _load_from_vault(self, vault_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration from HashiCorp Vault."""
        try:
            if not self._vault_client:
                self._vault_client = VaultClient(role=self.vault_role)
            
            data = self._vault_client.read_secret_data(vault_path)
            return data
            
        except Exception as e:
            # Vault errors are logged but not fatal - fall back to other sources
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load configuration from Vault: {e}")
            return None
    
    def _load_from_env_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from .env file."""
        env_files = [
            f".env.{self.service_name}",
            f".env.{os.getenv('ENVIRONMENT', 'development')}",
            ".env.local",
            ".env"
        ]
        
        for env_file in env_files:
            if os.path.exists(env_file):
                try:
                    config_data = {}
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                config_data[key.lower()] = value.strip('"\'')
                    return config_data
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load configuration from {env_file}: {e}")
        
        return None
    
    def _load_from_config_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from config file."""
        config_files = [
            f"config/{self.service_name}.yaml",
            f"config/{self.service_name}.yml",
            f"config/{self.service_name}.json",
            f"{self.service_name}/config.yaml",
            f"{self.service_name}/config.yml",
            f"{self.service_name}/config.json",
            "config.yaml",
            "config.yml",
            "config.json"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    return self._load_config_file(config_file)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to load configuration from {config_file}: {e}")
        
        return None
    
    def _load_config_file(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from a specific file."""
        path = Path(filepath)
        
        with open(filepath, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                raise ConfigError(f"Unsupported configuration file format: {path.suffix}")
    
    @classmethod
    def load_service_config(cls, service_name: str, config_class: Type[T], **kwargs) -> T:
        """
        Convenience method to load configuration for a service.
        
        Args:
            service_name: Name of the service
            config_class: Configuration class to instantiate
            **kwargs: Additional arguments for ConfigLoader
            
        Returns:
            Configured instance of config_class
        """
        loader = cls(service_name, **kwargs)
        return loader.load_config(config_class)