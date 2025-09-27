"""
Simple unit tests for shared configuration base classes.
"""

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from shared.config.base import DatabaseConfig, PulsarConfig, ObservabilityConfig, ConfigError


class TestDatabaseConfig:
    """Test DatabaseConfig functionality."""
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "q2"
        assert config.username == "q2user"
        assert config.password == ""
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.echo is False
    
    def test_database_config_custom_values(self):
        """Test DatabaseConfig with custom values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=3306,
            database="testdb",
            username="testuser",
            password="testpass",
            pool_size=5,
            max_overflow=10,
            echo=True
        )
        
        assert config.host == "db.example.com"
        assert config.port == 3306
        assert config.database == "testdb"
        assert config.username == "testuser"
        assert config.password == "testpass"
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.echo is True
    
    def test_get_url_with_password(self):
        """Test database URL generation with password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        url = config.get_url()
        expected = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert url == expected
    
    def test_get_url_without_password(self):
        """Test database URL generation without password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,  
            database="testdb",
            username="testuser"
        )
        
        url = config.get_url()
        expected = "postgresql://testuser@localhost:5432/testdb"
        assert url == expected


class TestPulsarConfig:
    """Test PulsarConfig functionality."""
    
    def test_pulsar_config_defaults(self):
        """Test PulsarConfig with default values."""
        config = PulsarConfig()
        
        assert config.service_url == "pulsar://localhost:6650"
        assert config.admin_url == "http://localhost:8080"
        assert config.tenant == "public"
        assert config.namespace == "default"
        assert config.connection_timeout == 30
        assert config.operation_timeout == 30
    
    def test_get_topic_name(self):
        """Test topic name generation."""
        config = PulsarConfig(tenant="test", namespace="dev")
        
        topic_name = config.get_topic_name("my-topic")
        expected = "persistent://test/dev/my-topic"
        assert topic_name == expected


class TestObservabilityConfig:
    """Test ObservabilityConfig functionality."""
    
    def test_observability_config_defaults(self):
        """Test ObservabilityConfig with default values."""
        config = ObservabilityConfig()
        
        assert config.logging_level == "INFO"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True
        assert config.service_name == ""


class TestConfigError:
    """Test ConfigError exception."""
    
    def test_config_error_creation(self):
        """Test ConfigError can be created and raised."""
        with pytest.raises(ConfigError) as exc_info:
            raise ConfigError("Test error message")
        
        assert "Test error message" in str(exc_info.value)