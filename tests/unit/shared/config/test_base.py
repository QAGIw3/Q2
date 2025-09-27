"""
Unit tests for shared configuration base classes.
"""

import pytest
from pydantic import ValidationError
from shared.config.base import BaseConfig, ConfigError, DatabaseConfig, PulsarConfig, ObservabilityConfig


class TestServiceConfig(BaseConfig):
    """Test configuration class."""
    
    def validate_config(self):
        """Test validation method."""
        if self.service_name == "invalid":
            raise ConfigError("Invalid service name")


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
    
    def test_get_url_custom_driver(self):
        """Test database URL generation with custom driver."""
        config = DatabaseConfig(
            host="localhost",
            port=3306,
            database="testdb",
            username="testuser",
            password="testpass"
        )
        
        url = config.get_url("mysql")
        expected = "mysql://testuser:testpass@localhost:3306/testdb"
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


class TestBaseConfig:
    """Test BaseConfig functionality."""
    
    def test_base_config_creation(self):
        """Test basic configuration creation."""
        config = TestServiceConfig(service_name="test-service")
        
        assert config.service_name == "test-service"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ['development', 'testing', 'staging', 'production']:
            config = TestServiceConfig(service_name="test", environment=env)
            assert config.environment == env
        
        # Invalid environment
        with pytest.raises(ValidationError) as exc_info:
            TestServiceConfig(service_name="test", environment="invalid")
        
        assert "Environment must be one of" in str(exc_info.value)
    
    def test_service_name_in_observability(self):
        """Test that service name is set in observability config."""
        config = TestServiceConfig(service_name="test-service")
        
        assert config.observability.service_name == "test-service"
    
    def test_is_production(self):
        """Test production environment detection."""
        config = TestServiceConfig(service_name="test", environment="production")
        assert config.is_production() is True
        
        config = TestServiceConfig(service_name="test", environment="development")
        assert config.is_production() is False
    
    def test_is_development(self):
        """Test development environment detection."""
        config = TestServiceConfig(service_name="test", environment="development")
        assert config.is_development() is True
        
        config = TestServiceConfig(service_name="test", environment="production")
        assert config.is_development() is False
    
    def test_get_service_url(self):
        """Test service URL retrieval."""
        config = TestServiceConfig(
            service_name="test",
            services={
                "other-service": "http://localhost:8001",
                "another-service": "http://localhost:8002"
            }
        )
        
        assert config.get_service_url("other-service") == "http://localhost:8001"
        assert config.get_service_url("another-service") == "http://localhost:8002"
        assert config.get_service_url("non-existent") is None
    
    def test_dict_for_logging(self):
        """Test dictionary generation for logging."""
        config = TestServiceConfig(
            service_name="test",
            database=DatabaseConfig(password="secret"),
            vault={"token": "secret-token"}
        )
        
        log_dict = config.dict_for_logging()
        
        # Sensitive data should be masked
        assert log_dict["database"]["password"] == "***"
        # Other data should be preserved
        assert log_dict["service_name"] == "test"
        assert log_dict["database"]["host"] == "localhost"
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = TestServiceConfig(service_name="valid-service")
        # Should not raise an exception
        config.validate_config()
    
    def test_validate_config_failure(self):
        """Test configuration validation failure."""
        config = TestServiceConfig(service_name="invalid")
        
        with pytest.raises(ConfigError) as exc_info:
            config.validate_config()
        
        assert "Invalid service name" in str(exc_info.value)
    
    def test_nested_config_objects(self):
        """Test that nested config objects are properly created."""
        config = TestServiceConfig(service_name="test")
        
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.pulsar, PulsarConfig)
        assert isinstance(config.observability, ObservabilityConfig)
    
    def test_config_with_env_vars(self, monkeypatch):
        """Test configuration with environment variables."""
        # Set environment variables
        monkeypatch.setenv("SERVICE_NAME", "env-test")
        monkeypatch.setenv("VERSION", "2.0.0")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("API_PORT", "9000")
        
        config = TestServiceConfig()
        
        assert config.service_name == "env-test"
        assert config.version == "2.0.0"
        assert config.debug is True
        assert config.api_port == 9000