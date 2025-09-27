"""
Shared pytest configuration and fixtures for Q2 Platform tests.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Generator
import uuid
import json

# Test configuration constants
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_PULSAR_URL = "pulsar://localhost:16650"  # Different port for testing
TEST_IGNITE_ADDRESSES = ["127.0.0.1:10801"]  # Different port for testing


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Provides a test configuration that can be used across all services.
    """
    return {
        "environment": "testing",
        "debug": True,
        "database": {
            "url": TEST_DATABASE_URL,
            "pool_size": 1,
            "max_overflow": 0,
            "echo": False
        },
        "pulsar": {
            "service_url": TEST_PULSAR_URL,
            "tenant": "test",
            "namespace": "default",
            "connection_timeout": 5,
            "operation_timeout": 5
        },
        "ignite": {
            "addresses": TEST_IGNITE_ADDRESSES,
            "timeout": 5
        },
        "observability": {
            "logging_level": "DEBUG",
            "otlp_endpoint": None,  # Disable tracing in tests
            "metrics_enabled": False,
            "tracing_enabled": False,
            "service_name": "test-service"
        },
        "vault": {
            "url": "http://localhost:8200",
            "role": "test-role",
            "timeout": 5
        },
        "services": {
            "qpulse_url": "http://localhost:8002",
            "vectorstoreq_url": "http://localhost:8001",
            "knowledgegraphq_url": "http://localhost:8003",
            "managerq_url": "http://localhost:8004"
        }
    }


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Provides a temporary directory that's cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_vault_client():
    """
    Mock VaultClient for testing.
    """
    with patch('shared.vault_client.VaultClient') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.read_secret_data.return_value = {
            "database_url": TEST_DATABASE_URL,
            "pulsar_service_url": TEST_PULSAR_URL,
            "api_key": "test-api-key"
        }
        yield mock_instance


@pytest.fixture
def mock_pulsar_client():
    """
    Mock Pulsar client for testing.
    """
    with patch('pulsar.Client') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        
        # Mock producer
        mock_producer = MagicMock()
        mock_instance.create_producer.return_value = mock_producer
        
        # Mock consumer
        mock_consumer = MagicMock()
        mock_instance.subscribe.return_value = mock_consumer
        
        yield mock_instance


@pytest.fixture
def mock_ignite_client():
    """
    Mock Apache Ignite client for testing.
    """
    with patch('pyignite.Client') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.connect.return_value = True
        
        # Mock cache operations
        mock_cache = MagicMock()
        mock_instance.get_or_create_cache.return_value = mock_cache
        
        yield mock_instance


@pytest.fixture
def mock_qpulse_client():
    """
    Mock QuantumPulse client for testing.
    """
    with patch('shared.q_pulse_client.client.QuantumPulseClient') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        
        # Mock chat response
        mock_instance.chat.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response from mock QPulse"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        yield mock_instance


@pytest.fixture
def sample_workflow_task():
    """
    Provides a sample WorkflowTask for testing.
    """
    return {
        "task_id": f"test_task_{uuid.uuid4()}",
        "type": "task",
        "agent_personality": "default",
        "prompt": "This is a test task",
        "dependencies": [],
        "status": "PENDING"
    }


@pytest.fixture
def sample_workflow():
    """
    Provides a sample Workflow for testing.
    """
    task_id = f"test_task_{uuid.uuid4()}"
    return {
        "workflow_id": f"test_workflow_{uuid.uuid4()}",
        "original_prompt": "Test workflow prompt",
        "status": "RUNNING",
        "tasks": [
            {
                "task_id": task_id,
                "type": "task",
                "agent_personality": "default",
                "prompt": "Test task prompt",
                "dependencies": [],
                "status": "PENDING"
            }
        ],
        "shared_context": {}
    }


@pytest.fixture
def sample_agent_config():
    """
    Provides a sample agent configuration for testing.
    """
    return {
        "agent_id": f"test_agent_{uuid.uuid4()}",
        "personality": "default",
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "tools": [
            "human_tool",
            "vectorstore_tool",
            "knowledgegraph_tool"
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Automatically sets up test environment variables.
    """
    # Set test environment variables
    test_env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "TESTING": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "PULSAR_SERVICE_URL": TEST_PULSAR_URL,
        "DISABLE_TELEMETRY": "true",
        "LOG_LEVEL": "DEBUG"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def mock_logger():
    """
    Mock logger for testing.
    """
    with patch('structlog.get_logger') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


# Marks for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.smoke = pytest.mark.smoke


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location and naming.
    """
    for item in items:
        # Mark tests in integration directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark tests that might be slow
        elif any(keyword in item.name.lower() for keyword in ["slow", "performance", "load"]):
            item.add_marker(pytest.mark.slow)
        # Default to unit tests
        else:
            item.add_marker(pytest.mark.unit)


def pytest_configure(config):
    """
    Configure pytest with custom markers.
    """
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds or minutes)"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests (basic functionality)"
    )