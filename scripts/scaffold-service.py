#!/usr/bin/env python3
"""
Q2 Platform Service Scaffolding Tool

This script generates boilerplate code for new services following Q2 Platform conventions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List


class ServiceScaffolder:
    """Generates scaffolding for new Q2 Platform services."""
    
    def __init__(self, service_name: str, service_type: str = "api"):
        self.service_name = service_name
        self.service_type = service_type
        self.service_dir = Path(service_name)
        
        # Validate service name
        if not service_name.isalnum():
            raise ValueError("Service name must be alphanumeric")
        
        if self.service_dir.exists():
            raise ValueError(f"Service directory {service_name} already exists")
    
    def create_directory_structure(self) -> None:
        """Create the standard directory structure for a service."""
        directories = [
            self.service_dir,
            self.service_dir / "app",
            self.service_dir / "app" / "api",
            self.service_dir / "app" / "models",
            self.service_dir / "app" / "services",
            self.service_dir / "app" / "utils",
            self.service_dir / "tests",
            self.service_dir / "tests" / "unit",
            self.service_dir / "tests" / "integration",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile for the service."""
        return f"""# {self.service_name} Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r {self.service_name.lower()} && useradd -r -g {self.service_name.lower()} {self.service_name.lower()}
RUN chown -R {self.service_name.lower()}:{self.service_name.lower()} /app
USER {self.service_name.lower()}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "{self.service_name.lower()}.app.main"]
"""
    
    def generate_requirements_txt(self) -> str:
        """Generate requirements.txt for the service."""
        return """# {service_name} Service Dependencies

# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# HTTP client
httpx>=0.25.0

# Logging and observability
structlog>=23.2.0
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
opentelemetry-instrumentation-httpx>=0.42b0

# Messaging (Pulsar)
pulsar-client>=3.3.0

# Data validation and serialization
pydantic>=2.5.0

# Configuration
python-multipart>=0.0.6

# Testing dependencies (for development)
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.12.0
httpx>=0.25.0  # For testing async clients

# Development dependencies
black>=23.0.0
isort>=5.0.0
flake8>=6.0.0
mypy>=1.0.0
""".format(service_name=self.service_name)
    
    def generate_main_py(self) -> str:
        """Generate main.py for the service."""
        return f'''"""
{self.service_name} Service Main Module

This module initializes and runs the {self.service_name} service.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from {self.service_name.lower()}.app.api.health import router as health_router
from {self.service_name.lower()}.app.api.v1 import router as api_v1_router
from {self.service_name.lower()}.app.config import get_settings
from {self.service_name.lower()}.app.utils.logging import setup_logging


# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(log_level=settings.log_level)
logger = structlog.get_logger(__name__)


def setup_tracing() -> None:
    """Setup OpenTelemetry tracing."""
    resource = Resource(attributes={{
        SERVICE_NAME: "{self.service_name.lower()}"
    }})
    
    provider = TracerProvider(resource=resource)
    
    # Add Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.jaeger_host,
        agent_port=settings.jaeger_port,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)
    
    trace.set_tracer_provider(provider)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting {self.service_name} service", service="{self.service_name.lower()}")
    
    # Setup tracing if enabled
    if settings.tracing_enabled:
        setup_tracing()
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
    
    # Startup logic here
    yield
    
    # Shutdown logic here
    logger.info("Shutting down {self.service_name} service", service="{self.service_name.lower()}")


# Create FastAPI application
app = FastAPI(
    title="{self.service_name} API",
    description="{self.service_name} service for Q2 Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Include routers
app.include_router(health_router)
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {{
        "service": "{self.service_name.lower()}",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }}


async def main():
    """Main function to run the service."""
    try:
        config = uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            log_config=None,  # We handle logging ourselves
            access_log=False,
        )
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        logger.error("Failed to start service", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def generate_config_py(self) -> str:
        """Generate config.py for the service."""
        return f'''"""
{self.service_name} Service Configuration

This module defines configuration settings for the {self.service_name} service.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Service configuration
    service_name: str = Field(default="{self.service_name.lower()}", description="Service name")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Log level")
    
    # Tracing configuration
    tracing_enabled: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    jaeger_host: str = Field(default="localhost", description="Jaeger host")
    jaeger_port: int = Field(default=14268, description="Jaeger port")
    
    # Database configuration (if needed)
    database_url: Optional[str] = Field(default=None, description="Database URL")
    
    # Pulsar configuration
    pulsar_url: str = Field(default="pulsar://localhost:6650", description="Pulsar broker URL")
    
    # External service URLs
    agent_q_url: str = Field(default="http://localhost:8000", description="AgentQ service URL")
    manager_q_url: str = Field(default="http://localhost:8001", description="ManagerQ service URL")
    
    # Authentication
    keycloak_url: str = Field(default="http://localhost:8080", description="Keycloak URL")
    keycloak_realm: str = Field(default="q-platform", description="Keycloak realm")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
'''
    
    def generate_health_router(self) -> str:
        """Generate health check router."""
        return f'''"""
{self.service_name} Service Health Check API

This module defines health check endpoints for the {self.service_name} service.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel

from {self.service_name.lower()}.app.config import get_settings


router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    """Health check response model."""
    service: str
    status: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["health"],
    summary="Health check",
    description="Check if the service is healthy and running"
)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        service=settings.service_name,
        status="healthy",
        version="1.0.0"
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["health"],
    summary="Readiness check",
    description="Check if the service is ready to accept requests"
)
async def readiness_check() -> HealthResponse:
    """Readiness check endpoint."""
    # Add any readiness checks here (database connections, etc.)
    return HealthResponse(
        service=settings.service_name,
        status="ready",
        version="1.0.0"
    )
'''
    
    def generate_api_v1_router(self) -> str:
        """Generate API v1 router."""
        return f'''"""
{self.service_name} Service API v1 Router

This module defines the main API endpoints for the {self.service_name} service.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from {self.service_name.lower()}.app.services.{self.service_name.lower()}_service import {self.service_name}Service


router = APIRouter()


class ExampleRequest(BaseModel):
    """Example request model."""
    name: str
    description: str


class ExampleResponse(BaseModel):
    """Example response model."""
    id: str
    name: str
    description: str
    status: str


@router.get(
    "/status",
    response_model={{"status": str, "service": str}},
    tags=["{self.service_name.lower()}"],
    summary="Get service status"
)
async def get_status():
    """Get service status."""
    return {{
        "status": "operational",
        "service": "{self.service_name.lower()}"
    }}


@router.post(
    "/example",
    response_model=ExampleResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["{self.service_name.lower()}"],
    summary="Create example resource"
)
async def create_example(request: ExampleRequest) -> ExampleResponse:
    """Create an example resource."""
    service = {self.service_name}Service()
    
    try:
        result = await service.create_example(request.name, request.description)
        return ExampleResponse(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            status="created"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create example: {{str(e)}}"
        )


@router.get(
    "/example/{{example_id}}",
    response_model=ExampleResponse,
    tags=["{self.service_name.lower()}"],
    summary="Get example resource"
)
async def get_example(example_id: str) -> ExampleResponse:
    """Get an example resource by ID."""
    service = {self.service_name}Service()
    
    try:
        result = await service.get_example(example_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Example not found"
            )
        
        return ExampleResponse(
            id=result["id"],
            name=result["name"],
            description=result["description"],
            status="active"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get example: {{str(e)}}"
        )
'''
    
    def generate_service_class(self) -> str:
        """Generate service class."""
        return f'''"""
{self.service_name} Service Implementation

This module contains the core business logic for the {self.service_name} service.
"""

import uuid
from typing import Dict, Optional

import structlog

from {self.service_name.lower()}.app.config import get_settings


logger = structlog.get_logger(__name__)
settings = get_settings()


class {self.service_name}Service:
    """Core service implementation for {self.service_name}."""
    
    def __init__(self):
        """Initialize the service."""
        self.examples: Dict[str, Dict] = {{}}  # In-memory storage for demo
        logger.info("Initialized {self.service_name} service")
    
    async def create_example(self, name: str, description: str) -> Dict:
        """Create an example resource."""
        example_id = str(uuid.uuid4())
        
        example = {{
            "id": example_id,
            "name": name,
            "description": description,
            "created_at": "2024-01-01T00:00:00Z"  # Use proper timestamp
        }}
        
        self.examples[example_id] = example
        
        logger.info(
            "Created example resource",
            example_id=example_id,
            name=name
        )
        
        return example
    
    async def get_example(self, example_id: str) -> Optional[Dict]:
        """Get an example resource by ID."""
        example = self.examples.get(example_id)
        
        if example:
            logger.info("Retrieved example resource", example_id=example_id)
        else:
            logger.warning("Example resource not found", example_id=example_id)
        
        return example
    
    async def list_examples(self) -> list[Dict]:
        """List all example resources."""
        examples = list(self.examples.values())
        logger.info("Listed example resources", count=len(examples))
        return examples
    
    async def delete_example(self, example_id: str) -> bool:
        """Delete an example resource."""
        if example_id in self.examples:
            del self.examples[example_id]
            logger.info("Deleted example resource", example_id=example_id)
            return True
        else:
            logger.warning("Example resource not found for deletion", example_id=example_id)
            return False
'''
    
    def generate_logging_utils(self) -> str:
        """Generate logging utilities."""
        return f'''"""
{self.service_name} Service Logging Configuration

This module configures structured logging for the {self.service_name} service.
"""

import sys
from typing import Any, Dict

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured logging configuration."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_level == "DEBUG" 
            else structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "logger", "event"])
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )


def get_correlation_id() -> str:
    """Get correlation ID for request tracing."""
    import uuid
    return str(uuid.uuid4())


class LoggingMixin:
    """Mixin class to add structured logging to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = structlog.get_logger(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
    
    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log an event with additional context."""
        self.logger.info(event, **kwargs)
    
    def log_error(self, event: str, error: Exception, **kwargs: Any) -> None:
        """Log an error with additional context."""
        self.logger.error(
            event,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs,
            exc_info=True
        )
'''
    
    def generate_init_files(self) -> Dict[str, str]:
        """Generate __init__.py files for all packages."""
        return {
            "app/__init__.py": f'"""\\n{self.service_name} Service Application Package\\n"""\\n',
            "app/api/__init__.py": f'"""\\n{self.service_name} Service API Package\\n"""\\n',
            "app/models/__init__.py": f'"""\\n{self.service_name} Service Models Package\\n"""\\n',
            "app/services/__init__.py": f'"""\\n{self.service_name} Service Business Logic Package\\n"""\\n',
            "app/utils/__init__.py": f'"""\\n{self.service_name} Service Utilities Package\\n"""\\n',
            "tests/__init__.py": f'"""\\n{self.service_name} Service Tests Package\\n"""\\n',
            "tests/unit/__init__.py": f'"""\\n{self.service_name} Service Unit Tests Package\\n"""\\n',
            "tests/integration/__init__.py": f'"""\\n{self.service_name} Service Integration Tests Package\\n"""\\n',
        }
    
    def generate_test_files(self) -> Dict[str, str]:
        """Generate test files."""
        return {
            "tests/conftest.py": f'''"""
{self.service_name} Service Test Configuration

This module provides pytest fixtures and configuration for testing.
"""

import pytest
from fastapi.testclient import TestClient

from {self.service_name.lower()}.app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def service():
    """Create a service instance for testing."""
    from {self.service_name.lower()}.app.services.{self.service_name.lower()}_service import {self.service_name}Service
    return {self.service_name}Service()
''',
            "tests/unit/test_service.py": f'''"""
{self.service_name} Service Unit Tests

This module contains unit tests for the {self.service_name} service.
"""

import pytest

from {self.service_name.lower()}.app.services.{self.service_name.lower()}_service import {self.service_name}Service


@pytest.mark.asyncio
async def test_create_example():
    """Test creating an example resource."""
    service = {self.service_name}Service()
    
    result = await service.create_example("test", "test description")
    
    assert result["name"] == "test"
    assert result["description"] == "test description"
    assert "id" in result


@pytest.mark.asyncio
async def test_get_example():
    """Test getting an example resource."""
    service = {self.service_name}Service()
    
    # Create an example first
    created = await service.create_example("test", "test description")
    example_id = created["id"]
    
    # Get the example
    result = await service.get_example(example_id)
    
    assert result is not None
    assert result["id"] == example_id
    assert result["name"] == "test"


@pytest.mark.asyncio
async def test_get_nonexistent_example():
    """Test getting a non-existent example resource."""
    service = {self.service_name}Service()
    
    result = await service.get_example("nonexistent-id")
    
    assert result is None
''',
            "tests/integration/test_api.py": f'''"""
{self.service_name} Service API Integration Tests

This module contains integration tests for the {self.service_name} service API.
"""

import pytest
from fastapi import status


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["service"] == "{self.service_name.lower()}"
    assert data["status"] == "healthy"


def test_readiness_check(client):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["service"] == "{self.service_name.lower()}"
    assert data["status"] == "ready"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["service"] == "{self.service_name.lower()}"
    assert data["status"] == "running"


def test_api_status(client):
    """Test API status endpoint."""
    response = client.get("/api/v1/status")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "operational"
    assert data["service"] == "{self.service_name.lower()}"


def test_create_example(client):
    """Test creating an example resource via API."""
    payload = {{
        "name": "test example",
        "description": "test description"
    }}
    
    response = client.post("/api/v1/example", json=payload)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["description"] == payload["description"]
    assert "id" in data
'''
        }
    
    def generate_readme(self) -> str:
        """Generate README.md for the service."""
        return f"""# {self.service_name} Service

## Overview

{self.service_name} is a microservice in the Q2 Platform architecture. This service provides [describe the main functionality here].

## Features

- RESTful API with FastAPI
- OpenAPI/Swagger documentation
- Health and readiness checks
- Structured logging with correlation IDs
- OpenTelemetry tracing
- Comprehensive testing
- Docker containerization

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Service

#### Development Mode

```bash
# Run directly
python -m {self.service_name.lower()}.app.main

# Or use the project Makefile
make serve-{self.service_name.lower()}
```

#### Docker

```bash
# Build the image
docker build -t q2/{self.service_name.lower()}:latest .

# Run the container
docker run -p 8000:8000 q2/{self.service_name.lower()}:latest
```

## API Documentation

Once the service is running, you can access:

- **Interactive API docs (Swagger)**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI specification**: http://localhost:8000/openapi.json

## Configuration

The service can be configured using environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Host to bind to | `0.0.0.0` |
| `PORT` | Port to bind to | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` |
| `PULSAR_URL` | Pulsar broker URL | `pulsar://localhost:6650` |

## Testing

Run tests using pytest:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov={self.service_name.lower()} --cov-report=html

# Run specific test types
python -m pytest tests/unit/      # Unit tests only
python -m pytest tests/integration/  # Integration tests only
```

## Development

### Code Quality

This project uses several tools to maintain code quality:

```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

### Project Structure

```
{self.service_name}/
├── app/
│   ├── api/                 # API routes and endpoints
│   ├── models/              # Data models and schemas
│   ├── services/            # Business logic
│   ├── utils/               # Utility functions
│   ├── config.py           # Configuration
│   └── main.py             # Application entry point
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test configuration
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Monitoring and Observability

The service includes:

- **Health checks**: Available at `/health` and `/ready`
- **Structured logging**: JSON logs with correlation IDs
- **OpenTelemetry tracing**: Distributed tracing support
- **Metrics**: Prometheus-compatible metrics (if enabled)

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation as needed
6. Submit a pull request

## License

This project is part of the Q2 Platform and follows the project's licensing terms.
"""
    
    def generate_env_example(self) -> str:
        """Generate .env.example file."""
        return f"""# {self.service_name} Service Environment Configuration
# Copy this file to .env and customize for your environment

# Service configuration
HOST=0.0.0.0
PORT=8000
SERVICE_NAME={self.service_name.lower()}

# Logging
LOG_LEVEL=INFO

# Tracing
TRACING_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=14268

# Database (if needed)
# DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Pulsar messaging
PULSAR_URL=pulsar://localhost:6650

# External services
AGENT_Q_URL=http://localhost:8000
MANAGER_Q_URL=http://localhost:8001

# Authentication
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=q-platform
"""
    
    def generate_service(self) -> None:
        """Generate the complete service scaffolding."""
        print(f"Generating {self.service_name} service scaffolding...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Generate main files
        files_to_create = {
            "Dockerfile": self.generate_dockerfile(),
            "requirements.txt": self.generate_requirements_txt(),
            "README.md": self.generate_readme(),
            ".env.example": self.generate_env_example(),
            "app/main.py": self.generate_main_py(),
            "app/config.py": self.generate_config_py(),
            "app/api/health.py": self.generate_health_router(),
            "app/api/v1.py": self.generate_api_v1_router(),
            f"app/services/{self.service_name.lower()}_service.py": self.generate_service_class(),
            "app/utils/logging.py": self.generate_logging_utils(),
        }
        
        # Generate __init__.py files
        init_files = self.generate_init_files()
        files_to_create.update(init_files)
        
        # Generate test files
        test_files = self.generate_test_files()
        files_to_create.update(test_files)
        
        # Write all files
        for file_path, content in files_to_create.items():
            full_path = self.service_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            print(f"Created file: {full_path}")
        
        print(f"\\n✅ {self.service_name} service scaffolding complete!")
        print(f"\\nNext steps:")
        print(f"1. cd {self.service_name}")
        print(f"2. pip install -r requirements.txt")
        print(f"3. cp .env.example .env")
        print(f"4. python -m {self.service_name.lower()}.app.main")
        print(f"5. Visit http://localhost:8000/docs for API documentation")


def main():
    """Main function to run the scaffolding tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Q2 Platform service scaffolding")
    parser.add_argument("service_name", help="Name of the service to create (e.g., MyNewService)")
    parser.add_argument(
        "--type",
        choices=["api", "worker", "scheduler"],
        default="api",
        help="Type of service to generate"
    )
    
    args = parser.parse_args()
    
    try:
        scaffolder = ServiceScaffolder(args.service_name, args.type)
        scaffolder.generate_service()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()