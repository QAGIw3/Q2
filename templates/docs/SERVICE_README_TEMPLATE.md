# {SERVICE_NAME} Service

## Overview

{SERVICE_DESCRIPTION}

**Service Type:** {SERVICE_TYPE}  
**Port:** {SERVICE_PORT}  
**API Documentation:** http://localhost:{SERVICE_PORT}/docs  

## Architecture

### Responsibilities
- {RESPONSIBILITY_1}
- {RESPONSIBILITY_2}
- {RESPONSIBILITY_3}

### Key Features
- ✅ RESTful API with FastAPI
- ✅ OpenAPI/Swagger documentation  
- ✅ Health and readiness checks
- ✅ Structured logging with correlation IDs
- ✅ OpenTelemetry distributed tracing
- ✅ Comprehensive test coverage
- ✅ Docker containerization
- ✅ Kubernetes ready

### Dependencies
- **Internal Services:** {INTERNAL_DEPENDENCIES}
- **External Services:** {EXTERNAL_DEPENDENCIES}
- **Infrastructure:** {INFRASTRUCTURE_DEPENDENCIES}

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- Access to Q2 Platform infrastructure (Pulsar, Milvus, etc.)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the service:**
   ```bash
   python -m {SERVICE_MODULE}.app.main
   ```

4. **Verify it's running:**
   ```bash
   curl http://localhost:{SERVICE_PORT}/health
   ```

### Docker Deployment

```bash
# Build the image
docker build -t q2/{SERVICE_NAME_LOWER}:latest .

# Run the container
docker run -p {SERVICE_PORT}:{SERVICE_PORT} \
  -e PULSAR_URL=pulsar://pulsar:6650 \
  q2/{SERVICE_NAME_LOWER}:latest
```

### Kubernetes Deployment

```bash
# Apply the Kubernetes manifests
kubectl apply -f k8s/
```

## API Reference

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check |
| `GET` | `/ready` | Readiness check with dependencies |

### Core Endpoints

{API_ENDPOINTS_TABLE}

For complete API documentation, visit `/docs` when the service is running.

## Configuration

Configuration is managed via environment variables or `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HOST` | Host to bind to | `0.0.0.0` | No |
| `PORT` | Port to bind to | `{SERVICE_PORT}` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `PULSAR_URL` | Pulsar broker URL | `pulsar://localhost:6650` | Yes |
| {ADDITIONAL_CONFIG_VARS} | | | |

### Example Configuration

```bash
# .env file
HOST=0.0.0.0
PORT={SERVICE_PORT}
LOG_LEVEL=DEBUG
PULSAR_URL=pulsar://pulsar:6650
TRACING_ENABLED=true
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov={SERVICE_MODULE} --cov-report=html

# Run specific test types
pytest tests/unit/      # Unit tests
pytest tests/integration/  # Integration tests
```

### Code Quality

```bash
# Format code
make format

# Lint code  
make lint

# Security scan
make security-scan

# Run all quality checks
make dev-check
```

### Project Structure

```
{SERVICE_NAME}/
├── app/
│   ├── api/                 # API routes and endpoints
│   │   ├── v1/             # API version 1
│   │   └── health.py       # Health check endpoints
│   ├── models/             # Pydantic models and schemas
│   ├── services/           # Business logic layer
│   ├── utils/              # Utility functions
│   ├── config.py          # Configuration management
│   └── main.py            # Application entry point
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── fixtures/          # Test fixtures
│   └── conftest.py        # Test configuration
├── k8s/                   # Kubernetes manifests
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## Monitoring & Observability

### Health Checks
- **Health endpoint:** `/health` - Basic service health
- **Readiness endpoint:** `/ready` - Service ready to accept traffic
- **Liveness probe:** Kubernetes liveness probe configuration

### Logging
- Structured JSON logging with correlation IDs
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Request/response logging for all API calls

### Tracing
- OpenTelemetry distributed tracing
- Automatic instrumentation for FastAPI and HTTP clients
- Custom spans for business logic operations

### Metrics
- Prometheus metrics available at `/metrics`
- Custom business metrics: {CUSTOM_METRICS}
- Standard HTTP metrics (request count, duration, etc.)

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check if port is in use
lsof -i :{SERVICE_PORT}

# Check logs
python -m {SERVICE_MODULE}.app.main

# Verify dependencies
make validate-dev
```

#### Connection Issues
```bash
# Test service connectivity
curl -v http://localhost:{SERVICE_PORT}/health

# Check Pulsar connectivity
# (Add service-specific connectivity tests)
```

#### Performance Issues
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check resource usage
docker stats <container_id>

# Profile the application
# (Add service-specific profiling instructions)
```

### Debug Mode

Run the service in debug mode for development:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python -m {SERVICE_MODULE}.app.main
```

### Debugging Tools

```bash
# Debug all services
make debug-services

# Debug this service specifically  
make debug-service SERVICE={SERVICE_NAME_LOWER}

# Generate service diagnostics
python scripts/debug-service.py --service {SERVICE_NAME_LOWER} --output debug-report.json
```

## Contributing

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test:**
   ```bash
   make dev-check  # Run quality checks
   make test       # Run tests
   ```

3. **Update documentation:**
   - Update this README if adding new features
   - Update API documentation
   - Add/update tests

4. **Submit pull request:**
   - Ensure all tests pass
   - Include description of changes
   - Link to relevant issues

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Write docstrings for all public functions and classes
- Maintain test coverage above 80%
- Use structured logging with correlation IDs

## Deployment

### Production Checklist

- [ ] All tests passing
- [ ] Security scan clean
- [ ] Environment variables configured
- [ ] Health checks responding
- [ ] Monitoring/alerting configured
- [ ] Backup/recovery procedures tested

### Scaling Considerations

{SCALING_CONSIDERATIONS}

## Security

### Authentication
- Keycloak OIDC token validation
- Service-to-service authentication via mTLS
- API key authentication for external clients

### Authorization  
- Role-based access control (RBAC)
- Resource-level permissions
- Audit logging for all operations

### Data Protection
- Encryption in transit (TLS)
- Encryption at rest for sensitive data
- PII data handling compliance

## Support

### Getting Help
- Check the [Q2 Platform Developer Guide](../DEVELOPER_GUIDE.md)
- Search existing GitHub issues
- Create new issue with reproduction steps

### Runbooks
- {LINK_TO_RUNBOOK_1}
- {LINK_TO_RUNBOOK_2}

---

**Maintainers:** {MAINTAINERS}  
**Last Updated:** {LAST_UPDATED}  
**Version:** {SERVICE_VERSION}