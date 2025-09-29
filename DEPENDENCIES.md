# Q2 Platform Dependencies Guide

This document explains the dependency management setup for the Q2 Platform using `pyproject.toml`.

## Overview

The Q2 Platform uses a comprehensive `pyproject.toml` file that consolidates all dependencies required across the platform's microservices and components.

## Installation Options

### Basic Installation (Core Dependencies Only)
```bash
pip install -e .
```
This installs the 50 core dependencies required for basic platform functionality.

### Development Installation
```bash
pip install -e ".[dev]"
```
This includes all core dependencies plus development tools:
- Testing frameworks (pytest, pytest-cov, pytest-asyncio, pytest-mock)
- Code quality tools (black, isort, flake8, ruff, mypy)
- Security tools (bandit, safety)

### Full Development Installation
```bash
pip install -e ".[dev,spark,integration]"
```
This includes dev tools plus big data processing and integration testing capabilities.

### Using with Constraints
For consistent versions across environments:
```bash
pip install -e ".[dev]" -c constraints.txt
```

## Dependency Groups

### Core Dependencies (50 packages)
- **API Framework**: FastAPI, Uvicorn, Pydantic, WebSockets
- **HTTP & Config**: httpx, python-dotenv, PyYAML, Jinja2
- **Messaging**: pulsar-client, fastavro
- **Databases**: pymilvus, cassandra-driver, elasticsearch, duckdb, pyignite, gremlinpython
- **Authentication**: python-jose, PyJWT, bcrypt, cryptography, hvac
- **AI/ML**: torch, sentence-transformers, langchain, scikit-learn, scipy, networkx
- **Observability**: OpenTelemetry suite, Prometheus, structlog
- **Infrastructure**: kubernetes, PyGithub

### Optional Dependencies

#### `dev` (13 packages)
Development and testing tools including pytest, mypy, black, bandit, etc.

#### `spark` (3 packages)
Big data processing with PySpark, Delta Lake, and pandas.

#### `quantum` (commented)
Quantum computing libraries (qiskit, cirq, pennylane) - install separately due to size.

#### `neuromorphic` (commented)
Neuromorphic computing libraries (brian2, nengo, nest-simulator) - install separately.

#### `integration` (2 packages)
Integration testing tools (docker-compose, tenacity).

## Makefile Integration

The existing Makefile targets work with the new dependency structure:

```bash
# Install development dependencies
make setup-dev

# Check for missing dependencies
make check-deps

# Install service-specific dependencies
make install-deps
```

## Shared Libraries

The platform includes multiple shared libraries that are installed separately in development:

- `q_messaging_schemas`: Pulsar messaging schemas
- `q_vectorstore_client`: VectorStore service client
- `q_knowledgegraph_client`: Knowledge Graph service client
- `q_pulse_client`: QuantumPulse service client
- `q_auth_parser`: Authentication token parser
- `q_observability`: Observability utilities
- And more...

These are installed via `-e` flag during development setup.

## Version Management

- Core dependencies use exact version pinning (==) from `constraints.txt`
- Development tools use minimum versions (>=) for flexibility
- All versions are coordinated with the centralized `constraints.txt` file

## Service-Specific Requirements

Individual services still maintain their own `requirements.txt` files for service-specific needs, but these should reference the core dependencies defined in `pyproject.toml`.

## Upgrading Dependencies

1. Update versions in `constraints.txt` first
2. Update corresponding versions in `pyproject.toml`
3. Test across all services
4. Update individual service requirements if needed

## CI/CD Integration  

The CI/CD pipeline uses:
```bash
pip install -e ".[dev]" -c constraints.txt
```

This ensures consistent versions and includes all tools needed for testing, linting, and security scanning.