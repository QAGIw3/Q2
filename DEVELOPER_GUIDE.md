# Q2 Platform Developer Guide

## Welcome to Q2 Platform Development! 🚀

This guide will help you get up and running with the Q2 Platform development environment quickly and efficiently.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.11+** (Python 3.12 recommended)
- **Docker** and **Docker Compose**
- **Node.js 18+** and **npm** (for WebAppQ)
- **Git**
- **Make** (usually pre-installed on Linux/Mac)

### Optional but Recommended:
- **VS Code** with Python, Docker, and Kubernetes extensions
- **kubectl** for Kubernetes development
- **Helm** for infrastructure management

## 🏗️ Architecture Overview

Q2 Platform is a comprehensive microservices architecture with the following core services:

### Core Services:
- **agentQ**: AI reasoning engine with ReAct loop architecture
- **managerQ**: Service orchestration and management
- **VectorStoreQ**: Vector database operations
- **KnowledgeGraphQ**: Graph database and knowledge management
- **AuthQ**: Authentication and authorization
- **H2M**: Human-to-machine communication interface
- **WebAppQ**: Web UI built with React

### Supporting Services:
- **AgentSandbox**: Secure execution environment for agents
- **QuantumPulse**: Quantum computing experiments
- **UserProfileQ**: User profile management
- **IntegrationHub**: External service integrations
- **WorkflowEngine** & **WorkflowWorker**: Workflow orchestration

### Infrastructure:
- **Apache Pulsar**: Event streaming
- **Milvus**: Vector database
- **JanusGraph**: Knowledge graph database
- **Keycloak**: Identity management
- **Kubernetes**: Container orchestration

## 🛠️ Quick Setup

### 1. Clone and Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/QAGIw3/Q2.git
cd Q2

# Setup development environment (installs all necessary tools)
make dev-setup

# Install dependencies for all services
make install-deps
```

### 2. Verify Installation

```bash
# Check that all tools are properly installed
make check-deps

# Run code quality checks
make dev-check
```

### 3. Start Development Services

```bash
# Option 1: Start individual services
make serve-agentq    # Start AgentQ service
make serve-managerq  # Start ManagerQ service

# Option 2: Use Docker Compose for full stack
docker-compose up -d
```

## 📂 Project Structure

```
Q2/
├── agentQ/                 # AI reasoning engine
├── managerQ/               # Service orchestration
├── VectorStoreQ/           # Vector database service
├── KnowledgeGraphQ/        # Knowledge graph service
├── AuthQ/                  # Authentication service
├── H2M/                    # Human-machine interface
├── WebAppQ/                # React web application
├── AgentSandbox/           # Secure execution environment
├── shared/                 # Shared libraries and utilities
├── infra/                  # Infrastructure as Code (Terraform/K8s)
├── scripts/                # Utility scripts
├── tests/                  # Integration and end-to-end tests
├── workflows/              # GitHub Actions workflows
├── airflow/                # Airflow DAGs and configurations
├── Makefile               # Development commands
├── pyproject.toml         # Python project configuration
├── constraints.txt        # Python dependency constraints
└── docker-compose.yml     # Local development stack
```

## 🔧 Development Workflow

### Daily Development Commands

```bash
# Format your code
make format

# Run linting
make lint

# Run tests
make test

# Run all quality checks (what CI runs)
make dev-check

# Fix common issues automatically
make dev-fix
```

### Working with Services

Each service follows a consistent structure:
```
service_name/
├── app/                   # Application code
├── tests/                 # Service-specific tests
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── README.md            # Service documentation
└── setup.py             # Package setup
```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards:
   - Python: Follow PEP 8, use Black formatting (120 char line length)
   - Use type hints where possible
   - Write tests for new functionality
   - Update documentation

3. **Test your changes**:
   ```bash
   make dev-check  # Run all quality checks
   make test       # Run tests
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   git push origin feature/your-feature-name
   ```

## 🧪 Testing

### Test Structure
- **Unit tests**: `tests/unit/` - Fast, isolated tests
- **Integration tests**: `tests/integration/` - Cross-service tests
- **Service tests**: Each service has its own `tests/` directory

### Running Tests

```bash
make test              # Run all tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-coverage     # Generate coverage report
```

### Writing Tests

- Use `pytest` for Python tests
- Follow the existing test patterns in each service
- Aim for >80% code coverage
- Write integration tests for cross-service functionality

## 🐛 Debugging

### Common Issues and Solutions

#### Development Tools Not Found
```bash
# If you get "command not found" errors:
make setup-dev
```

#### Dependency Issues
```bash
# Clean and reinstall dependencies:
make clean
make install-deps
```

#### Docker Issues
```bash
# Clean Docker resources:
make clean-docker
docker system prune -a
```

### Debug Individual Services

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run service with debugger
cd agentQ
python -m pdb -m agentQ.app.main
```

## 📚 Documentation

### API Documentation
- Each service exposes OpenAPI/Swagger docs at `/docs`
- Visit `http://localhost:PORT/docs` when running services

### Service Documentation
- Each service has comprehensive README.md
- Architecture decisions are documented in ADRs (coming soon)

## 🛡️ Security

### Pre-commit Hooks
Pre-commit hooks run automatically and include:
- Code formatting (Black, isort)
- Linting (flake8)
- Security scanning (bandit)
- YAML/JSON validation

### Security Scanning
```bash
make security-scan  # Run bandit security scanner
make deps-check     # Check for vulnerable dependencies
```

## 🚀 Deployment

### Local Development
```bash
# Start full stack locally
docker-compose up -d

# Or start services individually
make serve-agentq
```

### Kubernetes Development
```bash
# Deploy to local K8s cluster
cd infra/terraform
terraform init
terraform apply
```

## 💡 Tips and Best Practices

### Code Quality
- Run `make dev-fix` before committing to fix common issues
- Use the provided VS Code settings for consistent formatting
- Write descriptive commit messages following conventional commits

### Performance
- Use async/await for I/O operations
- Profile code with `cProfile` for performance bottlenecks
- Monitor service metrics via Grafana dashboards

### Debugging
- Use structured logging with `structlog`
- Add correlation IDs for tracing requests across services
- Use OpenTelemetry for distributed tracing

## 🆘 Getting Help

### Resources
- **Architecture docs**: Check individual service READMEs
- **API docs**: Visit `/docs` endpoint on each service
- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub Discussions for questions

### Common Patterns
- **Error handling**: Use shared error handling patterns in `shared/error_handling/`
- **Logging**: Use structured logging with correlation IDs
- **Configuration**: Use Pydantic models for config validation
- **Testing**: Follow pytest patterns in existing tests

## 🔄 Contributing

1. Read this guide thoroughly
2. Set up your development environment
3. Pick an issue or discuss new features
4. Follow the development workflow
5. Submit a pull request with tests and documentation

---

Welcome to the Q2 Platform development team! 🎉

For more specific information, check the README.md files in individual service directories.