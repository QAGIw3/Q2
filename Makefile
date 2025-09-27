# Q2 Platform Development Makefile

.PHONY: help install lint format test clean build security-scan deps-check pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "Q2 Platform Development Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m%-30s\033[0m %s\n", "Target", "Description"} /^[a-zA-Z_-]+:.*?##/ { printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup Commands

setup-dev: ## Setup development environment
	@echo "Setting up development environment..."
	@bash scripts/dev-setup.sh

install-deps: ## Install all service dependencies
	@echo "Installing dependencies for all services..."
	@for service in agentQ managerQ VectorStoreQ KnowledgeGraphQ QuantumPulse AuthQ; do \
		if [ -f $$service/requirements.txt ]; then \
			echo "Installing $$service dependencies..."; \
			pip install --user -r $$service/requirements.txt -c constraints.txt; \
		fi; \
	done

##@ Code Quality Commands

lint: ## Run linting for all Python files
	@echo "Running flake8 linting..."
	flake8 . --count --statistics

format: ## Format all Python code
	@echo "Formatting Python code with black and isort..."
	black . --line-length=120
	isort . --profile=black --line-length=120

format-check: ## Check if code is properly formatted (CI)
	@echo "Checking code formatting..."
	black . --check --line-length=120
	isort . --check-only --profile=black --line-length=120

security-scan: ## Run security scanning
	@echo "Running security scan with bandit..."
	bandit -r . -f json -o security-report.json || true
	bandit -r . --exclude=tests,*/tests/* -ll

deps-check: ## Check for security vulnerabilities in dependencies
	@echo "Checking dependencies for security issues..."
	@for service in agentQ managerQ VectorStoreQ KnowledgeGraphQ QuantumPulse AuthQ; do \
		if [ -f $$service/requirements.txt ]; then \
			echo "Checking $$service dependencies..."; \
			safety check -r $$service/requirements.txt || true; \
		fi; \
	done

pre-commit: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

##@ Testing Commands

test: ## Run all tests
	@echo "Running tests for all services..."
	@for service in agentQ managerQ VectorStoreQ KnowledgeGraphQ QuantumPulse AuthQ; do \
		if [ -d $$service/tests ]; then \
			echo "Testing $$service..."; \
			cd $$service && python -m pytest tests/ -v --cov=. --cov-report=term-missing || cd ..; \
		fi; \
	done

test-unit: ## Run only unit tests
	@echo "Running unit tests..."
	pytest -m "unit" -v

test-integration: ## Run only integration tests
	@echo "Running integration tests..."
	pytest -m "integration" -v

test-coverage: ## Generate test coverage report
	@echo "Generating coverage report..."
	pytest --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing

##@ Build Commands

build: ## Build all Docker containers
	@echo "Building Docker containers..."
	@for service in agentQ managerQ VectorStoreQ KnowledgeGraphQ QuantumPulse AuthQ; do \
		if [ -f $$service/Dockerfile ]; then \
			echo "Building $$service..."; \
			docker build -t q2/$$service:latest $$service/; \
		fi; \
	done

build-service: ## Build specific service (usage: make build-service SERVICE=agentQ)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE parameter is required. Usage: make build-service SERVICE=agentQ"; \
		exit 1; \
	fi
	@if [ -f $(SERVICE)/Dockerfile ]; then \
		echo "Building $(SERVICE)..."; \
		docker build -t q2/$(SERVICE):latest $(SERVICE)/; \
	else \
		echo "Error: Dockerfile not found for $(SERVICE)"; \
	fi

##@ Cleanup Commands

clean: ## Clean up build artifacts and caches
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f security-report.json coverage.xml .coverage 2>/dev/null || true

clean-docker: ## Clean up Docker containers and images
	@echo "Cleaning up Docker resources..."
	docker system prune -f
	docker image prune -f

##@ Documentation Commands

docs-serve: ## Serve documentation locally (if available)
	@echo "Starting documentation server..."
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs serve; \
	else \
		echo "MkDocs not installed. Install with: pip install mkdocs"; \
	fi

docs-generate: ## Generate API documentation from services
	@echo "Generating API documentation..."
	@python3 scripts/generate-api-docs.py

docs-generate-live: ## Generate API documentation from running services
	@echo "Generating API documentation from running services..."
	@python3 scripts/generate-api-docs.py --fetch-running

docs-clean: ## Clean generated documentation
	@echo "Cleaning generated documentation..."
	@rm -rf docs/api/

##@ Development Tools

scaffold-service: ## Create a new service (usage: make scaffold-service SERVICE=MyNewService)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE parameter is required. Usage: make scaffold-service SERVICE=MyNewService"; \
		exit 1; \
	fi
	@python3 scripts/scaffold-service.py $(SERVICE)

scaffold-worker: ## Create a new worker service (usage: make scaffold-worker SERVICE=MyWorker)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE parameter is required. Usage: make scaffold-worker SERVICE=MyWorker"; \
		exit 1; \
	fi
	@python3 scripts/scaffold-service.py $(SERVICE) --type worker

debug-services: ## Debug all Q2 Platform services
	@echo "Debugging all Q2 Platform services..."
	@python3 scripts/debug-service.py

debug-service: ## Debug specific service (usage: make debug-service SERVICE=agentQ)
	@if [ -z "$(SERVICE)" ]; then \
		echo "Error: SERVICE parameter is required. Usage: make debug-service SERVICE=agentQ"; \
		exit 1; \
	fi
	@python3 scripts/debug-service.py --service $(SERVICE)

debug-infrastructure: ## Debug infrastructure services
	@echo "Debugging infrastructure services..."
	@python3 scripts/debug-service.py --infrastructure

##@ Development Workflow

dev-check: format-check lint security-scan test ## Run all code quality checks (CI pipeline)

dev-fix: format lint ## Fix common code quality issues

dev-setup: setup-dev install-deps ## Complete development environment setup

# Development server targets (if applicable)
serve-agentq: ## Start agentQ development server
	cd agentQ && python -m agentQ.app.main

serve-managerq: ## Start managerQ development server
	cd managerQ && python -m managerQ.app.main

# Utility targets
check-deps: ## Check for missing dependencies
	@echo "Checking for missing Python dependencies..."
	@bash scripts/validate-dev-env.sh

validate-dev: ## Validate development environment setup
	@echo "Validating development environment..."
	@bash scripts/validate-dev-env.sh