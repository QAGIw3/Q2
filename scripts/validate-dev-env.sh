#!/bin/bash

# Q2 Platform Development Environment Validation Script
# This script validates that the development environment is properly set up

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

log_warning() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Increment total checks
check() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
}

# Check Python installation and version
check_python() {
    check
    log_info "Checking Python installation..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
        log_error "Python version $PYTHON_VERSION is too old. Python 3.11+ required."
        return 1
    fi
    
    log_success "Python $PYTHON_VERSION is installed and compatible"
}

# Check pip installation
check_pip() {
    check
    log_info "Checking pip installation..."
    
    if ! python3 -m pip --version >/dev/null 2>&1; then
        log_error "pip is not properly installed"
        return 1
    fi
    
    log_success "pip is installed and working"
}

# Check development tools
check_dev_tools() {
    local tools=(
        "black:Code formatter"
        "isort:Import sorter" 
        "flake8:Linter"
        "pytest:Test runner"
        "pre-commit:Pre-commit hooks"
        "bandit:Security scanner"
        "mypy:Type checker"
    )
    
    for tool_info in "${tools[@]}"; do
        check
        tool=$(echo $tool_info | cut -d':' -f1)
        desc=$(echo $tool_info | cut -d':' -f2)
        
        log_info "Checking $desc ($tool)..."
        
        if command_exists "$tool"; then
            log_success "$desc is available"
        else
            log_error "$desc ($tool) is not available in PATH"
        fi
    done
}

# Check pre-commit hooks
check_pre_commit_hooks() {
    check
    log_info "Checking pre-commit hooks installation..."
    
    if [ -f ".git/hooks/pre-commit" ]; then
        log_success "Pre-commit hooks are installed"
    else
        log_error "Pre-commit hooks are not installed. Run 'pre-commit install'"
    fi
}

# Check Docker
check_docker() {
    check
    log_info "Checking Docker installation..."
    
    if ! command_exists docker; then
        log_warning "Docker is not installed (optional for development)"
        return 0
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_warning "Docker daemon is not running (optional for development)"
        return 0
    fi
    
    log_success "Docker is installed and running"
}

# Check Node.js (for WebAppQ)
check_nodejs() {
    check
    log_info "Checking Node.js installation..."
    
    if ! command_exists node; then
        log_warning "Node.js is not installed (needed for WebAppQ development)"
        return 0
    fi
    
    NODE_VERSION=$(node --version | sed 's/v//')
    MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
    
    if [ "$MAJOR" -lt 18 ]; then
        log_warning "Node.js version $NODE_VERSION is old. Version 18+ recommended."
        return 0
    fi
    
    log_success "Node.js $NODE_VERSION is installed"
}

# Check Python dependencies
check_python_dependencies() {
    check
    log_info "Checking Python dependencies from constraints.txt..."
    
    if [ ! -f "constraints.txt" ]; then
        log_warning "constraints.txt not found. Skipping dependency check."
        return 0
    fi
    
    if python3 -c "
import pkg_resources
import sys
try:
    with open('constraints.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    pkg_resources.require(requirements)
    print('All dependencies are satisfied')
except Exception as e:
    print(f'Dependency check failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_success "Python dependencies are satisfied"
    else
        log_error "Some Python dependencies are missing or incompatible"
    fi
}

# Check service structure
check_service_structure() {
    check
    log_info "Checking service structure..."
    
    local services=("agentQ" "managerQ" "VectorStoreQ" "KnowledgeGraphQ" "AuthQ" "H2M")
    local missing_services=()
    
    for service in "${services[@]}"; do
        if [ ! -d "$service" ]; then
            missing_services+=("$service")
        fi
    done
    
    if [ ${#missing_services[@]} -eq 0 ]; then
        log_success "All core services are present"
    else
        log_error "Missing services: ${missing_services[*]}"
    fi
}

# Check configuration files
check_config_files() {
    local configs=(
        "pyproject.toml:Python project configuration"
        ".pre-commit-config.yaml:Pre-commit configuration"
        "Makefile:Build configuration"
        ".flake8:Linting configuration"
    )
    
    for config_info in "${configs[@]}"; do
        check
        config=$(echo $config_info | cut -d':' -f1)
        desc=$(echo $config_info | cut -d':' -f2)
        
        log_info "Checking $desc..."
        
        if [ -f "$config" ]; then
            log_success "$desc exists"
        else
            log_error "$desc ($config) is missing"
        fi
    done
}

# Check IDE configuration
check_ide_config() {
    check
    log_info "Checking IDE configuration..."
    
    if [ -d ".vscode" ] && [ -f ".vscode/settings.json" ]; then
        log_success "VS Code configuration is available"
    else
        log_warning "VS Code configuration not found (run dev setup to create)"
    fi
}

# Check environment files
check_env_files() {
    check
    log_info "Checking environment configuration..."
    
    if [ -f ".env.example" ]; then
        log_success ".env.example template is available"
    else
        log_warning ".env.example template not found (run dev setup to create)"
    fi
    
    if [ -f ".env" ]; then
        log_info ".env file exists (good for local development)"
    else
        log_info ".env file not found (copy from .env.example if needed)"
    fi
}

# Print summary
print_summary() {
    echo
    echo "Validation Summary"
    echo "=================="
    echo "Total checks: $CHECKS_TOTAL"
    echo -e "Passed: ${GREEN}$CHECKS_PASSED${NC}"
    echo -e "Failed: ${RED}$CHECKS_FAILED${NC}"
    echo -e "Warnings: ${YELLOW}$((CHECKS_TOTAL - CHECKS_PASSED - CHECKS_FAILED))${NC}"
    echo
    
    if [ $CHECKS_FAILED -eq 0 ]; then
        log_success "🎉 Development environment is properly set up!"
        echo
        echo "You can now:"
        echo "  • Run 'make install-deps' to install service dependencies"
        echo "  • Run 'make test' to run tests"
        echo "  • Run 'make serve-agentq' to start AgentQ service"
        echo "  • Check DEVELOPER_GUIDE.md for development workflows"
    else
        log_error "❌ Development environment has issues that need to be fixed."
        echo
        echo "To fix issues:"
        echo "  • Run 'make setup-dev' to set up development tools"
        echo "  • Install missing dependencies manually"
        echo "  • Check the error messages above for specific guidance"
        echo
        return 1
    fi
}

# Main validation function
main() {
    echo "Q2 Platform Development Environment Validation"
    echo "=============================================="
    echo
    
    # Core requirements
    check_python
    check_pip
    check_dev_tools
    check_pre_commit_hooks
    
    # Project structure
    check_service_structure
    check_config_files
    
    # Dependencies
    check_python_dependencies
    
    # Optional but recommended
    check_docker
    check_nodejs
    check_ide_config
    check_env_files
    
    # Print summary and exit with appropriate code
    print_summary
}

# Run main function
main "$@"