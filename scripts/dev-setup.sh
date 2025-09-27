#!/bin/bash

# Q2 Platform Development Environment Setup Script
# This script sets up everything needed for Q2 Platform development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install Python development tools
install_python_tools() {
    log_info "Installing Python development tools..."
    
    # Check Python version
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.11+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 11 ]); then
        log_warning "Python version $PYTHON_VERSION detected. Python 3.11+ is recommended."
    else
        log_success "Python version $PYTHON_VERSION is compatible."
    fi
    
    # Install/upgrade pip
    python3 -m pip install --user --upgrade pip
    
    # Install development tools
    local tools=(
        "pre-commit>=3.0.0"
        "black>=23.0.0"
        "isort>=5.12.0"
        "flake8>=6.0.0"
        "flake8-bugbear"
        "flake8-comprehensions"
        "flake8-docstrings"
        "flake8-import-order"
        "pytest>=7.0.0"
        "pytest-cov>=4.0.0"
        "pytest-asyncio>=0.21.0"
        "bandit>=1.7.0"
        "safety>=2.0.0"
        "mypy>=1.0.0"
        "structlog>=23.0.0"
    )
    
    for tool in "${tools[@]}"; do
        log_info "Installing $tool..."
        python3 -m pip install --user "$tool"
    done
    
    log_success "Python development tools installed."
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    if ! command_exists pre-commit; then
        log_error "pre-commit not found in PATH. Make sure ~/.local/bin is in your PATH."
        return 1
    fi
    
    pre-commit install
    log_success "Pre-commit hooks installed."
}

# Check Docker installation
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command_exists docker; then
        log_warning "Docker is not installed. Please install Docker for container development."
        return 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_warning "Docker daemon is not running. Please start Docker."
        return 1
    fi
    
    log_success "Docker is installed and running."
}

# Check Node.js installation (for WebAppQ)
check_nodejs() {
    log_info "Checking Node.js installation..."
    
    if ! command_exists node; then
        log_warning "Node.js is not installed. Install Node.js 18+ for WebAppQ development."
        return 1
    fi
    
    NODE_VERSION=$(node --version | sed 's/v//')
    MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
    
    if [ "$MAJOR" -lt 18 ]; then
        log_warning "Node.js version $NODE_VERSION detected. Node.js 18+ is recommended."
    else
        log_success "Node.js version $NODE_VERSION is compatible."
    fi
}

# Install Node.js dependencies for WebAppQ
install_nodejs_deps() {
    log_info "Installing Node.js dependencies for WebAppQ..."
    
    if [ -d "WebAppQ/app" ] && [ -f "WebAppQ/app/package.json" ]; then
        cd WebAppQ/app
        npm install
        cd ../..
        log_success "WebAppQ dependencies installed."
    else
        log_warning "WebAppQ directory not found or no package.json. Skipping Node.js setup."
    fi
}

# Create development configuration files
create_dev_configs() {
    log_info "Creating development configuration files..."
    
    # Create .env.example if it doesn't exist
    if [ ! -f ".env.example" ]; then
        cat > .env.example << 'EOF'
# Q2 Platform Development Environment Variables
# Copy this file to .env and customize for your local setup

# Python environment
PYTHONPATH=.
LOG_LEVEL=DEBUG

# Service URLs (adjust ports as needed)
AGENT_Q_URL=http://localhost:8000
MANAGER_Q_URL=http://localhost:8001
VECTORSTORE_Q_URL=http://localhost:8002
KNOWLEDGEGRAPH_Q_URL=http://localhost:8003
AUTH_Q_URL=http://localhost:8004
H2M_URL=http://localhost:8005

# Database connections
PULSAR_URL=pulsar://localhost:6650
MILVUS_HOST=localhost
MILVUS_PORT=19530
JANUSGRAPH_URL=ws://localhost:8182/gremlin

# Authentication
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=q-platform
KEYCLOAK_CLIENT_ID=q-platform-client

# Development flags
DEV_MODE=true
DEBUG=true
TESTING=false
EOF
        log_success "Created .env.example file."
    fi
    
    # Create VS Code settings
    mkdir -p .vscode
    if [ ! -f ".vscode/settings.json" ]; then
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "python3",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "python.sortImports.args": ["--profile=black", "--line-length=120"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/htmlcov": true,
        "**/.pytest_cache": true,
        "**/node_modules": true
    },
    "python.analysis.extraPaths": [
        "shared"
    ]
}
EOF
        log_success "Created VS Code settings."
    fi
    
    # Create launch.json for debugging
    if [ ! -f ".vscode/launch.json" ]; then
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug AgentQ",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/agentQ/app/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/agentQ",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug ManagerQ",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/managerQ/app/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}/managerQ",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
EOF
        log_success "Created VS Code launch configuration."
    fi
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    local errors=0
    
    # Check Python tools
    local python_tools=("black" "isort" "flake8" "pytest" "pre-commit" "bandit")
    for tool in "${python_tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "$tool not found in PATH"
            errors=$((errors + 1))
        fi
    done
    
    # Check if pre-commit is set up
    if [ ! -f ".git/hooks/pre-commit" ]; then
        log_error "Pre-commit hooks not installed"
        errors=$((errors + 1))
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "All tools validated successfully!"
        return 0
    else
        log_error "Found $errors issue(s) during validation."
        return 1
    fi
}

# Print next steps
print_next_steps() {
    echo
    log_success "🎉 Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Add ~/.local/bin to your PATH if not already done:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo
    echo "2. Copy .env.example to .env and customize:"
    echo "   cp .env.example .env"
    echo
    echo "3. Install service dependencies:"
    echo "   make install-deps"
    echo
    echo "4. Run tests to verify everything works:"
    echo "   make test"
    echo
    echo "5. Start developing:"
    echo "   make serve-agentq  # Start AgentQ service"
    echo "   make serve-managerq  # Start ManagerQ service"
    echo
    log_info "Check DEVELOPER_GUIDE.md for detailed development instructions."
}

# Main execution
main() {
    echo "Q2 Platform Development Environment Setup"
    echo "========================================"
    echo
    
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    # Install Python tools
    if ! install_python_tools; then
        log_error "Failed to install Python tools"
        exit 1
    fi
    
    # Setup pre-commit
    if ! setup_pre_commit; then
        log_warning "Pre-commit setup failed, but continuing..."
    fi
    
    # Check optional dependencies
    check_docker || log_warning "Docker setup issues detected"
    check_nodejs || log_warning "Node.js setup issues detected"
    
    # Install Node.js dependencies if available
    if command_exists npm; then
        install_nodejs_deps || log_warning "Node.js dependencies installation failed"
    fi
    
    # Create development configuration
    create_dev_configs
    
    # Validate installation
    if validate_installation; then
        print_next_steps
    else
        log_error "Setup completed with issues. Check the errors above."
        exit 1
    fi
}

# Run main function
main "$@"