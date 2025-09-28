# Contributing to Q2 Platform

Thank you for your interest in contributing to the Q2 Platform! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** following the [Developer Guide](DEVELOPER_GUIDE.md)
4. **Create a feature branch** for your changes

## Development Workflow

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

5. **Create a pull request** with a clear description of your changes

### Code Quality Standards

- **Testing**: Maintain test coverage above 80%
- **Documentation**: Update relevant documentation with your changes
- **Code Style**: Use provided formatting tools (Black, isort)
- **Type Hints**: Include type annotations for new code
- **Security**: Follow security best practices

### Commit Message Format

We use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Security scan passes
- [ ] No breaking changes (or clearly documented)

### PR Description

Include:
- Summary of changes
- Motivation and context
- Type of change (bug fix, new feature, breaking change, etc.)
- Testing performed
- Screenshots (for UI changes)

## Issue Reporting

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages or logs

### Feature Requests

Include:
- Clear use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## Development Setup

See the [Developer Guide](DEVELOPER_GUIDE.md) for complete setup instructions.

Quick setup:
```bash
make setup-dev
make install-deps
make validate-dev
```

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold professional and respectful standards.

## Questions?

- Check the [Developer Guide](DEVELOPER_GUIDE.md)
- Search existing issues
- Create a new issue for questions

---

Thank you for contributing to Q2 Platform! 🚀