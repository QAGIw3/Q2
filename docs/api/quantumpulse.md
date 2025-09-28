# QuantumPulse API Documentation

## Overview

QuantumPulse is the quantum AI computation engine for the Q2 Platform architecture.

**Service Port:** 8050  
**Base URL:** http://localhost:8050  
**API Documentation:** http://localhost:8050/docs (when running)  
**OpenAPI Spec:** http://localhost:8050/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd QuantumPulse/app
   python -m quantumpulse.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8050/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Core Features

- **Quantum Machine Learning**: Train and execute quantum neural networks
- **Quantum Analytics**: Real-time quantum-enhanced data processing
- **AI Governance**: Automated compliance and bias detection
- **Agent Swarm Optimization**: Multi-agent quantum optimization

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.