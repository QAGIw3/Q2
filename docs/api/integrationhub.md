# IntegrationHub API Documentation

## Overview

IntegrationHub is the external system integration service for the Q2 Platform architecture.

**Service Port:** 8020  
**Base URL:** http://localhost:8020  
**API Documentation:** http://localhost:8020/docs (when running)  
**OpenAPI Spec:** http://localhost:8020/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd IntegrationHub/app
   python -m integrationhub.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8020/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Core Features

- **Workflow Integration**: Pre-defined workflow execution
- **External API Connectors**: Integration with third-party services
- **Event Processing**: Real-time event handling and routing
- **Authentication Management**: OAuth and API key management

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.