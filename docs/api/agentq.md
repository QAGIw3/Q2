# agentQ API Documentation

## Overview

agentQ is a core service in the Q2 Platform architecture.

**Service Port:** 8000  
**Base URL:** http://localhost:8000  
**API Documentation:** http://localhost:8000/docs (when running)  
**OpenAPI Spec:** http://localhost:8000/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd agentQ/app
   python -m agentq.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8000/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.
