# H2M API Documentation

## Overview

H2M is a core service in the Q2 Platform architecture.

**Service Port:** 8005  
**Base URL:** http://localhost:8005  
**API Documentation:** http://localhost:8005/docs (when running)  
**OpenAPI Spec:** http://localhost:8005/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd H2M/app
   python -m h2m.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8005/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.
