# VectorStoreQ API Documentation

## Overview

VectorStoreQ is a core service in the Q2 Platform architecture.

**Service Port:** 8002  
**Base URL:** http://localhost:8002  
**API Documentation:** http://localhost:8002/docs (when running)  
**OpenAPI Spec:** http://localhost:8002/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd VectorStoreQ/app
   python -m vectorstoreq.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8002/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.
