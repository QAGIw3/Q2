# UserProfileQ API Documentation

## Overview

UserProfileQ is a core service in the Q2 Platform architecture.

**Service Port:** 8007  
**Base URL:** http://localhost:8007  
**API Documentation:** http://localhost:8007/docs (when running)  
**OpenAPI Spec:** http://localhost:8007/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd UserProfileQ/app
   python -m userprofileq.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8007/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.
