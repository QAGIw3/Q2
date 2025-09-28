# WorkflowEngine API Documentation

## Overview

WorkflowEngine is the workflow orchestration service for the Q2 Platform architecture.

**Service Port:** 8030  
**Base URL:** http://localhost:8030  
**API Documentation:** http://localhost:8030/docs (when running)  
**OpenAPI Spec:** http://localhost:8030/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd WorkflowEngine/app
   python -m workflowengine.app.main
   ```

2. Visit the interactive API documentation at http://localhost:8030/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Core Features

- **Workflow Definition**: Define and manage complex workflows
- **Task Orchestration**: Coordinate multi-step processes
- **State Management**: Track workflow execution state
- **Error Handling**: Robust error recovery and retry mechanisms
- **Event Integration**: Integration with Apache Pulsar for events

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.