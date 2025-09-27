# Q2 Platform API Documentation

This documentation provides comprehensive API reference for all Q2 Platform services.

## Services

- [AgentSandbox](./agentsandbox.md) - AgentSandbox service API
 (Port: 8006)

- [AuthQ](./authq.md) - AuthQ service API
 (Port: 8004)

- [H2M](./h2m.md) - H2M service API
 (Port: 8005)

- [KnowledgeGraphQ](./knowledgegraphq.md) - KnowledgeGraphQ service API
 (Port: 8003)

- [UserProfileQ](./userprofileq.md) - UserProfileQ service API
 (Port: 8007)

- [VectorStoreQ](./vectorstoreq.md) - VectorStoreQ service API
 (Port: 8002)

- [agentQ](./agentq.md) - agentQ service API
 (Port: 8000)

- [managerQ](./managerq.md) - managerQ service API
 (Port: 8001)


## Authentication

Most services require authentication via Keycloak OIDC tokens.

Include the token in the Authorization header:

```
Authorization: Bearer <your-token>
```

## Error Handling

All services follow consistent error response format:

```json

{

  "error": {

    "code": "ERROR_CODE",

    "message": "Human readable error message",

    "details": {}

  }

}

```
