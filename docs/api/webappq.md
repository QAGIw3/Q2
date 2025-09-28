# WebAppQ API Documentation

## Overview

WebAppQ is the React web application frontend for the Q2 Platform architecture.

**Service Port:** 3000  
**Base URL:** http://localhost:3000  
**API Documentation:** Frontend application (no REST API)  

## Getting Started

1. Start the service:
   ```bash
   cd WebAppQ/app
   npm install
   npm start
   ```

2. Visit the web application at http://localhost:3000

3. Access the Q2 Platform through the web interface

## Core Features

- **User Interface**: React-based web application
- **Real-time Communication**: WebSocket integration with H2M service
- **Authentication**: Keycloak OIDC integration
- **Dashboard**: Platform monitoring and management
- **Agent Interaction**: Direct communication with agentQ through UI

## Authentication

This service integrates with Keycloak for user authentication and authorization.

## Integration

WebAppQ communicates with backend services through:
- **H2M Service**: WebSocket connection for real-time chat
- **AuthQ**: Authentication and authorization
- **managerQ**: Task submission and monitoring

---

**Note:** This is a frontend application and does not expose REST API endpoints. See individual backend service documentation for API details.