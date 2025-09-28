# managerQ Service

## Overview

managerQ is the control plane for the Q2 Platform's multi-agent system. It acts as a centralized dispatcher and coordinator, making the pool of autonomous agentQ workers a usable and scalable resource for the rest of the platform.

**Service Type:** Orchestration Service  
**Port:** 8001  
**API Documentation:** http://localhost:8001/docs  

## Architecture

### Core Responsibilities
- **Agent Discovery**: Maintaining a real-time registry of all active and available agentQ instances
- **Task Dispatching**: Providing a unified API for other services to submit tasks to the agent pool
- **Load Balancing**: Distributing incoming tasks across available agents with the correct personality
- **Result Correlation**: Listening for results from agents and correlating them back to original requests

### Key Features
- ✅ Real-time agent registry management
- ✅ Intelligent task dispatching and load balancing
- ✅ FastAPI service with background thread coordination
- ✅ Apache Pulsar integration for messaging
- ✅ Result correlation with asyncio event handling
- ✅ Multi-personality agent support
- ✅ Health monitoring and readiness checks

1.  **`AgentRegistry`**: A background thread consumes from the `q.agentq.registrations` topic. When an `agentQ` instance starts, it publishes a registration message. The registry adds it to an in-memory list of active agents.

2.  **`TaskDispatcher`**: When a request comes into the API, the dispatcher gets an available agent from the registry (based on the requested `personality`) and publishes the task message directly to that agent's unique task topic.

3.  **`ResultListener`**: A second background thread consumes from the shared `q.agentq.results` topic. When it receives a result, it uses an `asyncio.Event` to notify the original API request handler that the task is complete.

4.  **API Layer**: The FastAPI application exposes two main endpoints:
    - `POST /v1/tasks`: Submits a new task to an agent. This endpoint returns a `task_id`.
    - `GET /v1/tasks/{task_id}`: Retrieves the result of a task. This endpoint will block until the result is available or a timeout occurs.

## Getting Started

### 1. Prerequisites

-   An running Apache Pulsar cluster.
-   At least one running `agentQ` instance.

### 2. Running the Service

The service can be run directly via Uvicorn for development.

```bash
# From the project root
export PYTHONPATH=$(pwd)

# Run the server
uvicorn managerQ.app.main:app --reload
```

The API documentation will be available at `http://127.0.0.1:8003/docs`.
