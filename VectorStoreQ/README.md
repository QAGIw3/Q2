# VectorStoreQ Service

## Overview

VectorStoreQ is the centralized vector database service for the Q2 Platform. It provides a robust, scalable, and secure API for ingesting and searching high-dimensional embeddings that power Retrieval-Augmented Generation (RAG), semantic search, and other AI capabilities across the platform.

**Service Type:** Data Storage Service  
**Port:** 8002  
**API Documentation:** http://localhost:8002/docs  

## Architecture

### Core Components
1. **VectorStoreQ Service**: FastAPI microservice managing Milvus database access
2. **q_vectorstore_client Library**: Shared Python client library for service integration

### Key Features
- ✅ Centralized vector database management with Milvus backend
- ✅ High-performance semantic search and similarity matching
- ✅ Scalable ingestion pipeline for embeddings
- ✅ Multi-collection support for data organization
- ✅ RESTful API with comprehensive validation
- ✅ Shared client library for consistent integration
- ✅ Health monitoring and readiness checks
- ✅ Docker containerization and Kubernetes ready

### Dependencies
- **External Services:** Milvus cluster
- **Infrastructure:** Apache Pulsar (optional for events)

## Getting Started

### Prerequisites
- Python 3.11+
- Running Milvus cluster (use Milvus Lite for development)
- Docker (for containerized deployment)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r VectorStoreQ/requirements.txt
   ```

2. **Install client library (for integration):**
   ```bash
   # For other services to use the client
   pip install -e ./shared/q_vectorstore_client
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with Milvus connection details
   ```

4. **Run the service:**
   ```bash
   python -m vectorstoreq.app.main
   ```

5. **Verify it's running:**
   ```bash
   curl http://localhost:8002/health
   ```

### Docker Deployment

```bash
# Build the image
docker build -t q2/vectorstoreq:latest VectorStoreQ/

# Run the container
docker run -p 8002:8002 \
  -e MILVUS_HOST=milvus-server \
  -e MILVUS_PORT=19530 \
  q2/vectorstoreq:latest
```

The service is configured via `VectorStoreQ/config/vectorstore.yaml`. Ensure the `milvus` host and port point to your running cluster.

```yaml
milvus:
  host: "localhost"
  port: 19530
```

### 3. Running the Service

You can run the service directly via Uvicorn for development.

```bash
# From the root of the Q project, add the project to the PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the server
uvicorn VectorStoreQ.app.main:app --host 0.0.0.0 --port 8001 --reload
```

The API documentation will be available at `http://127.0.0.1:8001/docs`.

---

## 🐳 Docker Deployment

A `Dockerfile` is provided to containerize the service.

1.  **Build the Image**
    ```bash
    # From the root of the Q project
    docker build -f VectorStoreQ/Dockerfile -t vectorstore-q .
    ```

2.  **Run the Container**
    ```bash
    # This command maps the port and uses --network="host" to easily
    # connect to a Milvus instance running on the host's localhost.
    docker run -p 8001:8001 --network="host" --name vectorstore-q vectorstore-q
    ```

---

## API Endpoints

The service provides the following versioned API endpoints. All endpoints require a valid JWT from an authenticated user, passed via the Istio gateway.

### Ingestion

*   `POST /v1/ingest/upsert`
    *   **Purpose**: Inserts or updates a batch of vectors in a specified collection.
    *   **Authorization**: Requires a role of `admin` or `service-account`.
    *   **Request Body**: `UpsertRequest` (see `q_vectorstore_client/models.py`)
    *   **Response**: A confirmation with the number of inserted records and their primary keys.

### Search

*   `POST /v1/search`
    *   **Purpose**: Performs a batch similarity search across one or more query vectors.
    *   **Authorization**: Requires any authenticated user role.
    *   **Request Body**: `SearchRequest` (see `q_vectorstore_client/models.py`)
    *   **Response**: `SearchResponse`, containing a list of hits for each query. 