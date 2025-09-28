# agentQ Service

## Overview

agentQ is the core reasoning engine and autonomous agent of the Q2 Platform. It serves as a stateful, multi-tool autonomous agent capable of complex, multi-step problem-solving and decision-making across distributed environments.

**Service Type:** Core Processing Service  
**Port:** 8000  
**API Documentation:** http://localhost:8000/docs  

## Architecture

### Core Capabilities
The agent is built on a **ReAct (Reason, Act)** loop architecture that enables iterative problem-solving:

1. **Reason (Thought)**: Analyzes user queries and conversation history to generate reasoning plans
2. **Act (Action)**: Chooses specific actions and executes them via JSON-formatted commands  
3. **Observe**: Processes tool outputs and system responses
4. **Repeat**: Integrates new information and continues the reasoning cycle

### Key Features
- ✅ ReAct (Reason, Act) autonomous agent architecture
- ✅ Multi-tool integration with Q2 Platform services
- ✅ Stateful conversation management and memory
- ✅ Complex multi-step problem decomposition
- ✅ Integration with vector search and knowledge graphs
- ✅ Secure execution environment integration
- ✅ Real-time event processing via Apache Pulsar
- ✅ Quantum AI computation delegation

### Agent Toolbox
The agent has access to the complete Q2 Platform service ecosystem:

- **`search_knowledge_base`**: Semantic search via VectorStoreQ for unstructured information
- **`query_knowledge_graph`**: Gremlin queries via KnowledgeGraphQ for structured data relationships
- **`delegate_to_quantumpulse`**: Complex reasoning delegation to quantum AI computation engine
- **`trigger_integration_flow`**: Workflow execution via IntegrationHub for external system actions
- **`ask_human_for_clarification`**: Human-in-the-loop interaction for complex decision points

## Getting Started

### Prerequisites
- Python 3.11+
- Complete Q2 Platform stack (Pulsar, VectorStoreQ, KnowledgeGraphQ, etc.)
- HashiCorp Vault with API keys (e.g., OPENAI_API_KEY)
- Docker (for containerized deployment)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   # Set environment variables
   export PYTHONPATH=$(pwd)
   export VAULT_ADDR="http://your-vault-address:8200"
   export VAULT_TOKEN="your-vault-token"
   ```

3. **Run the service:**
   ```bash
   python agentQ/app/main.py
   ```

4. **Verify it's running:**
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Deployment

```bash
# Build the image
docker build -t q2/agentq:latest agentQ/

# Run the container
docker run -p 8000:8000 \
  -e VAULT_ADDR=http://vault:8200 \
  -e VAULT_TOKEN=your-token \
  q2/agentq:latest
```

The agent will start, register itself with `managerQ`, and begin listening for tasks on its unique Pulsar topic.
