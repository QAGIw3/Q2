# managerQ/app/main.py
import logging
import os
from contextlib import asynccontextmanager

import structlog
import uvicorn
import yaml  # added for loading predefined goals
from fastapi import FastAPI, Request

from managerQ.app.api import (
    agent_tasks,
    dashboard_ws,
    goals,
    model_registry,
    observability_ws,
    planner,
    search,
    tasks,
    user_workflows,
    workflows,
)
from managerQ.app.config import settings
from managerQ.app.core.agent_registry import AgentRegistry, agent_registry
from managerQ.app.core.autoscaler import autoscaler
from managerQ.app.core.event_listener import EventListener
from managerQ.app.core.goal_manager import GoalManager, goal_manager
from managerQ.app.core.goal_monitor import proactive_goal_monitor
from managerQ.app.core.result_listener import ResultListener, result_listener
from managerQ.app.core.task_dispatcher import TaskDispatcher, task_dispatcher
from managerQ.app.core.user_workflow_store import user_workflow_store
from managerQ.app.core.workflow_executor import workflow_executor
from managerQ.app.models import Goal
from shared.observability.logging_config import setup_logging
from shared.observability.metrics import setup_metrics
from shared.opentelemetry.tracing import setup_tracing
from shared.pulsar_client import shared_pulsar_client
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_vectorstore_client.client import VectorStoreClient
from shared.vault_client import VaultClient  # retained for future vault-based extensions (unused now)

OFFLINE = os.environ.get("MANAGERQ_OFFLINE", "0") == "1"

# --- Logging and Metrics ---
setup_logging(service_name=settings.service_name)
logger = structlog.get_logger(__name__)


def load_predefined_goals():
    """Loads goals from a YAML file and saves them to the GoalManager."""
    try:
        with open("managerQ/config/goals.yaml", "r") as f:
            goals_data = yaml.safe_load(f)

        if not goals_data:
            return

        for goal_data in goals_data:
            goal = Goal(**goal_data)
            goal_manager.create_goal(goal)
            logger.info(f"Loaded and saved pre-defined goal: {goal.goal_id}")
    except FileNotFoundError:
        logger.warning("goals.yaml not found, no pre-defined goals will be loaded.")
    except Exception as e:
        logger.error(f"Failed to load pre-defined goals: {e}", exc_info=True)


# Configuration now provided exclusively by pydantic settings (deterministic local fallback)
vectorstore_q_url = os.getenv("VECTORSTORE_Q_URL", "http://localhost:8020")
knowledgegraph_q_url = os.getenv("KNOWLEDGEGRAPH_Q_URL", "http://localhost:8030")
quantumpulse_url = getattr(settings, "qpulse_url", os.getenv("QUANTUMPULSE_URL", "http://localhost:8010"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application's resources.
    """
    logger.info("ManagerQ starting up... offline=%s", OFFLINE)

    # Initialize Pulsar client first, as other components depend on it
    # This assumes shared_pulsar_client is configured and connects on initialization

    # Initialize and start our new background services
    global agent_registry, task_dispatcher, result_listener

    if not OFFLINE:
        agent_registry = AgentRegistry(pulsar_client=shared_pulsar_client)
        task_dispatcher = TaskDispatcher(pulsar_client=shared_pulsar_client, agent_registry=agent_registry)
        result_listener = ResultListener(pulsar_client=shared_pulsar_client)
        agent_registry.start()
        result_listener.start()
    else:
        logger.warning("Skipping agent registry, dispatcher, and result listener startup (offline mode).")

    # Initialize API Clients
    app.state.vector_store_client = VectorStoreClient(base_url=vectorstore_q_url)
    app.state.kg_client = KnowledgeGraphClient(base_url=knowledgegraph_q_url)
    app.state.pulse_client = QuantumPulseClient(base_url=quantumpulse_url)

    # Initialize and start background services
    await user_workflow_store.connect()
    if not OFFLINE:
        dashboard_ws.manager.startup()
    else:
        logger.warning("Skipping dashboard websocket manager startup (offline mode).")

    if not OFFLINE:
        workflow_executor.start()
        load_predefined_goals()
        proactive_goal_monitor.start()
        autoscaler.start()
    else:
        logger.warning("Skipping executor, goal monitor, autoscaler (offline mode). Predefined goals still loaded.")
        load_predefined_goals()

    if not OFFLINE:
        platform_events_topic = getattr(
            settings.pulsar.topics, "platform_events", "persistent://public/default/platform-events"
        )
        event_listener_instance = EventListener(settings.pulsar.service_url, platform_events_topic)
        import threading

        threading.Thread(target=event_listener_instance.start, daemon=True).start()
    else:
        logger.warning("Skipping event listener (offline mode).")

    yield  # Application is running

    logger.info("ManagerQ shutting down...")

    # Stop background services
    if not OFFLINE:
        agent_registry.stop()
        result_listener.stop()
    if not OFFLINE:
        dashboard_ws.manager.shutdown()
    if not OFFLINE:
        workflow_executor.stop()
        proactive_goal_monitor.stop()
        autoscaler.stop()

    # Close the shared pulsar client
    if not OFFLINE:
        shared_pulsar_client.close()


# --- FastAPI App ---
app = FastAPI(
    title=settings.service_name,
    version=settings.version,
    description="A service to manage and orchestrate autonomous AI agents.",
    lifespan=lifespan,
)

# Setup Prometheus metrics
setup_metrics(app, app_name=settings.service_name)
setup_tracing(app, service_name=settings.service_name)


# --- Dependency Providers ---
def get_vector_store_client(request: Request) -> VectorStoreClient:
    return request.app.state.vector_store_client


def get_kg_client(request: Request) -> KnowledgeGraphClient:
    return request.app.state.kg_client


def get_pulse_client(request: Request) -> QuantumPulseClient:
    return request.app.state.pulse_client


# --- API Routers ---
app.include_router(tasks.router, prefix="/v1/tasks", tags=["Tasks"])
app.include_router(goals.router, prefix="/v1/goals", tags=["Goals"])
# Skip dashboard websocket endpoints entirely in offline mode to avoid
# referencing Pulsar topics that are not defined when MANAGERQ_OFFLINE=1.
if not OFFLINE:
    app.include_router(dashboard_ws.router, prefix="/v1/dashboard", tags=["Dashboard"])
app.include_router(agent_tasks.router, prefix="/v1/agent-tasks", tags=["Agent Tasks"])
app.include_router(workflows.router, prefix="/v1/workflows", tags=["Workflows"])
app.include_router(search.router, prefix="/v1/search", tags=["Search"])
app.include_router(model_registry.router, prefix="/v1/model-registry", tags=["Model Registry"])
app.include_router(planner.router, prefix="/v1/planner", tags=["Planner"])
app.include_router(user_workflows.router, prefix="/v1/user-workflows", tags=["User Workflows"])
app.include_router(observability_ws.router, prefix="/v1/observability", tags=["Observability"])


@app.get("/health", tags=["Health"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "managerQ.app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
    )
