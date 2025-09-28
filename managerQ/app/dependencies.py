# managerQ/app/dependencies.py
import os
from functools import lru_cache

from managerQ.app.core.result_listener import result_listener
from managerQ.app.core.task_dispatcher import task_dispatcher
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from shared.q_pulse_client.client import QuantumPulseClient
from shared.q_vectorstore_client.client import VectorStoreClient


# In a real app, config would come from a more robust source
@lru_cache()
def get_vector_store_client() -> VectorStoreClient:
    return VectorStoreClient(base_url=os.getenv("VECTORSTORE_Q_URL", "http://vectorstoreq:8000"))


@lru_cache()
def get_kg_client() -> KnowledgeGraphClient:
    return KnowledgeGraphClient(base_url=os.getenv("KNOWLEDGEGRAPH_Q_URL", "http://knowledgegraphq:8000"))


@lru_cache()
def get_pulse_client() -> QuantumPulseClient:
    return QuantumPulseClient(base_url=os.getenv("QUANTUMPULSE_URL", "http://quantumpulse:8000"))


def get_task_dispatcher():
    return task_dispatcher


def get_result_listener():
    return result_listener


# Simple auth stub (offline/dev): would validate JWT or session in production.
def get_current_user():
    return {"sub": "dev-user"}
