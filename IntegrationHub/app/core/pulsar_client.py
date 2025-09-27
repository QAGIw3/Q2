import pulsar
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from ..models.flow import Flow

# In a real application, this would come from configuration.
PULSAR_URL = 'pulsar://pulsar:6650'
# Topic for general platform events for KG ingestion
PLATFORM_EVENTS_TOPIC = 'persistent://public/default/platform-events'
# Legacy topic for the old trigger mechanism
TRIGGER_TOPIC = 'persistent://public/default/integration-hub-triggers'

_client = None
# A dictionary to hold producers for different topics
_producers: Dict[str, pulsar.Producer] = {}

def get_pulsar_producer(topic: str) -> pulsar.Producer:
    """
    Initializes and returns a singleton Pulsar producer for a specific topic.
    """
    global _client, _producers
    if topic not in _producers:
        if _client is None:
            # In a containerized setup, 'pulsar' is the typical service name.
            # For local dev, this might need to be 'localhost'.
            _client = pulsar.Client(PULSAR_URL)
        _producers[topic] = _client.create_producer(topic)
    return _producers[topic]

async def publish_event(event_type: str, source: str, payload: Dict[str, Any], topic: str = PLATFORM_EVENTS_TOPIC):
    """
    Constructs a standardized event and publishes it to a Pulsar topic.

    Args:
        event_type: A string identifying the type of event (e.g., "flow.triggered").
        source: The name of the service or component generating the event (e.g., "IntegrationHub").
        payload: A dictionary containing the event's data.
        topic: The Pulsar topic to publish to. Defaults to the platform-wide event topic.
    """
    producer = get_pulsar_producer(topic)
    
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload
    }
    
    producer.send_async(json.dumps(event).encode('utf-8'), callback=lambda res, msg_id: None)


def publish_flow_trigger(flow: Flow, trigger_data: dict = None):
    """
    Serializes a flow and publishes it to the trigger topic.
    [DEPRECATED] This function may be used by older parts of the system.
    """
    producer = get_pulsar_producer(TRIGGER_TOPIC)
    payload = {
        "flow_definition": flow.dict(),
        "trigger_data": trigger_data or {}
    }
    producer.send(json.dumps(payload).encode('utf-8'))

def close_pulsar_producers():
    """
    Closes all producers and the client. Should be called on application shutdown.
    """
    global _client, _producers
    for producer in _producers.values():
        try:
            producer.close()
        except Exception:
            pass # Ignore errors on close
    if _client:
        try:
            _client.close()
        except Exception:
            pass # Ignore errors on close
    _producers = {}
    _client = None 