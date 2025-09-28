import logging
from typing import Any, Dict, Optional

import pulsar
from app.models.inference import InferenceRequest
from opentelemetry import trace
from opentelemetry.propagate import inject
from pulsar.schema import JsonSchema

try:
    # Newer versions expose textmap via opentelemetry.propagators
    from opentelemetry.propagators import textmap as _otel_textmap
except ImportError:  # pragma: no cover
    # Fallback (older layouts) - unlikely with current pinned version
    from opentelemetry.propagate import textmap as _otel_textmap  # type: ignore

TextMapPropagator = getattr(_otel_textmap, "TextMapPropagator")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PulsarMessageTextMapPropagator(TextMapPropagator):
    """A custom propagator adapting OpenTelemetry TextMapPropagator for Pulsar message properties.

    Implements minimal inject/extract semantics leveraging the global composite propagator.
    """

    def inject(self, carrier: Dict[str, str], context=None, setter=None):  # type: ignore[override]
        from opentelemetry.propagate import get_global_textmap

        _setter = setter or (lambda c, k, v: c.__setitem__(k, v))
        get_global_textmap().inject(carrier, context=context, setter=_setter)

    def extract(self, carrier: Dict[str, str], context=None, getter=None):  # type: ignore[override]
        from opentelemetry.propagate import get_global_textmap

        _getter = getter or (lambda c, k: [c.get(k)] if k in c else [])
        return get_global_textmap().extract(carrier, context=context, getter=_getter)

    # Older interface compatibility helpers (not required but kept for clarity)
    def fields(self) -> set:  # type: ignore[override]
        return set()


propagator = PulsarMessageTextMapPropagator()


class PulsarManager:
    """
    Manages the connection to Pulsar and handles producers and consumers.
    """

    _client: Optional[pulsar.Client] = None
    _producers: Dict[str, pulsar.Producer] = {}

    def __init__(self, service_url: str, token: Optional[str] = None, tls_trust_certs_file_path: Optional[str] = None):
        self.service_url = service_url
        self.token = token
        self.tls_trust_certs_file_path = tls_trust_certs_file_path

    def connect(self):
        """
        Establishes the connection to the Pulsar cluster.
        """
        if self._client:
            logger.info("Pulsar client already connected.")
            return

        try:
            client_args = {"service_url": self.service_url}
            if self.token:
                client_args["authentication"] = pulsar.AuthenticationToken(self.token)
            if self.tls_trust_certs_file_path:
                client_args["tls_trust_certs_file_path"] = self.tls_trust_certs_file_path

            self._client = pulsar.Client(**client_args)
            logger.info("Successfully connected to Pulsar at %s", self.service_url)
        except Exception as e:
            logger.error("Failed to connect to Pulsar: %s", e, exc_info=True)
            raise

    def close(self):
        """
        Closes all producers and the client connection.
        """
        for topic, producer in self._producers.items():
            try:
                producer.close()
                logger.info("Closed producer for topic: %s", topic)
            except Exception as e:
                logger.error("Error closing producer for topic %s: %s", topic, e)

        if self._client:
            try:
                self._client.close()
                self._client = None
                logger.info("Pulsar client connection closed.")
            except Exception as e:
                logger.error("Error closing Pulsar client: %s", e)

        self._producers.clear()

    def get_producer(self, topic: str, schema: Any) -> pulsar.Producer:
        """
        Retrieves an existing producer or creates a new one for the given topic.
        """
        if not self._client:
            raise ConnectionError("Pulsar client is not connected. Call connect() first.")

        if topic not in self._producers:
            try:
                self._producers[topic] = self._client.create_producer(
                    topic, schema=schema, properties={"producer-name": f"quantumpulse-api-{topic}"}
                )
                logger.info("Created producer for topic: %s", topic)
            except Exception as e:
                logger.error("Failed to create producer for topic %s: %s", topic, e, exc_info=True)
                raise
        return self._producers[topic]

    def publish_request(self, topic: str, request: InferenceRequest):
        """
        Publishes an inference request to a specified Pulsar topic.
        """
        try:
            properties = {}
            # Inject the current tracing context into the message properties
            inject(properties, propagator=propagator)

            producer = self.get_producer(topic, JsonSchema(type(request)))
            producer.send(request, properties=properties)
            logger.info("Published request %s to topic %s with trace context.", request.request_id, topic)
        except Exception as e:
            logger.error("Failed to publish request %s to topic %s: %s", request.request_id, topic, e, exc_info=True)
            # Optionally re-raise or handle the error
            raise


# A global instance to be initialized on app startup
pulsar_manager: Optional[PulsarManager] = None


def get_pulsar_manager() -> PulsarManager:
    """
    Dependency injector for the PulsarManager.
    """
    if not pulsar_manager:
        raise RuntimeError("PulsarManager has not been initialized.")
    return pulsar_manager
