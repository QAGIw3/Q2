"""
Enhanced QuantumPulse client using standardized Q2 client patterns.

This is a refactored version of the original QuantumPulse client that demonstrates
the new standardized patterns for Q2 Platform service clients.
"""

from typing import Any, Dict, Optional, AsyncGenerator
from dataclasses import dataclass

from ..clients.http import HTTPClient, HTTPClientConfig, ClientResponse
from ..error_handling import Q2ValidationError, Q2ExternalServiceError
from .models import InferenceRequest, QPChatRequest, QPChatResponse


@dataclass
class QuantumPulseConfig(HTTPClientConfig):
    """Configuration specific to QuantumPulse client."""

    # QuantumPulse specific settings
    stream_timeout: float = 120.0
    max_chat_tokens: int = 4096
    default_temperature: float = 0.7

    def __post_init__(self):
        # Override service name if not explicitly set
        if not hasattr(self, "_service_name_set"):
            self.service_name = "quantum-pulse"


class EnhancedQuantumPulseClient(HTTPClient):
    """
    Enhanced QuantumPulse client with standardized error handling and monitoring.

    This client demonstrates the improved patterns:
    - Automatic retry with exponential backoff
    - Circuit breaker for service protection
    - Comprehensive error handling and logging
    - Request/response correlation tracking
    - Metrics collection
    """

    def __init__(self, config: QuantumPulseConfig):
        super().__init__(config)
        self.qp_config = config

    async def submit_inference(self, request: InferenceRequest) -> str:
        """
        Submit an inference request to QuantumPulse.

        Args:
            request: Inference request object

        Returns:
            Request ID for tracking

        Raises:
            Q2ValidationError: If request validation fails
            Q2ExternalServiceError: If submission fails
        """
        self.logger.info(
            "Submitting inference request",
            model=request.model,
            request_id=request.request_id,
        )

        # Validate request
        self._validate_inference_request(request)

        try:
            response = await self.post("/v1/inference", data=request.dict())

            if response.status_code != 202:
                raise Q2ExternalServiceError(
                    "Unexpected response status for inference submission",
                    context={
                        "expected_status": 202,
                        "actual_status": response.status_code,
                        "response_data": response.data,
                    },
                )

            self.logger.info(
                "Inference request submitted successfully",
                request_id=request.request_id,
                response_data=response.data,
            )

            return request.request_id

        except Exception as e:
            self.logger.error(
                "Failed to submit inference request",
                exception=e,
                request_id=request.request_id,
            )
            raise

    async def get_chat_completion(self, request: QPChatRequest) -> QPChatResponse:
        """
        Get a chat completion from QuantumPulse.

        Args:
            request: Chat completion request

        Returns:
            Chat completion response

        Raises:
            Q2ValidationError: If request validation fails
            Q2ExternalServiceError: If completion fails
        """
        self.logger.info(
            "Requesting chat completion",
            model=request.model,
            message_count=len(request.messages),
        )

        # Validate request
        self._validate_chat_request(request)

        try:
            response = await self.post("/v1/chat/completions", data=request.dict())

            if not response.is_success:
                raise Q2ExternalServiceError(
                    "Chat completion request failed",
                    context={
                        "status_code": response.status_code,
                        "response_data": response.data,
                    },
                )

            # Parse response into model
            try:
                completion = QPChatResponse(**response.data)

                self.logger.info(
                    "Chat completion received",
                    completion_tokens=completion.usage.completion_tokens if completion.usage else None,
                    finish_reason=completion.choices[0].finish_reason if completion.choices else None,
                )

                return completion

            except (KeyError, TypeError, ValueError) as e:
                raise Q2ExternalServiceError(
                    "Invalid chat completion response format",
                    context={"response_data": response.data},
                    cause=e,
                )

        except Exception as e:
            self.logger.error("Failed to get chat completion", exception=e)
            raise

    async def stream_chat_completion(self, request: QPChatRequest) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from QuantumPulse.

        Args:
            request: Chat completion request with stream=True

        Yields:
            Streaming completion chunks

        Raises:
            Q2ValidationError: If request validation fails
            Q2ExternalServiceError: If streaming fails
        """
        self.logger.info(
            "Starting streaming chat completion",
            model=request.model,
            message_count=len(request.messages),
        )

        # Ensure streaming is enabled
        request.stream = True

        # Validate request
        self._validate_chat_request(request)

        try:
            # Use longer timeout for streaming
            streaming_response = await self.stream(
                "POST",
                "/v1/chat/completions",
                data=request.dict(),
                timeout=self.qp_config.stream_timeout,
            )

            chunk_count = 0
            async for chunk in streaming_response:
                chunk_count += 1
                yield chunk

            self.logger.info(
                "Streaming chat completion completed",
                chunks_received=chunk_count,
            )

        except Exception as e:
            self.logger.error("Failed to stream chat completion", exception=e)
            raise

    async def get_models(self) -> Dict[str, Any]:
        """
        Get available models from QuantumPulse.

        Returns:
            Dictionary of available models and their capabilities
        """
        self.logger.debug("Fetching available models")

        try:
            response = await self.get("/v1/models")

            if not response.is_success:
                raise Q2ExternalServiceError(
                    "Failed to fetch models",
                    context={
                        "status_code": response.status_code,
                        "response_data": response.data,
                    },
                )

            self.logger.debug(
                "Models fetched successfully",
                model_count=len(response.data.get("data", [])) if isinstance(response.data, dict) else None,
            )

            return response.data

        except Exception as e:
            self.logger.error("Failed to fetch models", exception=e)
            raise

    def _validate_inference_request(self, request: InferenceRequest) -> None:
        """Validate inference request."""
        if not request.model:
            raise Q2ValidationError(
                "Model is required for inference request",
                context={"request_id": request.request_id},
                suggestions=["Specify a valid model name"],
            )

        if not request.prompt and not request.messages:
            raise Q2ValidationError(
                "Either prompt or messages must be provided",
                context={"request_id": request.request_id},
                suggestions=["Provide a prompt string or messages array"],
            )

    def _validate_chat_request(self, request: QPChatRequest) -> None:
        """Validate chat completion request."""
        if not request.model:
            raise Q2ValidationError(
                "Model is required for chat request",
                suggestions=["Specify a valid model name"],
            )

        if not request.messages:
            raise Q2ValidationError(
                "Messages are required for chat request",
                suggestions=["Provide at least one message"],
            )

        # Check token limits
        if request.max_tokens and request.max_tokens > self.qp_config.max_chat_tokens:
            raise Q2ValidationError(
                f"Max tokens exceeds limit of {self.qp_config.max_chat_tokens}",
                context={"requested_tokens": request.max_tokens},
                suggestions=[f"Use max_tokens <= {self.qp_config.max_chat_tokens}"],
            )

        # Validate temperature
        if request.temperature is not None and not (0.0 <= request.temperature <= 2.0):
            raise Q2ValidationError(
                "Temperature must be between 0.0 and 2.0",
                context={"temperature": request.temperature},
                suggestions=["Use temperature between 0.0 and 2.0"],
            )

    async def health_check(self) -> bool:
        """Perform QuantumPulse-specific health check."""
        try:
            # Try to get models as a health check
            models = await self.get_models()
            return isinstance(models, dict) and "data" in models

        except Exception as e:
            self.logger.warning("QuantumPulse health check failed", exception=e)
            return False


# Convenience function for creating QuantumPulse client


def create_quantum_pulse_client(
    base_url: str,
    auth_token: Optional[str] = None,
    timeout: float = 60.0,
    **kwargs,
) -> EnhancedQuantumPulseClient:
    """
    Create a QuantumPulse client with standard configuration.

    Args:
        base_url: Base URL for QuantumPulse service
        auth_token: Authentication token
        timeout: Request timeout
        **kwargs: Additional configuration options

    Returns:
        Configured EnhancedQuantumPulseClient instance
    """
    config = QuantumPulseConfig(
        service_name="quantum-pulse",
        base_url=base_url,
        auth_token=auth_token,
        timeout=timeout,
        **kwargs,
    )
    return EnhancedQuantumPulseClient(config)
