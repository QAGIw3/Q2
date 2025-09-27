"""
HTTP client implementation for Q2 Platform services.

Provides standardized HTTP client with authentication, compression,
and streaming support built on top of the base client.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, AsyncGenerator
import httpx

from .base import BaseQ2Client, ClientConfig, ClientResponse, StreamingResponse
from ..error_handling import (
    Q2NetworkError,
    Q2TimeoutError,
    Q2AuthenticationError,
    Q2AuthorizationError,
    Q2ValidationError,
    Q2ResourceNotFoundError,
    Q2RateLimitError,
    Q2ExternalServiceError,
)


@dataclass
class HTTPClientConfig(ClientConfig):
    """Configuration for HTTP clients."""

    # Authentication
    auth_token: Optional[str] = None
    auth_type: str = "Bearer"  # Bearer, Basic, etc.
    username: Optional[str] = None
    password: Optional[str] = None

    # Request configuration
    follow_redirects: bool = True
    verify_ssl: bool = True
    compression: bool = True
    user_agent: str = "Q2-Platform-Client/1.0"

    # Connection pooling
    connection_pool_size: int = 100
    max_keepalive_connections: int = 20

    # Additional headers
    default_headers: Dict[str, str] = field(default_factory=dict)

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}

        if self.auth_token:
            headers["Authorization"] = f"{self.auth_type} {self.auth_token}"
        elif self.username and self.password:
            import base64

            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    def get_default_headers(self) -> Dict[str, str]:
        """Get all default headers including auth."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.compression:
            headers["Accept-Encoding"] = "gzip, deflate"

        # Add authentication headers
        headers.update(self.get_auth_headers())

        # Add any additional default headers
        headers.update(self.default_headers)

        # Add base headers from parent config
        headers.update(self.headers)

        return headers


class HTTPClient(BaseQ2Client):
    """
    HTTP client implementation with comprehensive error handling.

    Provides standardized HTTP communication with automatic:
    - Status code to exception mapping
    - Authentication handling
    - Request/response logging
    - Compression support
    - Connection pooling
    """

    def __init__(self, config: HTTPClientConfig):
        super().__init__(config)
        self.http_config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_connections=self.http_config.connection_pool_size,
                max_keepalive_connections=self.http_config.max_keepalive_connections,
            )

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                limits=limits,
                follow_redirects=self.http_config.follow_redirects,
                verify=self.http_config.verify_ssl,
                headers=self.http_config.get_default_headers(),
            )

        return self._client

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> ClientResponse:
        """Make HTTP request with comprehensive error handling."""
        client = await self._get_client()

        # Prepare request data
        request_kwargs = {
            "url": path,
            "params": params,
            **kwargs,
        }

        # Add headers
        if headers:
            merged_headers = self.http_config.get_default_headers().copy()
            merged_headers.update(headers)
            request_kwargs["headers"] = merged_headers

        # Handle request body
        if data is not None:
            if isinstance(data, dict):
                request_kwargs["json"] = data
            elif isinstance(data, str):
                request_kwargs["content"] = data
            else:
                request_kwargs["data"] = data

        try:
            response = await client.request(method, **request_kwargs)

            # Check for HTTP errors and convert to appropriate Q2 exceptions
            self._handle_http_status(response)

            # Parse response data
            response_data = await self._parse_response_data(response)

            return ClientResponse(
                data=response_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                request_id=response.headers.get("x-request-id"),
                correlation_id=response.headers.get("x-correlation-id"),
            )

        except httpx.TimeoutException as e:
            raise Q2TimeoutError(
                f"Request to {self.config.service_name} timed out",
                context={"method": method, "path": path, "timeout": self.config.timeout},
                cause=e,
            )

        except httpx.NetworkError as e:
            raise Q2NetworkError(
                f"Network error communicating with {self.config.service_name}",
                context={"method": method, "path": path, "error": str(e)},
                cause=e,
            )

        except httpx.HTTPStatusError as e:
            # This should be handled by _handle_http_status, but just in case
            raise Q2ExternalServiceError(
                f"HTTP error from {self.config.service_name}",
                context={
                    "method": method,
                    "path": path,
                    "status_code": e.response.status_code,
                    "response_text": e.response.text,
                },
                cause=e,
            )

    def _handle_http_status(self, response: httpx.Response) -> None:
        """Convert HTTP status codes to appropriate Q2 exceptions."""
        if response.is_success:
            return

        error_context = {
            "status_code": response.status_code,
            "url": str(response.url),
            "method": response.request.method,
        }

        # Try to extract error details from response
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                error_context.update(error_data)
        except Exception:
            error_context["response_text"] = response.text

        if response.status_code == 400:
            raise Q2ValidationError(
                "Bad request to service",
                context=error_context,
                suggestions=["Check request format and required fields"],
            )
        elif response.status_code == 401:
            raise Q2AuthenticationError(
                "Authentication failed",
                context=error_context,
                suggestions=["Check authentication credentials", "Verify token is not expired"],
            )
        elif response.status_code == 403:
            raise Q2AuthorizationError(
                "Access denied by service",
                context=error_context,
                suggestions=["Check user permissions", "Verify required roles"],
            )
        elif response.status_code == 404:
            raise Q2ResourceNotFoundError(
                "Resource not found",
                context=error_context,
                suggestions=["Check resource ID or path", "Verify resource exists"],
            )
        elif response.status_code == 429:
            retry_after = response.headers.get("retry-after", "60")
            raise Q2RateLimitError(
                "Rate limit exceeded",
                context={**error_context, "retry_after": retry_after},
                suggestions=[f"Wait {retry_after} seconds before retrying", "Reduce request frequency"],
            )
        elif 500 <= response.status_code < 600:
            raise Q2ExternalServiceError(
                f"Server error from {self.config.service_name}",
                context=error_context,
                suggestions=["Retry the request", "Check service health", "Contact service administrator"],
            )
        else:
            raise Q2ExternalServiceError(
                f"Unexpected HTTP status from {self.config.service_name}",
                context=error_context,
            )

    async def _parse_response_data(self, response: httpx.Response) -> Any:
        """Parse response data based on content type."""
        content_type = response.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            try:
                return response.json()
            except json.JSONDecodeError as e:
                raise Q2ExternalServiceError(
                    "Invalid JSON response from service",
                    context={"content_type": content_type, "response_text": response.text[:500]},
                    cause=e,
                )
        elif "text/" in content_type:
            return response.text
        else:
            return response.content

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> ClientResponse:
        """Make GET request."""
        return await self.request("GET", path, params=params, **kwargs)

    async def post(
        self, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """Make POST request."""
        return await self.request("POST", path, data=data, params=params, **kwargs)

    async def put(
        self, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """Make PUT request."""
        return await self.request("PUT", path, data=data, params=params, **kwargs)

    async def patch(
        self, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ClientResponse:
        """Make PATCH request."""
        return await self.request("PATCH", path, data=data, params=params, **kwargs)

    async def delete(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> ClientResponse:
        """Make DELETE request."""
        return await self.request("DELETE", path, params=params, **kwargs)

    async def stream(
        self, method: str, path: str, data: Optional[Any] = None, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> StreamingResponse:
        """Make streaming request."""
        client = await self._get_client()

        request_kwargs = {
            "url": path,
            "params": params,
            **kwargs,
        }

        if data is not None:
            if isinstance(data, dict):
                request_kwargs["json"] = data
            else:
                request_kwargs["data"] = data

        try:
            async with client.stream(method, **request_kwargs) as response:
                self._handle_http_status(response)

                async def stream_generator():
                    async for chunk in response.aiter_text():
                        yield chunk

                return StreamingResponse(stream_generator(), dict(response.headers))

        except httpx.TimeoutException as e:
            raise Q2TimeoutError(
                f"Streaming request to {self.config.service_name} timed out",
                context={"method": method, "path": path},
                cause=e,
            )
        except httpx.NetworkError as e:
            raise Q2NetworkError(
                f"Network error in streaming request to {self.config.service_name}",
                context={"method": method, "path": path},
                cause=e,
            )

    async def health_check(self) -> bool:
        """Perform HTTP health check."""
        try:
            # Try common health check endpoints
            health_endpoints = ["/health", "/healthz", "/ping", "/status"]

            for endpoint in health_endpoints:
                try:
                    response = await self.get(endpoint)
                    if response.is_success:
                        return True
                except Q2ResourceNotFoundError:
                    # Try next endpoint
                    continue

            return False

        except Exception as e:
            self.logger.warning(f"Health check failed for {self.config.service_name}", exception=e)
            return False

    async def close(self) -> None:
        """Close HTTP client and clean up connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        await super().close()


# Convenience functions for creating HTTP clients


def create_http_client(
    service_name: str,
    base_url: str,
    auth_token: Optional[str] = None,
    timeout: float = 30.0,
    **kwargs,
) -> HTTPClient:
    """
    Create an HTTP client with common configuration.

    Args:
        service_name: Name of the service
        base_url: Base URL for the service
        auth_token: Authentication token
        timeout: Request timeout
        **kwargs: Additional configuration options

    Returns:
        Configured HTTPClient instance
    """
    config = HTTPClientConfig(
        service_name=service_name, base_url=base_url, auth_token=auth_token, timeout=timeout, **kwargs
    )
    return HTTPClient(config)


def create_authenticated_http_client(
    service_name: str, base_url: str, username: str, password: str, timeout: float = 30.0, **kwargs
) -> HTTPClient:
    """
    Create an HTTP client with basic authentication.

    Args:
        service_name: Name of the service
        base_url: Base URL for the service
        username: Username for basic auth
        password: Password for basic auth
        timeout: Request timeout
        **kwargs: Additional configuration options

    Returns:
        Configured HTTPClient instance
    """
    config = HTTPClientConfig(
        service_name=service_name,
        base_url=base_url,
        username=username,
        password=password,
        timeout=timeout,
        **kwargs,
    )
    return HTTPClient(config)
