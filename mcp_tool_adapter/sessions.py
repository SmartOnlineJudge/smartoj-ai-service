from typing import AsyncIterator, Any, Protocol, Literal
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from typing_extensions import TypedDict, NotRequired
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_STREAMABLE_HTTP_TIMEOUT = timedelta(seconds=30)
DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT = timedelta(seconds=60 * 5)


class McpHttpClientFactory(Protocol):
    """Protocol for creating httpx.AsyncClient instances for MCP connections."""

    def __call__(
        self,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient instance.

        Args:
            headers: HTTP headers to include in requests.
            timeout: Request timeout configuration.
            auth: Authentication configuration.

        Returns:
            Configured httpx.AsyncClient instance.
        """
        ...


class StreamableHttpConnection(TypedDict):
    """Connection configuration for Streamable HTTP transport."""

    transport: Literal["streamable_http"]

    url: str
    """The URL of the endpoint to connect to."""

    headers: NotRequired[dict[str, Any] | None]
    """HTTP headers to send to the endpoint."""

    timeout: NotRequired[timedelta]
    """HTTP timeout."""

    sse_read_timeout: NotRequired[timedelta]
    """How long (in seconds) the client will wait for a new event before disconnecting.
    All other HTTP operations are controlled by `timeout`."""

    terminate_on_close: NotRequired[bool]
    """Whether to terminate the session on close."""

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession."""

    httpx_client_factory: NotRequired[McpHttpClientFactory | None]
    """Custom factory for httpx.AsyncClient (optional)."""

    auth: NotRequired[httpx.Auth]
    """Optional authentication for the HTTP client."""


class WebsocketConnection(TypedDict):
    """Configuration for WebSocket transport connections to MCP servers."""

    transport: Literal["websocket"]

    url: str
    """The URL of the Websocket endpoint to connect to."""

    session_kwargs: NotRequired[dict[str, Any] | None]
    """Additional keyword arguments to pass to the ClientSession"""


Connection = StreamableHttpConnection | WebsocketConnection


@asynccontextmanager
async def _create_streamable_http_session(  # noqa: PLR0913
    *,
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: timedelta = DEFAULT_STREAMABLE_HTTP_TIMEOUT,
    sse_read_timeout: timedelta = DEFAULT_STREAMABLE_HTTP_SSE_READ_TIMEOUT,
    terminate_on_close: bool = True,
    session_kwargs: dict[str, Any] | None = None,
    httpx_client_factory: McpHttpClientFactory | None = None,
    auth: httpx.Auth | None = None,
) -> AsyncIterator[ClientSession]:
    """Create a new session to an MCP server using Streamable HTTP.

    Args:
        url: URL of the endpoint to connect to.
        headers: HTTP headers to send to the endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: How long the client will wait for a new event before
            disconnecting.
        terminate_on_close: Whether to terminate the session on close.
        session_kwargs: Additional keyword arguments to pass to the ClientSession.
        httpx_client_factory: Custom factory for httpx.AsyncClient (optional).
        auth: Authentication for the HTTP client.

    Yields:
        An initialized ClientSession.
    """
    # Create and store the connection
    kwargs = {}
    if httpx_client_factory is not None:
        kwargs["httpx_client_factory"] = httpx_client_factory

    async with (
        streamablehttp_client(
            url,
            headers,
            timeout,
            sse_read_timeout,
            terminate_on_close,
            auth=auth,
            **kwargs,
        ) as (read, write, _),
        ClientSession(read, write, **(session_kwargs or {})) as session,
    ):
        yield session


@asynccontextmanager
async def create_session(connection: Connection) -> AsyncIterator[ClientSession]:  # noqa: C901
    """Create a new session to an MCP server.

    Args:
        connection: Connection config to use to connect to the server

    Raises:
        ValueError: If transport is not recognized
        ValueError: If required parameters for the specified transport are missing

    Yields:
        A ClientSession
    """
    if "transport" not in connection:
        msg = (
            "Configuration error: Missing 'transport' key in server configuration. "
            "Each server must include 'transport' with one of: "
            "'stdio', 'sse', 'websocket', 'streamable_http'. "
            "Please refer to the langchain-mcp-adapters documentation for more details."
        )
        raise ValueError(msg)

    params = {k: v for k, v in connection.items() if k != "transport"}

    if "url" not in params:
        msg = "'url' parameter is required for Streamable HTTP connection"
        raise ValueError(msg)
    async with _create_streamable_http_session(**params) as session:
        yield session

