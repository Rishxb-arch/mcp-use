from __future__ import annotations

import inspect
import logging
import os
import time
import weakref
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import mcp.server.lowlevel.server as lowlevel
import mcp.server.session as mcp_session
from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    AnyFunction,
    CallToolRequest,
    CompleteRequest,
    Completion,
    ContentBlock,
    GetPromptRequest,
    GetPromptResult,
    Icon,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    Prompt,
    ReadResourceRequest,
    Resource,
    ServerResult,
    SetLevelRequest,
    SubscribeRequest,
    Tool,
    UnsubscribeRequest,
)
from pydantic import AnyUrl

# Import auth components
from mcp_use.client.connectors.base import BaseConnector
from mcp_use.server.auth import AuthMiddleware, BearerAuthProvider
from mcp_use.server.context import Context as MCPContext
from mcp_use.server.logging import MCPLoggingMiddleware
from mcp_use.server.middleware import (
    Middleware,
    MiddlewareManager,
    ServerMiddlewareContext,
    TelemetryMiddleware,
)
from mcp_use.server.middleware.server_session import MiddlewareServerSession
from mcp_use.server.mount import MountRegistry, wrap_lifespan
from mcp_use.server.runner import ServerRunner
from mcp_use.server.types import TransportType
from mcp_use.server.utils.inspector import _inspector_index, _inspector_static
from mcp_use.server.utils.json_schema import simplify_optional_schema
from mcp_use.server.utils.routes import docs_ui, openmcp_json
from mcp_use.telemetry.telemetry import Telemetry, telemetry
from mcp_use.telemetry.utils import track_server_run_from_server

# Monkey patching for init request of middleware
mcp_session.ServerSession = MiddlewareServerSession
lowlevel.ServerSession = MiddlewareServerSession

if TYPE_CHECKING:
    from mcp.server.session import ServerSession

    from mcp_use.server.router import MCPRouter


logger = logging.getLogger(__name__)
_telemetry = Telemetry()


def _normalize_inspector_path(path: str) -> str:
    """Normalize inspector path input into a concrete route ending in `/inspector`."""
    if not path or path == "/":
        return "/inspector"

    normalized = f"/{path.strip('/')}"
    if normalized.endswith("/inspector"):
        return normalized
    return f"{normalized}/inspector"


class MCPServer(FastMCP):
    """Main MCP Server class with integrated inspector and development tools."""

    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        instructions: str | None = None,
        icons: list[Icon] | None = None,
        auth: BearerAuthProvider | None = None,
        middleware: list[Middleware] | None = None,
        debug: bool = False,
        mcp_path: str = "/mcp",
        docs_path: str = "/docs",
        inspector_path: str = "/",
        openmcp_path: str = "/openmcp.json",
        show_inspector_logs: bool = False,
        pretty_print_jsonrpc: bool = False,
        mcp_logs_only: bool = False,
        host: str = "0.0.0.0",
        port: int = 8000,
        dns_rebinding_protection: bool = False,
    ):
        """Initialize an MCP server.

        Args:
            name: Server name for identification
            version: Server version string
            instructions: Instructions for the AI model using this server
            icons: Optional list of icons for the server implementation
            middleware: List of middleware to apply to requests
            debug: Enable debug mode (adds /docs, /inspector, /openmcp.json endpoints)
            mcp_path: Path for MCP endpoint (default: "/mcp")
            docs_path: Path for documentation endpoint (default: "/docs")
            inspector_path: Base path prefix for the inspector UI; final route is `<prefix>/inspector`.
                  Examples: `/` -> `/inspector`, `/mcp` -> `/mcp/inspector`.
                  Passing `/mcp/inspector` is also accepted for backward compatibility.
            openmcp_path: Path for OpenMCP metadata (default: "/openmcp.json")
            show_inspector_logs: Show inspector-related logs
            pretty_print_jsonrpc: Pretty print JSON-RPC messages in logs
            mcp_logs_only: Only show MCP protocol logs, suppress HTTP access logs (default: False)
            host: Default host for server binding (default: "0.0.0.0"). Can be overridden in run().
            port: Default port for server binding (default: 8000). Can be overridden in run().
            dns_rebinding_protection: Enable DNS rebinding protection by validating Host/Origin
                  headers. When True, only requests from localhost origins are accepted.
                  Recommended for local development servers. Default: False.
        """
        self._start_time = time.time()
        self._dns_rebinding_protection = dns_rebinding_protection
        super().__init__(
            name=name or "mcp-use server",
            instructions=instructions,
            icons=icons,
            host=host,
            port=port,
        )
        self._mounts = MountRegistry()

        # Apply DNS rebinding protection if requested
        if dns_rebinding_protection:
            self._apply_dns_rebinding_protection(host)

        if version:
            self._mcp_server.version = version

        # Register default protocol handlers for full MCP spec compliance
        self._register_default_protocol_handlers()

        # Store auth provider
        self._auth = auth

        # Set debug level: DEBUG env var takes precedence, then debug parameter
        env_debug_level = self._parse_debug_level()
        if env_debug_level > 0:
            # Environment variable overrides parameter
            self.debug_level = env_debug_level
        else:
            # Use debug parameter (0 or 1)
            self.debug_level = 1 if debug else 0

        # Set route paths
        self.mcp_path = mcp_path
        self.docs_path = docs_path
        self.inspector_path = _normalize_inspector_path(inspector_path)
        self.openmcp_path = openmcp_path
        self.show_inspector_logs = show_inspector_logs
        self.pretty_print_jsonrpc = pretty_print_jsonrpc
        self.mcp_logs_only = mcp_logs_only
        self._transport_type: TransportType = "streamable-http"

        self.middleware_manager = MiddlewareManager()
        self.middleware_manager.add_middleware(TelemetryMiddleware())

        if middleware:
            for middleware_instance in middleware:
                self.middleware_manager.add_middleware(middleware_instance)

        # Add dev routes only in DEBUG=1 and above
        if self.debug_level >= 1:
            self._add_dev_routes()

        self.app = self.streamable_http_app()

        # Inject middleware in the ServerSession
        MiddlewareServerSession._middleware_manager = self.middleware_manager
        MiddlewareServerSession._transport_type = self._transport_type

    @property
    def debug(self) -> bool:
        """Whether debug mode is enabled."""
        return self.debug_level >= 1

    def _parse_debug_level(self) -> int:
        """Parse DEBUG environment variable to get debug level.

        Returns:
            0: Production mode (clean logs only)
            1: Debug mode (clean logs + dev routes)
            2: Full debug mode (clean logs + dev routes + JSON-RPC logging)
        """
        debug_env = os.environ.get("DEBUG", "0")
        try:
            level = int(debug_env)
            return max(0, min(2, level))  # Clamp between 0-2
        except ValueError:
            # Handle string values
            if debug_env.lower() in ("1", "true", "yes"):
                return 1
            elif debug_env.lower() in ("2", "full", "verbose"):
                return 2
            else:
                return 0

    def _add_dev_routes(self):
        """Add development routes for debugging and inspection."""

        # OpenMCP configuration
        async def openmcp_handler(request):
            return await openmcp_json(request, self)

        self.custom_route(self.openmcp_path, methods=["GET"])(openmcp_handler)

        # Documentation UI
        self.custom_route(self.docs_path, methods=["GET"])(docs_ui)

        # Inspector routes - wrap to pass mcp_path
        async def inspector_index_handler(request):
            return await _inspector_index(
                request,
                mcp_path=self.mcp_path,
                inspector_path=self.inspector_path,
            )

        self.custom_route(self.inspector_path, methods=["GET"])(inspector_index_handler)
        self.custom_route(f"{self.inspector_path}/{{path:path}}", methods=["GET"])(_inspector_static)

    def add_tool(
        self,
        fn: AnyFunction,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: Any = None,
        icons: Any = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ) -> None:
        """Register a tool, simplifying Pydantic's nullable ``anyOf`` schemas.

        Pydantic emits ``{"anyOf": [{"type": "T"}, {"type": "null"}]}`` for
        ``Optional[T]`` fields.  MCP expresses optionality by omitting the
        property from ``required``, so the ``anyOf``/null wrapper is redundant
        and breaks description rendering in several MCP clients.

        This override lets the upstream ``FastMCP`` build the tool as usual,
        then rewrites its ``parameters`` schema into the simpler form.
        """
        super().add_tool(
            fn,
            name=name,
            title=title,
            description=description,
            annotations=annotations,
            icons=icons,
            meta=meta,
            structured_output=structured_output,
        )
        # The tool was registered under its resolved name — look it up and
        # simplify the schema that Pydantic generated.
        tool_name = name or fn.__name__
        tool = self._tool_manager.get_tool(tool_name)
        if tool is not None:
            tool.parameters = simplify_optional_schema(tool.parameters)

    @telemetry("server_router_used")
    def include_router(self, router: MCPRouter, prefix: str = "", enabled: bool = True) -> None:
        """
        Include a router's tools, resources, and prompts into this server.

        Similar to FastAPI's include_router, this allows you to organize your
        MCP server into multiple files/modules.

        Args:
            router: The MCPRouter instance to include
            prefix: Optional prefix to add to all tool names (e.g., "math" -> "math_add")
            enabled: Whether to enable this router (default True). Set to False to skip registration.

        Example:
            ```python
            from mcp_use.server import MCPServer, MCPRouter

            # In routes/math.py
            router = MCPRouter()

            @router.tool()
            def add(a: int, b: int) -> int:
                return a + b

            # In main.py
            server = MCPServer(name="my-server")
            server.include_router(router, prefix="math")  # Tool becomes "math_add"
            server.include_router(other_router, enabled=False)  # Skip this router
            ```
        """
        if not enabled:
            return
        # Register all tools from the router
        for tool in router.tools:
            tool_name = tool.name or getattr(tool.fn, "__name__", "unknown")
            if prefix:
                tool_name = f"{prefix}_{tool_name}"

            self.add_tool(
                tool.fn,
                name=tool_name,
                title=tool.title,
                description=tool.description,
                annotations=tool.annotations,
                structured_output=tool.structured_output,
            )

        # Register all resources from the router
        for resource in router.resources:
            resource_name = resource.name or getattr(resource.fn, "__name__", "unknown")
            if prefix:
                resource_name = f"{prefix}_{resource_name}"
            self.resource(
                uri=resource.uri,
                name=resource.name,
                description=resource.description,
                mime_type=resource.mime_type,
            )(resource.fn)

        # Register all prompts from the router
        for prompt in router.prompts:
            prompt_name = prompt.name or getattr(prompt.fn, "__name__", "unknown")
            if prefix:
                prompt_name = f"{prefix}_{prompt_name}"

            self.prompt(
                name=prompt_name,
                description=prompt.description,
            )(prompt.fn)

    @telemetry("server_mount_used")
    def mount(
        self,
        connector: BaseConnector,
        prefix: str | None = None,
        *,
        resources: bool = True,
        prompts: bool = True,
    ) -> None:
        """Expose a connector's tools, and optionally resources and prompts, on this server.

        Call this before ``run()``. Mounts added later are connected on the next
        request and when the server lifespan starts. The connector is connected
        for the server lifetime and disconnected on shutdown.

        A prefix becomes ``{prefix}_{name}`` for tools and prompts, the same
        separator ``include_router`` uses. Prefixes must match ``[A-Za-z0-9_]``
        and are rejected, not rewritten, when they do not. Omit ``prefix`` to
        keep upstream names. Colliding names are an error when the catalog is
        built, not a silent override.

        The caller's ``Authorization`` header is not forwarded. Configure
        credentials on the connector itself.

        Mounted ``call_tool`` results are :class:`~mcp.types.CallToolResult`
        objects (content, structured content, and ``isError``). Local tools
        keep the existing FastMCP return value.

        Example:
            ```python
            from mcp_use import MCPServer
            from mcp_use.client.connectors import HttpConnector

            server = MCPServer(name="composed")
            server.mount(HttpConnector("http://127.0.0.1:9000/mcp"), prefix="remote")

            @server.tool()
            async def shout(text: str) -> str:
                result = await server.call_tool("remote_echo", {"text": text})
                return result.content[0].text.upper()
            ```
        """
        self._mounts.add(connector, prefix, resources=resources, prompts=prompts)

    async def list_tools(self) -> list[Tool]:
        """List local tools plus tools from mounted connectors."""
        local = await super().list_tools()
        if not self._mounts:
            return local
        await self._mounts.connect_all()
        mounted = await self._mounts.exposed_tools({tool.name for tool in local})
        return [*local, *mounted]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        """Call a local tool, or a mounted tool by its exposed name.

        Mounted tools return a :class:`~mcp.types.CallToolResult` at runtime
        (``isError`` and structured content included). The declared return type
        matches FastMCP. Local tools still return FastMCP's ``(content, structured)``
        tuple.
        """
        if self._mounts:
            await self._mounts.connect_all()
            local_names = {tool.name for tool in await super().list_tools()}
            mounted = await self._mounts.call_tool(name, arguments or {}, local_names)
            if mounted is not None:
                return cast(Sequence[ContentBlock] | dict[str, Any], mounted)
        return await super().call_tool(name, arguments)

    async def list_resources(self) -> list[Resource]:
        """List local resources plus resources from mounts that enable them."""
        local = await super().list_resources()
        if not self._mounts:
            return local
        await self._mounts.connect_all()
        mounted = await self._mounts.exposed_resources({str(resource.uri) for resource in local})
        return [*local, *mounted]

    async def read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        """Read a local resource, or a mounted one by its exposed URI."""
        if self._mounts:
            await self._mounts.connect_all()
            local = await super().list_resources()
            # Build the catalog first so URI collisions fail closed, same as list_resources.
            await self._mounts.exposed_resources({str(resource.uri) for resource in local})
            mounted = await self._mounts.read_resource(str(uri))
            if mounted is not None:
                return mounted
        return await super().read_resource(uri)

    async def list_prompts(self) -> list[Prompt]:
        """List local prompts plus prompts from mounts that enable them."""
        local = await super().list_prompts()
        if not self._mounts:
            return local
        await self._mounts.connect_all()
        mounted = await self._mounts.exposed_prompts({prompt.name for prompt in local})
        return [*local, *mounted]

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> GetPromptResult:
        """Render a local prompt, or a mounted one by its exposed name."""
        if self._mounts:
            await self._mounts.connect_all()
            local_names = {prompt.name for prompt in await super().list_prompts()}
            mounted = await self._mounts.get_prompt(name, arguments, local_names)
            if mounted is not None:
                return mounted
        return await super().get_prompt(name, arguments)

    async def run_stdio_async(self) -> None:
        """Run stdio, connecting mounted connectors for the process lifetime."""
        try:
            await self._mounts.connect_all()
            await super().run_stdio_async()
        finally:
            await self._mounts.disconnect_all()

    def _apply_dns_rebinding_protection(self, host: str) -> None:
        """Configure transport security to reject non-localhost Host/Origin headers."""
        from mcp.server.transport_security import TransportSecuritySettings

        localhost_host = host if host in ("127.0.0.1", "localhost", "::1") else "127.0.0.1"
        self.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{localhost_host}:*", "localhost:*", "127.0.0.1:*", "[::1]:*"],
            allowed_origins=[
                f"http://{localhost_host}:*",
                "http://localhost:*",
                "http://127.0.0.1:*",
                "http://[::1]:*",
            ],
        )
        # Reset session manager so it picks up new security settings
        self._session_manager = None

    def _register_default_protocol_handlers(self) -> None:
        """Register default handlers for MCP protocol methods not covered by FastMCP.

        Registers logging/setLevel, resources/subscribe, resources/unsubscribe,
        and completion/complete so the server advertises these capabilities and
        responds to requests correctly.
        """

        # logging/setLevel — store the level so Context.log() can filter
        self._client_log_level: str = "debug"  # Default: send all levels

        @self._mcp_server.set_logging_level()
        async def _handle_set_logging_level(level: Any) -> None:
            self._client_log_level = str(level)

        # resources/subscribe + unsubscribe — track per-URI subscriptions
        # Uses WeakSet so that subscriptions are automatically cleaned up when
        # a client disconnects and its session object is garbage-collected,
        # preventing the memory leak described in GitHub issue #1092.
        self._resource_subscriptions: dict[str, weakref.WeakSet] = {}  # uri -> WeakSet of sessions

        @self._mcp_server.subscribe_resource()
        async def _handle_subscribe(uri: Any) -> None:
            session = self._current_session()
            if session:
                self._resource_subscriptions.setdefault(str(uri), weakref.WeakSet()).add(session)

        @self._mcp_server.unsubscribe_resource()
        async def _handle_unsubscribe(uri: Any) -> None:
            session = self._current_session()
            if session:
                subscribers = self._resource_subscriptions.get(str(uri))
                if subscribers:
                    subscribers.discard(session)
                    if not subscribers:
                        del self._resource_subscriptions[str(uri)]

        # Patch capabilities to advertise subscribe support
        original_get_capabilities = self._mcp_server.get_capabilities

        def _patched_get_capabilities(notification_options: Any, experimental_capabilities: Any) -> Any:
            caps = original_get_capabilities(notification_options, experimental_capabilities)
            if caps.resources is not None:
                caps.resources.subscribe = True
            return caps

        self._mcp_server.get_capabilities = _patched_get_capabilities  # type: ignore[assignment]

        # completion/complete — default empty completions (override via mcp.completion())
        @self._mcp_server.completion()
        async def _handle_completion(_ref: Any, _argument: Any, _context: Any = None) -> Completion:
            return Completion(values=[], total=0, hasMore=False)

    async def notify_resource_updated(self, uri: str) -> None:
        """Notify all subscribed sessions that a resource has been updated.

        Broadcasts a ``notifications/resources/updated`` message to every session
        that called ``resources/subscribe`` for this URI.

        Can be called from within a tool handler or from any async context.

        Args:
            uri: The URI of the resource that was updated.
        """
        subscribers = self._resource_subscriptions.get(uri)
        if not subscribers:
            # Clean up empty entry left behind after all sessions were GC'd
            self._resource_subscriptions.pop(uri, None)
            return

        # Iterate a snapshot; the WeakSet may shrink during iteration
        # if sessions are garbage-collected concurrently.
        for session in list(subscribers):
            try:
                await session.send_resource_updated(uri=uri)
            except Exception:
                pass  # Session may have disconnected

    def streamable_http_app(self):
        """Override to add our custom middleware."""
        from starlette.middleware.cors import CORSMiddleware

        app = super().streamable_http_app()

        # Add MCP logging middleware (cast to satisfy type checker)
        app.add_middleware(
            cast(type, MCPLoggingMiddleware),
            debug_level=self.debug_level,
            mcp_path=self.mcp_path,
            pretty_print_jsonrpc=self.pretty_print_jsonrpc,
        )

        # Add auth middleware if provider is configured
        if self._auth:
            app.add_middleware(
                AuthMiddleware,
                auth_provider=self._auth,
            )
            logger.debug("AuthMiddleware added to application")

            # Add CORS middleware when auth is enabled (handles OPTIONS preflight)
            # Note: allow_credentials=False because allow_origins=["*"] - browsers reject the combination
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
                expose_headers=["WWW-Authenticate"],
            )

        app.router.lifespan_context = wrap_lifespan(self._mounts, app.router.lifespan_context)
        return app

    def _wrap_handlers_with_middleware(self) -> None:
        handlers = self._mcp_server.request_handlers

        if self.debug_level >= 1:
            logger.debug(f"Wrapping handlers. Available handlers: {list(handlers.keys())}")

        def wrap_request(request_cls: type, method: str) -> None:
            if request_cls not in handlers:
                return

            original = handlers[request_cls]

            async def wrapped(request: Any) -> ServerResult:
                # Get session ID from HTTP headers if available
                session_id = self._get_session_id_from_request()

                context = ServerMiddlewareContext(
                    message=request.params,
                    method=method,
                    timestamp=datetime.now(UTC),
                    transport=self._transport_type,
                    session_id=session_id,
                )

                async def call_original(_: ServerMiddlewareContext[Any]) -> Any:
                    return await original(request)

                return await self.middleware_manager.process_request(context, call_original)

            handlers[request_cls] = wrapped

        wrap_request(CallToolRequest, "tools/call")
        wrap_request(ReadResourceRequest, "resources/read")
        wrap_request(GetPromptRequest, "prompts/get")
        wrap_request(ListToolsRequest, "tools/list")
        wrap_request(ListResourcesRequest, "resources/list")
        wrap_request(ListPromptsRequest, "prompts/list")
        wrap_request(SetLevelRequest, "logging/setLevel")
        wrap_request(SubscribeRequest, "resources/subscribe")
        wrap_request(UnsubscribeRequest, "resources/unsubscribe")
        wrap_request(CompleteRequest, "completion/complete")

    def run(  # type: ignore[override]
        self,
        transport: TransportType = "streamable-http",
        host: str | None = None,
        port: int | None = None,
        reload: bool = False,
        debug: bool = False,
    ) -> None:
        """Run the MCP server.

        Args:
            transport: Transport protocol to use ("stdio", "streamable-http" or "sse")
            host: Host to bind to. If provided, overrides __init__ value and reconfigures
                  DNS rebinding protection accordingly.
            port: Port to bind to. If not provided, uses the value from __init__.
            reload: Whether to enable auto-reload
            debug: Whether to enable debug mode. Overrides the server's debug setting,
                   adds /docs and /openmcp.json endpoints if not already added.
        """
        # Use settings from __init__, run() values override
        final_host = host if host is not None else self.settings.host
        final_port = port if port is not None else self.settings.port

        # If host changed, update settings and rebuild app
        if final_host != self.settings.host:
            self.settings.host = final_host
            self.settings.port = final_port
            if self._dns_rebinding_protection:
                self._apply_dns_rebinding_protection(final_host)
            # Rebuild app with updated settings
            self._session_manager = None
            self.app = self.streamable_http_app()

        # Override debug_level if debug=True is passed to run()
        if debug and self.debug_level < 1:
            self.debug_level = 1
            self._add_dev_routes()
            # Rebuild the Starlette app so the new routes are included
            self._session_manager = None
            self.app = self.streamable_http_app()

        self._transport_type = transport
        track_server_run_from_server(self, transport, final_host, final_port, _telemetry)

        self._wrap_handlers_with_middleware()

        runner = ServerRunner(self)
        runner.run(transport=transport, host=final_host, port=final_port, reload=reload)

    def get_context(self) -> MCPContext:  # type: ignore[override]
        """Use the extended MCP-Use context that adds convenience helpers."""
        return MCPContext(request_context=self._get_request_context(), fastmcp=self)  # type: ignore[override]

    def _resource_is_template(self, fn: AnyFunction, uri: str) -> bool:
        has_uri_params = "{" in uri and "}" in uri
        if has_uri_params:
            return True
        return bool(inspect.signature(fn).parameters)

    def _current_session(self) -> ServerSession | None:
        request_context = self._get_request_context()
        if request_context is None:
            return None
        return request_context.session

    def _get_request_context(self):
        try:
            return self._mcp_server.request_context
        except LookupError:
            return None

    def _get_session_id(self) -> str | None:
        """Get session ID from the session object (deprecated - use _get_session_id_from_request)."""
        session = self._current_session()
        if session is None:
            return None

        try:
            return session.session_id  # type: ignore[attr-defined]
        except AttributeError:
            try:
                return session.id  # type: ignore[attr-defined]
            except AttributeError:
                return None

    def _get_session_id_from_request(self) -> str | None:
        """Get session ID from HTTP request headers (for Streamable HTTP transport).

        The session ID is managed at the transport layer and sent via the
        mcp-session-id header according to the MCP specification.
        """
        request_context = self._get_request_context()
        if request_context is None:
            return None

        # Try to get the HTTP request from context
        request = getattr(request_context, "request", None)
        if request is None:
            return None

        # Extract mcp-session-id header
        try:
            return request.headers.get("mcp-session-id")
        except (AttributeError, KeyError):
            return None
