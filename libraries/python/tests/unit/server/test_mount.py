"""Unit tests for MCPServer.mount() using in-process fake connectors."""

import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    GetPromptResult,
    ListToolsRequest,
    Prompt,
    PromptMessage,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl

from mcp_use.client.connectors.base import BaseConnector
from mcp_use.server import MCPServer, MountError
from mcp_use.server.middleware import Middleware


def _tool(name: str, description: str = "A tool") -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )


def _result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


class FakeConnector(BaseConnector):
    """Connector that never opens a socket. list/call are served from memory."""

    def __init__(
        self,
        identifier: str,
        tools: list[Tool] | None = None,
        resources: list[Resource] | None = None,
        prompts: list[Prompt] | None = None,
    ):
        super().__init__()
        self._identifier = identifier
        self._tools = list(tools or [])
        self._resources = list(resources or [])
        self._prompts = list(prompts or [])
        self.headers: dict[str, str] = {}
        self.connect_calls = 0
        self.initialize_calls = 0
        self.disconnect_calls = 0
        self.calls: list[tuple[str, dict]] = []
        self.read_uris: list[str] = []
        self.prompt_calls: list[tuple[str, dict | None]] = []

    @property
    def public_identifier(self) -> str:
        return self._identifier

    async def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True

    async def initialize(self):  # type: ignore[override]
        self.initialize_calls += 1
        self._initialized = True
        return None

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False
        self._initialized = False

    async def list_tools(self) -> list[Tool]:
        return list(self._tools)

    async def list_resources(self) -> list[Resource]:
        return list(self._resources)

    async def list_prompts(self) -> list[Prompt]:
        return list(self._prompts)

    async def call_tool(self, name: str, arguments: dict, read_timeout_seconds=None) -> CallToolResult:
        self.calls.append((name, arguments))
        if name == "missing":
            raise RuntimeError("upstream failed")
        if name == "boom":
            return _result("nope", is_error=True)
        text = arguments.get("text", name)
        return _result(f"{self._identifier}:{text}")

    async def read_resource(self, uri: AnyUrl) -> ReadResourceResult:
        self.read_uris.append(str(uri))
        return ReadResourceResult(contents=[TextResourceContents(uri=uri, text=f"body:{uri}", mimeType="text/plain")])

    async def get_prompt(self, name: str, arguments: dict | None = None) -> GetPromptResult:
        self.prompt_calls.append((name, arguments))
        who = (arguments or {}).get("name", "there")
        return GetPromptResult(
            description=name,
            messages=[PromptMessage(role="user", content=TextContent(type="text", text=f"Hello {who}"))],
        )


def _server() -> MCPServer:
    return MCPServer(name="composer", host="127.0.0.1", debug=False)


def _names(tools: list[Tool]) -> list[str]:
    return [tool.name for tool in tools]


class TestMountRegistration:
    def test_rejects_empty_invalid_and_long_prefixes(self):
        server = _server()
        connector = FakeConnector("remote", tools=[_tool("echo")])

        with pytest.raises(MountError, match="non-empty"):
            server.mount(connector, prefix="")
        with pytest.raises(MountError, match="not rewritten"):
            server.mount(connector, prefix="browser-tools")
        with pytest.raises(MountError, match="maximum"):
            server.mount(connector, prefix="p" * 49)
        with pytest.raises(TypeError, match="BaseConnector"):
            server.mount(object(), prefix="remote")  # type: ignore[arg-type]

    def test_rejects_duplicate_prefix_and_connector(self):
        server = _server()
        first = FakeConnector("one", tools=[_tool("echo")])
        second = FakeConnector("two", tools=[_tool("echo")])
        server.mount(first, prefix="remote")

        with pytest.raises(MountError, match="already used"):
            server.mount(second, prefix="remote")
        with pytest.raises(MountError, match="already mounted"):
            server.mount(first, prefix="other")

    def test_digit_prefix_is_accepted(self):
        server = _server()
        server.mount(FakeConnector("remote", tools=[_tool("echo")]), prefix="v2")
        # Registration only; listing happens in the async tests.


class TestMountedTools:
    @pytest.mark.asyncio
    async def test_list_call_and_internal_orchestration(self):
        server = _server()
        remote = FakeConnector("remote", tools=[_tool("echo", "Repeat text")])
        other = FakeConnector("other", tools=[_tool("ping")])
        server.mount(remote, prefix="remote")
        server.mount(other, prefix="other")

        @server.tool()
        def local_only(text: str) -> str:
            """A tool defined on this server."""
            return f"local:{text}"

        @server.tool()
        async def shout(text: str) -> str:
            """Orchestrate a mounted tool from a local one."""
            result = await server.call_tool("remote_echo", {"text": text})
            assert isinstance(result, CallToolResult)
            return result.content[0].text.upper()

        tools = await server.list_tools()
        assert _names(tools) == ["local_only", "shout", "remote_echo", "other_ping"]
        echo = next(tool for tool in tools if tool.name == "remote_echo")
        assert echo.description == "Repeat text"
        assert echo.inputSchema["properties"]["text"]["type"] == "string"
        assert echo.meta["mcp-use/mount"]["upstream"] == "echo"
        # The connector's own tool object keeps its upstream name.
        assert remote._tools[0].name == "echo"

        mounted = await server.call_tool("remote_echo", {"text": "hi"})
        assert isinstance(mounted, CallToolResult)
        assert mounted.content[0].text == "remote:hi"
        assert remote.calls == [("echo", {"text": "hi"})]
        assert remote.headers == {}

        shouted = await server.call_tool("shout", {"text": "hi"})
        # Local tools keep FastMCP's (content, structured) tuple.
        assert shouted[0][0].text == "REMOTE:HI"

        local = await server.call_tool("local_only", {"text": "ok"})
        assert local[0][0].text == "local:ok"

    @pytest.mark.asyncio
    async def test_unprefixed_mount_and_name_collision(self):
        server = _server()

        @server.tool()
        def echo(text: str) -> str:
            return text

        server.mount(FakeConnector("remote", tools=[_tool("echo")]))
        with pytest.raises(MountError, match="collide"):
            await server.list_tools()

    @pytest.mark.asyncio
    async def test_two_mounts_cannot_expose_the_same_name(self):
        server = _server()
        server.mount(FakeConnector("one", tools=[_tool("echo")]))
        server.mount(FakeConnector("two", tools=[_tool("echo")]))
        with pytest.raises(MountError, match="echo"):
            await server.list_tools()

    @pytest.mark.asyncio
    async def test_error_result_is_preserved(self):
        server = _server()
        server.mount(FakeConnector("remote", tools=[_tool("boom")]), prefix="remote")
        result = await server.call_tool("remote_boom", {})
        assert isinstance(result, CallToolResult)
        assert result.isError is True
        assert result.content[0].text == "nope"

    @pytest.mark.asyncio
    async def test_unknown_mounted_name_falls_through_to_local_tools(self):
        server = _server()
        server.mount(FakeConnector("remote", tools=[_tool("echo")]), prefix="remote")
        with pytest.raises(Exception, match="Unknown tool"):
            await server.call_tool("not_a_tool", {})

    @pytest.mark.asyncio
    async def test_protocol_handlers_route_list_and_call(self):
        server = _server()
        remote = FakeConnector("remote", tools=[_tool("echo")])
        server.mount(remote, prefix="remote")
        server._wrap_handlers_with_middleware()

        listed = await server._mcp_server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
        assert [tool.name for tool in listed.root.tools] == ["remote_echo"]

        called = await server._mcp_server.request_handlers[CallToolRequest](
            CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="remote_echo", arguments={"text": "via-protocol"}),
            )
        )
        assert called.root.content[0].text == "remote:via-protocol"
        assert remote.calls[0][0] == "echo"

    @pytest.mark.asyncio
    async def test_middleware_sees_the_prefixed_catalog(self):
        seen: list[str] = []

        class RecordingMiddleware(Middleware):
            async def on_list_tools(self, context, call_next):
                result = await call_next(context)
                seen.extend(tool.name for tool in result.root.tools)
                return result

            async def on_call_tool(self, context, call_next):
                seen.append(context.message.name)
                return await call_next(context)

        server = MCPServer(name="composer", host="127.0.0.1", debug=False, middleware=[RecordingMiddleware()])
        server.mount(FakeConnector("remote", tools=[_tool("echo")]), prefix="remote")
        server._wrap_handlers_with_middleware()

        await server._mcp_server.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
        await server._mcp_server.request_handlers[CallToolRequest](
            CallToolRequest(method="tools/call", params=CallToolRequestParams(name="remote_echo", arguments={}))
        )
        assert seen == ["remote_echo", "remote_echo"]

    @pytest.mark.asyncio
    async def test_connects_once_under_concurrent_lists(self):
        server = _server()
        remote = FakeConnector("remote", tools=[_tool("echo")])
        server.mount(remote, prefix="remote")

        await asyncio.gather(server.list_tools(), server.list_tools(), server.list_tools())
        assert remote.connect_calls == 1
        assert remote.initialize_calls == 1

    @pytest.mark.asyncio
    async def test_follows_upstream_list_cursors(self):
        class Pager:
            def __init__(self):
                self.calls: list[str | None] = []

            async def list_tools(self, cursor=None):
                self.calls.append(cursor)
                if cursor is None:
                    return SimpleNamespace(tools=[_tool("one")], nextCursor="page-2")
                return SimpleNamespace(tools=[_tool("two")], nextCursor=None)

        server = _server()
        remote = FakeConnector("remote", tools=[_tool("ignored")])
        pager = Pager()
        remote.client_session = pager  # type: ignore[assignment]
        server.mount(remote, prefix="remote")

        tools = await server.list_tools()
        assert _names(tools) == ["remote_one", "remote_two"]
        assert pager.calls == [None, "page-2"]

    @pytest.mark.asyncio
    async def test_lifespan_connects_and_disconnects(self):
        server = _server()
        remote = FakeConnector("remote", tools=[_tool("echo")])
        server.mount(remote, prefix="remote")
        app = server.streamable_http_app()

        async with app.router.lifespan_context(app):
            assert remote.connect_calls == 1
            assert remote.initialize_calls == 1
            tools = await server.list_tools()
            assert _names(tools) == ["remote_echo"]
            # Already connected by the lifespan; listing must not connect again.
            assert remote.connect_calls == 1

        assert remote.disconnect_calls == 1

    @pytest.mark.asyncio
    async def test_stdio_run_connects_and_disconnects(self, monkeypatch: pytest.MonkeyPatch):
        server = _server()
        remote = FakeConnector("remote", tools=[_tool("echo")])
        server.mount(remote, prefix="remote")

        async def fake_stdio(self):
            assert remote.connect_calls == 1

        monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run_stdio_async", fake_stdio)
        await server.run_stdio_async()
        assert remote.disconnect_calls == 1

    @pytest.mark.asyncio
    async def test_repeated_cursor_does_not_loop(self):
        class Stuck:
            async def list_tools(self, cursor=None):
                return SimpleNamespace(tools=[_tool("echo")], nextCursor="stuck")

        server = _server()
        remote = FakeConnector("remote")
        remote.client_session = Stuck()  # type: ignore[assignment]
        server.mount(remote, prefix="remote")
        tools = await server.list_tools()
        assert _names(tools) == ["remote_echo"]


class TestMountedResourcesAndPrompts:
    @pytest.mark.asyncio
    async def test_resources_and_prompts_are_prefixed_and_routed(self):
        server = _server()
        remote = FakeConnector(
            "remote",
            resources=[Resource(name="note", uri=AnyUrl("note://hello"), description="A note")],
            prompts=[Prompt(name="greet", description="Say hello")],
        )
        server.mount(remote, prefix="remote")

        @server.resource(uri="local://item", name="item")
        def local_item() -> str:
            return "local"

        @server.prompt(name="local_prompt")
        def local_prompt() -> str:
            return "local prompt"

        resources = await server.list_resources()
        uris = [str(resource.uri) for resource in resources]
        assert "local://item" in uris
        mounted_uri = next(uri for uri in uris if uri.startswith("mount://"))
        assert "remote" in mounted_uri
        body = await server.read_resource(mounted_uri)
        assert body[0].content == "body:note://hello"
        assert remote.read_uris == ["note://hello"]

        prompts = await server.list_prompts()
        assert {prompt.name for prompt in prompts} >= {"local_prompt", "remote_greet"}
        rendered = await server.get_prompt("remote_greet", {"name": "Ada"})
        assert rendered.messages[0].content.text == "Hello Ada"
        assert remote.prompt_calls == [("greet", {"name": "Ada"})]

    @pytest.mark.asyncio
    async def test_resources_and_prompts_can_be_disabled(self):
        server = _server()
        remote = FakeConnector(
            "remote",
            tools=[_tool("echo")],
            resources=[Resource(name="note", uri=AnyUrl("note://hello"))],
            prompts=[Prompt(name="greet")],
        )
        server.mount(remote, prefix="remote", resources=False, prompts=False)

        assert _names(await server.list_tools()) == ["remote_echo"]
        assert [str(resource.uri) for resource in await server.list_resources()] == []
        assert [prompt.name for prompt in await server.list_prompts()] == []

    @pytest.mark.asyncio
    async def test_resource_uri_collision_is_rejected(self):
        server = _server()

        @server.resource(uri="note://hello", name="note")
        def note() -> str:
            return "local"

        server.mount(
            FakeConnector("remote", resources=[Resource(name="note", uri=AnyUrl("note://hello"))]),
        )
        with pytest.raises(MountError, match="resource URI"):
            await server.list_resources()
