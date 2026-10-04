"""Mount a real local streamable-HTTP server and call it through a composer.

Nothing here talks to an external service. Both servers bind 127.0.0.1 on
ephemeral ports.
"""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from mcp.types import CallToolResult

from mcp_use.client.connectors import HttpConnector
from mcp_use.server import MCPServer
from mcp_use.server.auth import AccessToken, BearerAuthProvider, get_access_token


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def _serve(server: MCPServer) -> AsyncIterator[str]:
    """Serve ``server`` over streamable HTTP and yield its MCP URL."""
    port = _free_port()
    server._wrap_handlers_with_middleware()
    app = server.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    running = uvicorn.Server(config)
    task = asyncio.create_task(running.serve())
    try:
        for _ in range(100):
            if running.started:
                break
            if task.done():
                error = task.exception()
                raise RuntimeError(f"server exited during startup: {error}")
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f"server did not start on 127.0.0.1:{port}")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        running.should_exit = True
        await asyncio.wait_for(task, timeout=5)


def _upstream() -> MCPServer:
    server = MCPServer(name="upstream", host="127.0.0.1", debug=False)

    @server.tool()
    def echo(text: str) -> str:
        """Repeat text from the upstream server."""
        return f"upstream:{text}"

    @server.resource(uri="note://hello", name="hello", description="Upstream note", mime_type="text/plain")
    def hello() -> str:
        return "hello-from-upstream"

    @server.prompt(name="greet", description="Greet someone")
    def greet(name: str) -> str:
        return f"Hello {name}"

    return server


def _composer(upstream_url: str) -> MCPServer:
    server = MCPServer(name="composer", host="127.0.0.1", debug=False)
    server.mount(HttpConnector(upstream_url, timeout=10, sse_read_timeout=10), prefix="remote")

    @server.tool()
    def local_only(text: str) -> str:
        """A tool that stays on the composer."""
        return f"local:{text}"

    @server.tool()
    async def shout(text: str) -> str:
        """Call the mounted echo tool and uppercase its text."""
        result = await server.call_tool("remote_echo", {"text": text})
        assert isinstance(result, CallToolResult)
        return result.content[0].text.upper()

    return server


class TestStreamableHttpMount:
    @pytest.mark.asyncio
    async def test_client_sees_prefixed_tools_resources_and_prompts(self):
        async with _serve(_upstream()) as upstream_url:
            composer = _composer(upstream_url)
            async with _serve(composer) as composer_url:
                client = HttpConnector(composer_url, timeout=10, sse_read_timeout=10)
                await client.connect()
                try:
                    tools = await client.list_tools()
                    names = {tool.name for tool in tools}
                    assert {"local_only", "shout", "remote_echo"} <= names
                    remote_echo = next(tool for tool in tools if tool.name == "remote_echo")
                    assert remote_echo.meta["mcp-use/mount"]["upstream"] == "echo"
                    assert remote_echo.meta["mcp-use/mount"]["prefix"] == "remote"

                    echoed = await client.call_tool("remote_echo", {"text": "hi"})
                    assert echoed.content[0].text == "upstream:hi"

                    shouted = await client.call_tool("shout", {"text": "hi"})
                    assert shouted.content[0].text == "UPSTREAM:HI"

                    local = await client.call_tool("local_only", {"text": "ok"})
                    assert local.content[0].text == "local:ok"

                    resources = await client.list_resources()
                    mounted = next(resource for resource in resources if str(resource.uri).startswith("mount://"))
                    read = await client.read_resource(mounted.uri)
                    assert read.contents[0].text == "hello-from-upstream"

                    prompts = await client.list_prompts()
                    assert "remote_greet" in {prompt.name for prompt in prompts}
                    prompt = await client.get_prompt("remote_greet", {"name": "Ada"})
                    assert "Ada" in prompt.messages[0].content.text
                finally:
                    await client.disconnect()


class _TokenAuth(BearerAuthProvider):
    def __init__(self, expected: str, seen: list[str]):
        self._expected = expected
        self._seen = seen

    async def verify_token(self, token: str) -> AccessToken | None:
        self._seen.append(token)
        if token != self._expected:
            return None
        return AccessToken(token=token, claims={"token": token})


class TestAuthIsNotForwarded:
    @pytest.mark.asyncio
    async def test_caller_token_is_not_sent_to_the_mounted_server(self):
        """The composer authenticates its caller. The mount uses its own token.

        A mistaken auth passthrough would present ``caller-token`` to the
        upstream, which only accepts ``upstream-token``.
        """
        upstream_seen: list[str] = []
        upstream = MCPServer(
            name="upstream",
            host="127.0.0.1",
            debug=False,
            auth=_TokenAuth("upstream-token", upstream_seen),
        )

        @upstream.tool()
        def whoami() -> str:
            token = get_access_token()
            return "" if token is None else token.token

        composer_seen: list[str] = []
        composer = MCPServer(
            name="composer",
            host="127.0.0.1",
            debug=False,
            auth=_TokenAuth("caller-token", composer_seen),
        )

        async with _serve(upstream) as upstream_url:
            composer.mount(
                HttpConnector(upstream_url, auth="upstream-token", timeout=10, sse_read_timeout=10),
                prefix="remote",
            )
            async with _serve(composer) as composer_url:
                client = HttpConnector(composer_url, auth="caller-token", timeout=10, sse_read_timeout=10)
                await client.connect()
                try:
                    result = await client.call_tool("remote_whoami", {})
                finally:
                    await client.disconnect()

        assert result.content[0].text == "upstream-token"
        assert "caller-token" not in upstream_seen
        assert "upstream-token" in upstream_seen
        assert "caller-token" in composer_seen
