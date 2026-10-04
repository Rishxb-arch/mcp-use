"""Compose a local MCP server with a remote one.

The upstream server in this file stands in for any streamable-HTTP MCP server.
The composer exposes that server's ``echo`` tool as ``remote_echo`` and adds a
local tool that calls it.

    python examples/server/mount_example.py

Then open http://127.0.0.1:<composer port>/inspector (debug mode) or point an
MCP client at http://127.0.0.1:<composer port>/mcp. The process prints both ports.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request

from mcp.types import CallToolResult

from mcp_use import MCPServer
from mcp_use.client.connectors import HttpConnector


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_upstream(port: int) -> MCPServer:
    server = MCPServer(name="upstream", host="127.0.0.1", port=port, debug=False)

    @server.tool()
    def echo(text: str) -> str:
        """Repeat text from the upstream server."""
        return f"upstream:{text}"

    return server


def build_composer(upstream_port: int, composer_port: int) -> MCPServer:
    server = MCPServer(name="composer", host="127.0.0.1", port=composer_port, debug=True)
    server.mount(
        HttpConnector(f"http://127.0.0.1:{upstream_port}/mcp"),
        prefix="remote",
    )

    @server.tool()
    async def shout(text: str) -> str:
        """Call the mounted echo tool and uppercase the result."""
        result = await server.call_tool("remote_echo", {"text": text})
        if not isinstance(result, CallToolResult):
            raise TypeError(f"Expected CallToolResult from a mounted tool, got {type(result).__name__}")
        return result.content[0].text.upper()

    return server


def _wait_until_up(port: int) -> None:
    deadline = time.time() + 15
    url = f"http://127.0.0.1:{port}/mcp"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return
        except urllib.error.HTTPError:
            # The MCP endpoint is up; GET is just not its method.
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"upstream server did not accept connections on {url}")


def main() -> None:
    upstream_port = _free_port()
    composer_port = _free_port()
    upstream = build_upstream(upstream_port)
    thread = threading.Thread(
        target=lambda: upstream.run(transport="streamable-http", host="127.0.0.1", port=upstream_port),
        name="upstream-mcp",
        daemon=True,
    )
    thread.start()
    _wait_until_up(upstream_port)

    print(f"Upstream MCP: http://127.0.0.1:{upstream_port}/mcp")
    print(f"Composer MCP: http://127.0.0.1:{composer_port}/mcp")
    print(f"Inspector:    http://127.0.0.1:{composer_port}/inspector")
    composer = build_composer(upstream_port, composer_port)
    composer.run(transport="streamable-http", host="127.0.0.1", port=composer_port)


if __name__ == "__main__":
    main()
