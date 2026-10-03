"""Unit tests for paginated tools, resources, and prompts listing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import Prompt, Resource, Tool

from mcp_use.client.connectors.base import BaseConnector, RepeatedCursorError


class _ListingConnector(BaseConnector):
    """Minimal connector so list helpers can be exercised without a transport."""

    async def connect(self) -> None:
        self._connected = True

    @property
    def public_identifier(self) -> str:
        return "listing-test"


def _tool(name: str) -> Tool:
    return Tool(name=name, description=name, inputSchema={"type": "object"})


def _resource(name: str) -> Resource:
    return Resource(name=name, uri=f"file:///{name}")


def _prompt(name: str) -> Prompt:
    return Prompt(name=name, description=name)


class _PagingSession:
    """Fake client session whose list methods return scripted pages."""

    def __init__(self, pages: dict[str, list[tuple[list, str | None]]]) -> None:
        self._pages = pages
        self.calls: dict[str, list[str | None]] = {name: [] for name in pages}
        self.init_result = SimpleNamespace(capabilities=SimpleNamespace(tools=True, resources=True, prompts=True))

    def _page(self, method: str, attr: str, cursor: str | None):
        self.calls[method].append(cursor)
        items, nxt = self._pages[method][len(self.calls[method]) - 1]
        page = SimpleNamespace(nextCursor=nxt)
        setattr(page, attr, items)
        return page

    async def initialize(self):
        return self.init_result

    async def list_tools(self, cursor: str | None = None):
        return self._page("tools", "tools", cursor)

    async def list_resources(self, cursor: str | None = None):
        return self._page("resources", "resources", cursor)

    async def list_prompts(self, cursor: str | None = None):
        return self._page("prompts", "prompts", cursor)


def _ready(session: _PagingSession) -> _ListingConnector:
    connector = _ListingConnector()
    connector.client_session = session  # type: ignore[assignment]
    connector._connected = True
    return connector


def _names(items: list) -> list[str]:
    return [item.name for item in items]


THREE_PAGES = {
    "tools": [([_tool("t0"), _tool("t1")], "c1"), ([_tool("t2")], ""), ([_tool("t3"), _tool("t4")], None)],
    "resources": [([_resource("r0")], "c1"), ([_resource("r1")], ""), ([_resource("r2")], None)],
    "prompts": [([_prompt("p0")], "c1"), ([_prompt("p1")], ""), ([_prompt("p2")], None)],
}

NO_PAGES = {
    "tools": [([_tool("t0"), _tool("t1")], None)],
    "resources": [([_resource("r0")], None)],
    "prompts": [([_prompt("p0"), _prompt("p1"), _prompt("p2")], None)],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "kind"),
    [("list_tools", "tools"), ("list_resources", "resources"), ("list_prompts", "prompts")],
)
async def test_list_follows_next_cursor_across_three_pages(method: str, kind: str) -> None:
    """Three pages are concatenated, and an empty-string cursor is still followed."""
    session = _PagingSession(THREE_PAGES)
    connector = _ready(session)

    items = await getattr(connector, method)()

    expected = [item for page, _cursor in THREE_PAGES[kind] for item in page]
    assert _names(items) == _names(expected)
    assert session.calls[kind] == [None, "c1", ""]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "kind"),
    [("list_tools", "tools"), ("list_resources", "resources"), ("list_prompts", "prompts")],
)
async def test_list_without_pagination_calls_once(method: str, kind: str) -> None:
    """A missing nextCursor returns the single page and does not request another."""
    session = _PagingSession(NO_PAGES)
    connector = _ready(session)

    items = await getattr(connector, method)()

    expected = NO_PAGES[kind][0][0]
    assert _names(items) == _names(expected)
    assert session.calls[kind] == [None]


@pytest.mark.asyncio
async def test_repeated_cursor_raises_logs_and_stops() -> None:
    """A cursor that comes back twice is logged, raised, and not followed again."""
    session = _PagingSession({"tools": [([_tool("t0")], "again"), ([_tool("t1")], "again")]})
    connector = _ready(session)

    with patch("mcp_use.client.connectors.base.logger") as mocked_logger:
        with pytest.raises(RepeatedCursorError) as exc_info:
            await connector.list_tools()

    assert session.calls["tools"] == [None, "again"]
    assert _names(exc_info.value.items) == ["t0", "t1"]
    assert connector._tools is not None
    assert _names(connector._tools) == ["t0", "t1"]
    mocked_logger.error.assert_called_once()
    assert "again" in mocked_logger.error.call_args.args[1]


@pytest.mark.asyncio
async def test_initialize_collects_every_page() -> None:
    """Session startup caches tools, resources, and prompts from every page."""
    session = _PagingSession(THREE_PAGES)
    connector = _ready(session)

    result = await connector.initialize()

    assert result is session.init_result
    assert _names(connector._tools or []) == ["t0", "t1", "t2", "t3", "t4"]
    assert _names(connector._resources or []) == ["r0", "r1", "r2"]
    assert _names(connector._prompts or []) == ["p0", "p1", "p2"]
    assert session.calls["tools"] == [None, "c1", ""]
