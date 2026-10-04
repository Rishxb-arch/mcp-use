"""Compose remote MCP connectors into an :class:`~mcp_use.server.server.MCPServer`.

``MCPServer.mount()`` exposes another server's tools, resources, and prompts
through this process. Mounted tools are named ``{prefix}_{upstream}`` when a
prefix is set, matching :meth:`~mcp_use.server.server.MCPServer.include_router`.

Design choices that are easy to revisit:

* Prefixes are validated and rejected, not rewritten. A hyphen or space must
  be changed by the caller so two prefixes cannot collapse into one name.
* A name or resource URI that is already exposed (local or another mount) is
  an error. The catalog is not served with one side silently hidden.
* The caller's ``Authorization`` header is not copied onto the connector.
  Mounted servers keep the credentials the connector was built with. A shared
  connector session cannot safely carry a different caller token per request.
* Resource URIs are rewritten to ``mount://mounted/{prefix}/{quoted}`` so two
  upstreams can both expose ``file:///readme``. The original URI is restored
  on read. Prompts use the same name prefix as tools.
* Upstream pagination cursors are followed and flattened into this server's
  single list. ``BaseConnector.list_*`` returns only the first page.
* Resource templates, sampling, elicitation, and list-changed notifications
  are not proxied.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    Resource,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl

from mcp_use.client.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Leave room for ``{prefix}_{tool}`` under adapters that cap names at 64 chars.
_MAX_PREFIX_LENGTH = 48
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_]+$")
_MAX_LIST_PAGES = 100
_MOUNT_URI_SCHEME = "mount"
_MOUNT_URI_HOST = "mounted"
_META_KEY = "mcp-use/mount"


class MountError(ValueError):
    """The mount configuration is invalid or two exposed names collide."""


@dataclass
class _Mount:
    connector: BaseConnector
    prefix: str | None
    resources: bool
    prompts: bool
    connected: bool = False

    @property
    def source(self) -> str:
        if self.prefix is None:
            return self.connector.public_identifier
        return f"prefix {self.prefix!r} ({self.connector.public_identifier})"


@dataclass(frozen=True)
class _ToolRoute:
    connector: BaseConnector
    upstream_name: str
    source: str


@dataclass(frozen=True)
class _PromptRoute:
    connector: BaseConnector
    upstream_name: str
    source: str


def validate_mount_prefix(prefix: str | None) -> str | None:
    """Return a usable prefix, or ``None`` when the caller omitted one.

    Empty strings are rejected. Characters outside ``[A-Za-z0-9_]`` are
    rejected rather than replaced, so ``browser-tools`` cannot silently
    become the same prefix as ``browser_tools``.
    """
    if prefix is None:
        return None
    if not isinstance(prefix, str):
        raise MountError(f"Mount prefix must be a string or None, got {type(prefix).__name__}")
    if prefix == "":
        raise MountError("Mount prefix must be non-empty. Omit prefix to expose upstream names unchanged.")
    if len(prefix) > _MAX_PREFIX_LENGTH:
        raise MountError(
            f"Mount prefix {prefix!r} is {len(prefix)} characters; the maximum is {_MAX_PREFIX_LENGTH} "
            "so prefixed tool names can still fit common 64-character adapter limits."
        )
    if _PREFIX_RE.fullmatch(prefix) is None:
        raise MountError(
            f"Invalid mount prefix {prefix!r}. Use only letters, digits, and underscores "
            f"({_PREFIX_RE.pattern}). Prefixes are not rewritten."
        )
    return prefix


def exposed_name(prefix: str | None, upstream: str) -> str:
    """Public name for an upstream tool or prompt."""
    if prefix is None:
        return upstream
    return f"{prefix}_{upstream}"


def exposed_resource_uri(prefix: str | None, uri: str) -> str:
    """Public URI for an upstream resource.

    Prefixed resources use a synthetic URI so two mounts can expose the same
    upstream URI. Unprefixed resources keep the upstream URI.
    """
    if prefix is None:
        return uri
    return f"{_MOUNT_URI_SCHEME}://{_MOUNT_URI_HOST}/{prefix}/{quote(uri, safe='')}"


def parse_exposed_resource_uri(uri: str) -> tuple[str, str] | None:
    """Split a synthetic resource URI into ``(prefix, upstream_uri)``."""
    parsed = urlsplit(uri)
    if parsed.scheme != _MOUNT_URI_SCHEME or parsed.netloc != _MOUNT_URI_HOST:
        return None
    path = parsed.path.lstrip("/")
    prefix, separator, quoted = path.partition("/")
    if not separator or not prefix or not quoted:
        return None
    return prefix, unquote(quoted)


def wrap_lifespan(registry: MountRegistry, original: Callable[..., Any]) -> Callable[..., Any]:
    """Connect mounted connectors for the Starlette app lifetime."""

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        async with original(app):
            try:
                await registry.connect_all()
                yield
            finally:
                await registry.disconnect_all()

    return lifespan


class MountRegistry:
    """Mounted connectors for one server, and the routing tables over them."""

    def __init__(self) -> None:
        self._entries: list[_Mount] = []
        self._lock = asyncio.Lock()

    def __bool__(self) -> bool:
        return bool(self._entries)

    def add(
        self,
        connector: BaseConnector,
        prefix: str | None,
        *,
        resources: bool,
        prompts: bool,
    ) -> None:
        if not isinstance(connector, BaseConnector):
            raise TypeError(f"mount() expects a BaseConnector, got {type(connector).__name__}")
        normalized = validate_mount_prefix(prefix)
        for existing in self._entries:
            if existing.connector is connector:
                raise MountError(f"Connector {connector.public_identifier!r} is already mounted on this server.")
            if normalized is not None and existing.prefix == normalized:
                raise MountError(
                    f"Prefix {normalized!r} is already used by {existing.connector.public_identifier}. "
                    "Each prefixed mount needs its own prefix."
                )
        self._entries.append(_Mount(connector=connector, prefix=normalized, resources=resources, prompts=prompts))
        logger.debug(
            "Mounted %s with prefix=%s resources=%s prompts=%s",
            connector.public_identifier,
            normalized,
            resources,
            prompts,
        )

    async def connect_all(self) -> None:
        """Connect and initialize every mount that is not connected yet."""
        async with self._lock:
            for entry in self._entries:
                if entry.connected:
                    continue
                await entry.connector.connect()
                try:
                    if not getattr(entry.connector, "_initialized", False):
                        await entry.connector.initialize()
                except Exception:
                    try:
                        await entry.connector.disconnect()
                    except Exception:
                        logger.debug("Failed to disconnect %s after initialize error", entry.source, exc_info=True)
                    raise
                entry.connected = True

    async def disconnect_all(self) -> None:
        """Disconnect mounts this registry connected. Failures are logged."""
        async with self._lock:
            for entry in reversed(self._entries):
                if not entry.connected:
                    continue
                try:
                    await entry.connector.disconnect()
                except Exception:
                    logger.warning("Failed to disconnect mounted connector %s", entry.source, exc_info=True)
                finally:
                    entry.connected = False

    async def _snapshot(self) -> list[_Mount]:
        async with self._lock:
            return list(self._entries)

    async def tool_routes(self, local_names: set[str]) -> dict[str, _ToolRoute]:
        routes: dict[str, _ToolRoute] = {}
        owners: dict[str, str] = {}
        for entry in await self._snapshot():
            for tool in await _list_tools(entry.connector):
                name = exposed_name(entry.prefix, tool.name)
                route = _ToolRoute(connector=entry.connector, upstream_name=tool.name, source=entry.source)
                _claim(routes, owners, name, route, kind="tool")
        _reject_local_overlap(local_names, owners, kind="tool")
        return routes

    async def exposed_tools(self, local_names: set[str]) -> list[Tool]:
        """Prefixed tool definitions, after collision checks."""
        exposed: list[Tool] = []
        owners: dict[str, str] = {}
        seen: dict[str, Tool] = {}
        for entry in await self._snapshot():
            for tool in await _list_tools(entry.connector):
                name = exposed_name(entry.prefix, tool.name)
                copied = tool.model_copy(update={"name": name, "meta": _mount_meta(tool.meta, entry, tool.name)})
                _claim(seen, owners, name, copied, kind="tool")
                exposed.append(copied)
        _reject_local_overlap(local_names, owners, kind="tool")
        return exposed

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        local_names: set[str],
    ) -> CallToolResult | None:
        routes = await self.tool_routes(local_names)
        route = routes.get(name)
        if route is None:
            return None
        return await route.connector.call_tool(route.upstream_name, arguments)

    async def exposed_resources(self, local_uris: set[str]) -> list[Resource]:
        exposed: list[Resource] = []
        owners: dict[str, str] = {}
        seen: dict[str, Resource] = {}
        for entry in await self._snapshot():
            if not entry.resources:
                continue
            for resource in await _list_resources(entry.connector):
                uri = exposed_resource_uri(entry.prefix, str(resource.uri))
                name = exposed_name(entry.prefix, resource.name) if resource.name else resource.name
                copied = resource.model_copy(
                    update={
                        "uri": AnyUrl(uri),
                        "name": name,
                        "meta": _mount_meta(resource.meta, entry, str(resource.uri)),
                    }
                )
                _claim(seen, owners, uri, copied, kind="resource URI")
                exposed.append(copied)
        _reject_local_overlap(local_uris, owners, kind="resource URI")
        return exposed

    async def read_resource(self, uri: str) -> list[ReadResourceContents] | None:
        parsed = parse_exposed_resource_uri(uri)
        if parsed is not None:
            prefix, upstream = parsed
            for entry in await self._snapshot():
                if entry.prefix == prefix and entry.resources:
                    result = await entry.connector.read_resource(AnyUrl(upstream))
                    return _read_contents(result)
            return None

        for entry in await self._snapshot():
            if entry.prefix is not None or not entry.resources:
                continue
            for resource in await _list_resources(entry.connector):
                if str(resource.uri) == uri:
                    result = await entry.connector.read_resource(AnyUrl(uri))
                    return _read_contents(result)
        return None

    async def exposed_prompts(self, local_names: set[str]) -> list[Prompt]:
        exposed: list[Prompt] = []
        owners: dict[str, str] = {}
        seen: dict[str, Prompt] = {}
        for entry in await self._snapshot():
            if not entry.prompts:
                continue
            for prompt in await _list_prompts(entry.connector):
                name = exposed_name(entry.prefix, prompt.name)
                copied = prompt.model_copy(update={"name": name, "meta": _mount_meta(prompt.meta, entry, prompt.name)})
                _claim(seen, owners, name, copied, kind="prompt")
                exposed.append(copied)
        _reject_local_overlap(local_names, owners, kind="prompt")
        return exposed

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        local_names: set[str],
    ) -> GetPromptResult | None:
        routes: dict[str, _PromptRoute] = {}
        owners: dict[str, str] = {}
        for entry in await self._snapshot():
            if not entry.prompts:
                continue
            for prompt in await _list_prompts(entry.connector):
                exposed = exposed_name(entry.prefix, prompt.name)
                route = _PromptRoute(connector=entry.connector, upstream_name=prompt.name, source=entry.source)
                _claim(routes, owners, exposed, route, kind="prompt")
        _reject_local_overlap(local_names, owners, kind="prompt")
        route = routes.get(name)
        if route is None:
            return None
        return await route.connector.get_prompt(route.upstream_name, arguments)


def _source_of(value: Any) -> str:
    source = getattr(value, "source", None)
    if source:
        return str(source)
    meta = getattr(value, "meta", None) or {}
    mount_meta = meta.get(_META_KEY) if isinstance(meta, dict) else None
    if isinstance(mount_meta, dict) and mount_meta.get("source"):
        return str(mount_meta["source"])
    return "another mount"


def _claim(routes: dict[str, Any], owners: dict[str, str], key: str, value: Any, *, kind: str) -> None:
    source = _source_of(value)
    previous = owners.get(key)
    if previous is not None:
        raise MountError(
            f"{kind.capitalize()} {key!r} is exposed by {previous} and {source}. "
            "Use distinct prefixes so mounted names do not collide."
        )
    owners[key] = source
    routes[key] = value


def _reject_local_overlap(local_keys: set[str], owners: dict[str, str], *, kind: str) -> None:
    overlap = sorted(local_keys & set(owners))
    if not overlap:
        return
    details = ", ".join(f"{key} ({owners[key]})" for key in overlap)
    raise MountError(
        f"Mounted {kind}(s) collide with definitions on this server: {details}. "
        "Choose a prefix that does not overlap, or rename the local definition."
    )


def _mount_meta(existing: dict[str, Any] | None, entry: _Mount, upstream: str) -> dict[str, Any]:
    meta = dict(existing or {})
    meta[_META_KEY] = {"prefix": entry.prefix, "upstream": upstream, "source": entry.source}
    return meta


def _supports(connector: BaseConnector, capability: str) -> bool:
    capabilities = connector.capabilities
    if capabilities is None:
        return True
    return bool(getattr(capabilities, capability))


async def _list_pages(
    connector: BaseConnector,
    *,
    capability: str,
    list_name: str,
    page_attr: str,
    fallback: Callable[[], Awaitable[list[Any]]],
) -> list[Any]:
    if not _supports(connector, capability):
        return []
    session = connector.client_session
    if session is None:
        try:
            return list(await fallback())
        except Exception as exc:
            raise MountError(f"Failed to list {capability} from {connector.public_identifier}: {exc}") from exc

    fetch = getattr(session, list_name)
    items: list[Any] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(_MAX_LIST_PAGES):
        try:
            result = await fetch(cursor=cursor)
        except Exception as exc:
            raise MountError(f"Failed to list {capability} from {connector.public_identifier}: {exc}") from exc
        items.extend(getattr(result, page_attr))
        next_cursor = getattr(result, "nextCursor", None)
        if not next_cursor or next_cursor in seen:
            return items
        seen.add(next_cursor)
        cursor = next_cursor
    raise MountError(
        f"{connector.public_identifier} returned more than {_MAX_LIST_PAGES} {capability} pages. "
        "Refusing to flatten an unbounded catalog."
    )


async def _list_tools(connector: BaseConnector) -> list[Tool]:
    tools = await _list_pages(
        connector,
        capability="tools",
        list_name="list_tools",
        page_attr="tools",
        fallback=connector.list_tools,
    )
    return _dedupe(tools, lambda tool: tool.name)


async def _list_resources(connector: BaseConnector) -> list[Resource]:
    resources = await _list_pages(
        connector,
        capability="resources",
        list_name="list_resources",
        page_attr="resources",
        fallback=connector.list_resources,
    )
    return _dedupe(resources, lambda resource: str(resource.uri))


async def _list_prompts(connector: BaseConnector) -> list[Prompt]:
    prompts = await _list_pages(
        connector,
        capability="prompts",
        list_name="list_prompts",
        page_attr="prompts",
        fallback=connector.list_prompts,
    )
    return _dedupe(prompts, lambda prompt: prompt.name)


def _dedupe(items: list[Any], key: Callable[[Any], str]) -> list[Any]:
    """Drop repeated pages from one connector. Names across connectors still collide."""
    seen: set[str] = set()
    unique: list[Any] = []
    for item in items:
        marker = key(item)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _read_contents(result: ReadResourceResult) -> list[ReadResourceContents]:
    contents: list[ReadResourceContents] = []
    for item in result.contents:
        if isinstance(item, TextResourceContents):
            contents.append(ReadResourceContents(content=item.text, mime_type=item.mimeType, meta=item.meta))
        elif isinstance(item, BlobResourceContents):
            try:
                blob = base64.b64decode(item.blob)
            except Exception as exc:
                raise MountError(f"Mounted resource returned invalid base64 contents: {exc}") from exc
            contents.append(ReadResourceContents(content=blob, mime_type=item.mimeType, meta=item.meta))
        else:
            raise MountError(f"Unsupported mounted resource content type: {type(item).__name__}")
    return contents
