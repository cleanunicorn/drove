"""Approval policy and output limits for the chat tools, independent of the TUI.

The chat asks before a tool reads outside its working directory, before any read
when the model runs on a remote endpoint, before each shell command or file write,
and before fetching from a host it has not been allowed to reach. "Allow for
session" and "Allow always" record a :class:`Grant` scoped to one directory or
host, never a bare tool name.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

#: Every tool result is cut to this many UTF-8 bytes, notice included.
TOOL_OUTPUT_LIMIT = 64 * 1024
TRUNCATION_NOTICE = "\n… [output truncated]"

READ_TOOLS = frozenset({"read_file", "list_files", "search_files"})
#: Tools that always ask, once per call: no session or saved grant applies to them.
PER_CALL_TOOLS = frozenset({"shell_execute", "write_file"})

_GRANT_VERSION = 1


def cap_output(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """Cut *text* to at most *limit* UTF-8 bytes, ending with a truncation notice.

    Keeps the type of *text* (a :class:`~drove.tools.ToolFailure` stays one), never
    splits a character, and leaves already-capped text unchanged.
    """
    data = text.encode("utf-8", errors="surrogatepass")
    if len(data) <= limit:
        return text
    keep = max(limit - len(TRUNCATION_NOTICE.encode("utf-8")), 0)
    head = data[:keep].decode("utf-8", errors="ignore")
    return type(text)(head + TRUNCATION_NOTICE)


def cap_tool_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy *messages* with every tool result capped; other messages are kept as is."""
    capped = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            message = {**message, "content": cap_output(content)}
        capped.append(message)
    return capped


def is_blocked_address(address: str) -> bool:
    """True for loopback, private, link-local, unspecified and other non-public IPs."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


def canonical_host(url: str) -> str | None:
    """The lowercase, IDNA-encoded host of an http(s) *url*, or None when it has none."""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL, TypeError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.raw_host:
        return None
    # raw_host is the IDNA (ASCII) form, so a Unicode and a punycode spelling match.
    host = parsed.raw_host.decode("ascii", errors="replace").lower().rstrip(".")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host or None


def is_local_endpoint(base_url: str) -> bool:
    """True when the chat endpoint is this machine: ``localhost`` or a loopback IP."""
    host = canonical_host(base_url)
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Grant:
    """Permission for one tool on one directory (and below) or one exact host."""

    tool: str
    directory: str | None = None
    host: str | None = None
    internal: bool = False  # fetch only: the host may resolve to an internal address

    def token(self) -> str:
        """The canonical string stored in ``allowed_tools``."""
        data: dict[str, Any] = {"v": _GRANT_VERSION, "tool": self.tool}
        if self.directory is not None:
            data["dir"] = self.directory
        if self.host is not None:
            data["host"] = self.host
        if self.internal:
            data["internal"] = True
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def describe(self) -> str:
        if self.directory is not None:
            return f"{self.tool} in {self.directory}"
        suffix = ", including internal addresses" if self.internal else ""
        return f"{self.tool} from {self.host}{suffix}"


def parse_grant(token: str) -> Grant | None:
    """Parse an ``allowed_tools`` entry; bare tool names and unknown shapes give None."""
    try:
        data = json.loads(token)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("v") != _GRANT_VERSION:
        return None
    tool, directory, host = data.get("tool"), data.get("dir"), data.get("host")
    if not isinstance(tool, str) or tool in PER_CALL_TOOLS:
        return None
    if tool in READ_TOOLS and isinstance(directory, str) and host is None:
        return Grant(tool, directory=directory)
    if tool == "fetch_url" and isinstance(host, str) and directory is None:
        return Grant(tool, host=host, internal=data.get("internal") is True)
    return None


def parse_grants(tokens: Iterable[str]) -> frozenset[Grant]:
    return frozenset(g for token in tokens if (g := parse_grant(token)) is not None)


@dataclass(frozen=True)
class PolicyContext:
    """What a decision depends on: the chat's working directory, endpoint and grants."""

    cwd: Path  # resolved
    remote_endpoint: bool
    grants: frozenset[Grant] = frozenset()


@dataclass(frozen=True)
class Decision:
    """Whether a call must be confirmed, and what a session/always choice would grant."""

    prompt: bool
    grant: Grant | None = None  # the scope this call is approved for
    scoped_choices: bool = False  # offer "Allow for session" and "Allow always"
    reason: str = ""


_NO_PROMPT = Decision(prompt=False)


def resolve_path(value: str, cwd: Path) -> Path:
    """Resolve a tool's *value* path against *cwd*, expanding ``~`` and following symlinks."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


def decide(name: str, arguments: str, ctx: PolicyContext) -> Decision:
    """Decide whether the tool call *name* with JSON *arguments* needs the user's approval.

    Calls whose arguments the tool itself rejects need no prompt: they fail before
    doing anything.
    """
    if name in PER_CALL_TOOLS:
        return Decision(prompt=True)
    if name not in READ_TOOLS and name != "fetch_url":
        return _NO_PROMPT
    try:
        args = json.loads(arguments)
    except ValueError:
        return _NO_PROMPT
    if not isinstance(args, dict):
        return _NO_PROMPT
    if name == "fetch_url":
        return _decide_fetch(args, ctx)
    return _decide_read(name, args, ctx)


def _decide_read(name: str, args: dict[str, Any], ctx: PolicyContext) -> Decision:
    value = args.get("path", "" if name == "read_file" else ".")
    if not isinstance(value, str):
        return _NO_PROMPT
    try:
        target = resolve_path(value, ctx.cwd)
        scope = target.parent if name == "read_file" and not target.is_dir() else target
    except OSError, RuntimeError:
        return Decision(prompt=True, reason=f"Could not resolve {value!r}.")
    grant = Grant(name, directory=str(scope))
    if ctx.remote_endpoint:
        reason = (
            f"The chat endpoint is not local: {name} sends what it reads in {target}"
            " to a remote server."
        )
        return Decision(prompt=True, grant=grant, reason=reason)
    if target.is_relative_to(ctx.cwd):
        return _NO_PROMPT
    for granted in ctx.grants:
        if granted.tool == name and granted.directory is not None:
            if target.is_relative_to(granted.directory):
                return _NO_PROMPT
    reason = f"{target} is outside the working directory {ctx.cwd}."
    return Decision(prompt=True, grant=grant, scoped_choices=True, reason=reason)


def _decide_fetch(args: dict[str, Any], ctx: PolicyContext) -> Decision:
    url = args.get("url")
    host = canonical_host(url) if isinstance(url, str) else None
    if host is None:
        return _NO_PROMPT
    try:
        internal = is_blocked_address(host)
    except ValueError:
        internal = False  # a name: checked against its DNS answers when fetched
    for granted in ctx.grants:
        if granted.tool == "fetch_url" and granted.host == host:
            if granted.internal or not internal:
                return _NO_PROMPT
    return host_decision(host, internal)


def host_decision(host: str, internal: bool) -> Decision:
    """The prompt for fetching from *host*, also asked about a host met mid-fetch."""
    reason = f"Fetch from {host}."
    if internal:
        reason = f"{host} is or resolves to an internal (private, loopback or link-local) address."
    grant = Grant("fetch_url", host=host, internal=internal)
    return Decision(prompt=True, grant=grant, scoped_choices=True, reason=reason)


def fetch_hosts(ctx: PolicyContext, approved: Grant | None) -> tuple[set[str], set[str]]:
    """Hosts fetch_url may reach, and those of them it may reach at internal addresses."""
    grants = [g for g in ctx.grants if g.tool == "fetch_url" and g.host is not None]
    if approved is not None and approved.tool == "fetch_url" and approved.host is not None:
        grants.append(approved)
    hosts = {g.host for g in grants if g.host is not None}
    return hosts, {g.host for g in grants if g.host is not None and g.internal}


def read_roots(name: str, ctx: PolicyContext, approved: Grant | None) -> tuple[Path, ...]:
    """Directories a read tool may open at execution: cwd and grants (local endpoints
    only) plus the scope approved for this call."""
    roots: list[Path] = []
    if not ctx.remote_endpoint:
        roots.append(ctx.cwd)
        roots.extend(
            Path(g.directory)
            for g in sorted(ctx.grants, key=lambda g: g.directory or "")
            if g.tool == name and g.directory is not None
        )
    if approved is not None and approved.tool == name and approved.directory is not None:
        roots.append(Path(approved.directory))
    return tuple(dict.fromkeys(roots))
