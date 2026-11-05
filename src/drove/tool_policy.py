"""Approval policy and output limits for the chat tools, independent of the TUI.

The chat asks before a tool reads outside its working directory, before any read
when the model runs on a remote endpoint, before each file write or unapproved shell command,
and before fetching from an origin (scheme, host and port) it has not been allowed
to reach. "Allow for session" and "Allow always" record a :class:`Grant` scoped to
one directory, one origin, or one exact command in its working directory, never a
bare tool name.
"""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import httpx

#: Every tool result is cut to this many UTF-8 bytes, notice included.
TOOL_OUTPUT_LIMIT = 64 * 1024
TRUNCATION_NOTICE = "\n… [output truncated]"

READ_TOOLS = frozenset({"read_file", "list_files", "search_files"})
#: Tools that ask per call by default; only the conversation editor can grant them.
PER_CALL_TOOLS = frozenset({"shell_execute", "write_file"})

_GRANT_VERSION = 1


def normalize_command(command: str) -> str:
    """Trim only shell whitespace, identically for execution and command grants."""
    return command.strip(" \t\n")


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
    """True for loopback, private, link-local, unspecified and other non-public IPs.

    An IPv6 address that carries an IPv4 one (mapped, IPv4-compatible, well-known NAT64,
    6to4, Teredo) is judged by the IPv4 address, which a translator or tunnel would
    reach. Local-use NAT64 addresses, whose layout cannot be known, are always blocked.
    """
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _NAT64_LOCAL_USE:
            return True
        if embedded := _embedded_ipv4(ip):
            return any(_is_internal(v4) for v4 in embedded)
    return _is_internal(ip)


# The well-known NAT64 prefix (RFC 6052) is always a /96 with the IPv4 address in the
# last 32 bits. The local-use prefix (RFC 8215) may be split at /48 to /96, each placing
# the IPv4 bits differently, so no single decoding is safe: it is always internal.
_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_LOCAL_USE = ipaddress.IPv6Network("64:ff9b:1::/48")
_IPV4_COMPATIBLE = ipaddress.IPv6Network("::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    if ip.ipv4_mapped is not None:
        return [ip.ipv4_mapped]
    if ip.sixtofour is not None:
        return [ip.sixtofour]
    if ip.teredo is not None:
        return list(ip.teredo)  # the relay server and the client
    if ip in _NAT64_WELL_KNOWN or (ip in _IPV4_COMPATIBLE and int(ip) > 1):
        return [ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)]  # :: and ::1 are not IPv4
    return []


def _is_internal(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
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


def canonical_origin(url: str) -> tuple[str, str] | None:
    """The ``scheme://host:port`` origin of an http(s) *url* and its host, or None.

    The port is always explicit (80 or 443 by default), so ``https://example.com``
    and ``https://example.com:443`` are one origin while another port is not.
    """
    host = canonical_host(url)
    if host is None:
        return None
    parsed = httpx.URL(url)
    port = url_port(parsed)
    shown = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{shown}:{port}", host


def url_port(url: httpx.URL) -> int:
    """The port *url* connects to: its explicit port, else 443 for https and 80 otherwise.

    Shared by the approval scope (canonical_origin) and fetch_url's resolver, so the
    approved origin and the connection target always agree.
    """
    return url.port or (443 if url.scheme == "https" else 80)


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
    """Permission on a directory, exact origin, or editor-only exact command and cwd."""

    tool: str
    directory: str | None = None
    origin: str | None = None  # fetch only: scheme://host:port, see canonical_origin
    internal: bool = False  # fetch only: the host may resolve to an internal address
    command: str | None = None  # editor only: exact trimmed shell command, bound to directory

    def token(self) -> str:
        """The canonical string stored in ``allowed_tools``."""
        data: dict[str, Any] = {"v": _GRANT_VERSION, "tool": self.tool}
        if self.command is not None:
            data["command"] = self.command
        if self.directory is not None:
            data["dir"] = self.directory
        if self.origin is not None:
            data["origin"] = self.origin
        if self.internal:
            data["internal"] = True
        return json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=self.command is not None
        )

    def describe(self) -> str:
        if self.command is not None:
            return (
                f"{self.tool} exactly {json.dumps(self.command, ensure_ascii=True)}"
                f" in {json.dumps(self.directory, ensure_ascii=True)}"
            )
        if self.directory is not None:
            return f"{self.tool} in {self.directory}"
        suffix = ", including internal addresses" if self.internal else ""
        return f"{self.tool} from {self.origin}{suffix}"


def parse_grant(token: str) -> Grant | None:
    """Parse an ``allowed_tools`` entry; bare tool names and unknown shapes give None.

    That includes host-only fetch grants (``"host"``, no ``"origin"``): they would cover
    every scheme and port, so they are dropped and the user is asked again.
    """
    try:
        data = json.loads(token)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("v") != _GRANT_VERSION:
        return None
    tool, directory, origin = data.get("tool"), data.get("dir"), data.get("origin")
    if not isinstance(tool, str) or "host" in data:
        return None
    # shell_execute is in PER_CALL_TOOLS but can have saved grants; write_file cannot.
    if tool in PER_CALL_TOOLS and tool != "shell_execute":
        return None
    if tool == "shell_execute":
        command = data.get("command")
        if set(data) != {"v", "tool", "dir", "command"} or type(data["v"]) is not int:
            return None
        if not valid_command(command) or command != normalize_command(command):
            return None
        if not isinstance(directory, str) or not directory or "\x00" in directory:
            return None
        path = Path(directory)
        if not path.is_absolute() or str(path) != directory or ".." in path.parts:
            return None
        return Grant(tool, directory=directory, command=command)
    if tool in READ_TOOLS and isinstance(directory, str) and origin is None:
        return Grant(tool, directory=directory)
    if tool == "fetch_url" and isinstance(origin, str) and directory is None:
        # Only a canonical origin matches anything; re-derive it to reject odd spellings.
        if (parsed := canonical_origin(origin)) is None or parsed[0] != origin:
            return None
        return Grant(tool, origin=origin, internal=data.get("internal") is True)
    return None


def parse_grants(tokens: Iterable[str]) -> frozenset[Grant]:
    """Parse every valid grant in *tokens*; entries parse_grant rejects are dropped."""
    return frozenset(g for token in tokens if (g := parse_grant(token)) is not None)


@dataclass(frozen=True)
class PolicyContext:
    """What a decision depends on: the chat's working directory, endpoint and grants."""

    cwd: Path  # resolved
    remote_endpoint: bool
    grants: frozenset[Grant] = frozenset()
    allow_all: bool = False
    managed: bool = False  # conversation editor may authorize shell/write rules


@dataclass(frozen=True)
class Decision:
    """Whether a call must be confirmed, and what a session/always choice would grant."""

    prompt: bool
    grant: Grant | None = None  # the scope this call is approved for
    scoped_choices: bool = False  # offer "Allow for session" and "Allow always"
    reason: str = ""


_NO_PROMPT = Decision(prompt=False)


def join_tool_path(value: str, cwd: Path) -> Path:
    """A tool's *value* path against *cwd*, with ``~`` expanded but symlinks not followed.

    The policy (resolve_path) and the tools (which show the path as the model wrote
    it) both build on this, so they always start from the same path.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else cwd / path


def resolve_path(value: str, cwd: Path) -> Path:
    """Resolve a tool's *value* path against *cwd*, expanding ``~`` and following symlinks."""
    return join_tool_path(value, cwd).resolve()


def decide(name: str, arguments: str, ctx: PolicyContext) -> Decision:
    """Decide whether the tool call *name* with JSON *arguments* needs the user's approval.

    Calls whose arguments the tool itself rejects need no prompt: they fail before
    doing anything.
    """
    if ctx.allow_all:
        return _NO_PROMPT
    if name == "shell_execute":
        return _decide_shell(arguments, ctx)
    if name in PER_CALL_TOOLS:
        if ctx.managed:
            try:
                args = json.loads(arguments)
                if not isinstance(args, dict):
                    return Decision(prompt=True)
                for grant in ctx.grants:
                    if grant.tool != name or grant.directory is None:
                        continue
                    if isinstance(args.get("path"), str):
                        if resolve_path(args["path"], ctx.cwd).is_relative_to(grant.directory):
                            return Decision(prompt=False, grant=grant)
            except ValueError, OSError, RuntimeError:
                pass
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


def valid_command(command: Any) -> TypeGuard[str]:
    """Whether a command can be executed and represented by a persistent grant."""
    if not isinstance(command, str) or not normalize_command(command) or "\x00" in command:
        return False
    try:
        command.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _decide_shell(arguments: str, ctx: PolicyContext) -> Decision:
    try:
        args = json.loads(arguments)
    except ValueError:
        return Decision(prompt=True)
    command = args.get("command") if isinstance(args, dict) else None
    if not valid_command(command):
        return Decision(prompt=True)
    normalized = normalize_command(command)
    cwd_str = str(ctx.cwd.resolve())
    # Check existing grants for this exact command and directory
    for grant in ctx.grants:
        if (
            grant.tool == "shell_execute"
            and grant.command == normalized
            and grant.directory == cwd_str
        ):
            return Decision(prompt=False, grant=grant)
    grant = Grant("shell_execute", directory=cwd_str, command=normalized)
    reason = (
        "The chat endpoint is not local: the command's output is sent to a remote server."
        if ctx.remote_endpoint
        else ""
    )
    return Decision(prompt=True, grant=grant, scoped_choices=True, reason=reason)


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
    parsed = canonical_origin(url) if isinstance(url, str) else None
    if parsed is None:
        return _NO_PROMPT
    origin, host = parsed
    try:
        internal = is_blocked_address(host)
    except ValueError:
        internal = False  # a name: checked against its DNS answers when fetched
    for granted in ctx.grants:
        if granted.tool == "fetch_url" and granted.origin == origin:
            if granted.internal or not internal:
                return _NO_PROMPT
    return origin_decision(origin, internal)


def origin_decision(origin: str, internal: bool) -> Decision:
    """The prompt for fetching from *origin*, also asked about one met mid-fetch."""
    reason = f"Fetch from {origin}."
    if internal:
        reason = (
            f"{origin} is or resolves to an internal (private, loopback or link-local) address."
        )
    grant = Grant("fetch_url", origin=origin, internal=internal)
    return Decision(prompt=True, grant=grant, scoped_choices=True, reason=reason)


def fetch_origins(ctx: PolicyContext, approved: Grant | None) -> tuple[set[str], set[str]]:
    """Origins fetch_url may reach, and those it may reach at internal addresses."""
    grants = [g for g in ctx.grants if g.tool == "fetch_url" and g.origin is not None]
    if approved is not None and approved.tool == "fetch_url" and approved.origin is not None:
        grants.append(approved)
    origins = {g.origin for g in grants if g.origin is not None}
    return origins, {g.origin for g in grants if g.origin is not None and g.internal}


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
