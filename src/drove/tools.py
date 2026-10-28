"""Tool definitions and execution for TUI function calling."""

from __future__ import annotations

import codecs
import fnmatch
import io
import json
import math
import os
import selectors
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx

from drove.tool_policy import (
    TOOL_OUTPUT_LIMIT,
    TRUNCATION_NOTICE,
    canonical_origin,
    cap_output,
    is_blocked_address,
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file from disk."
                " For large files, use offset and limit to read a range of lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line to start from (0-based, default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read (default: 200).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file on disk. Creates parent directories automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default: current directory).",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list subdirectories recursively.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_execute",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run command in background (returns task ID).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (1-600, default: 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the content of an http(s) URL (HTTP GET, follows redirects).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name pattern or content string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory to search in (default: current directory).",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Glob pattern for filenames (e.g. '*.py').",
                    },
                    "content_query": {
                        "type": "string",
                        "description": "Substring to search for within file contents.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_background_task",
            "description": "Check the status and output of a background task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to check.",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
]


#: Receives decoded output while a foreground shell command runs (on its worker thread).
OutputCallback = Callable[[str], None]
#: Asks the user, from the tool's worker thread, whether fetch_url may reach an origin
#: (scheme://host:port); the flag is set when the host resolves to an internal address.
OriginApproval = Callable[[str, bool], bool]

_READ_CHUNK = 4096
_READ_DEFAULT_LINES = 200
_SEARCH_MAX_FILE_BYTES = 1024 * 1024  # content queries skip larger files
_SEARCH_BINARY_SAMPLE = 8192  # a NUL byte in this prefix marks a file as binary
_SHELL_DEFAULT_TIMEOUT = 30
_SHELL_MAX_TIMEOUT = 600
# What a shell result keeps of each stream; both shares and the frame fit the output cap.
_STDOUT_LIMIT = 40 * 1024
_STDERR_LIMIT = 20 * 1024
_MAX_BACKGROUND_RUNNING = 4
_MAX_BACKGROUND_FINISHED = 16
_BACKGROUND_RETENTION_SECONDS = 600.0
_FETCH_MAX_REDIRECTS = 5
_FETCH_TIMEOUT = 30.0

#: Replaced by tests with a fake network; None means real connections.
_fetch_transport: httpx.BaseTransport | None = None


class ToolFailure(str):
    """A tool result that reports a failure; its text is sent to the model unchanged."""


@dataclass
class ToolContext:
    """What the chat approved for one call. Direct callers may leave it out.

    *read_roots* limits the directories read tools open (None: no limit).
    *approved_origins* are the scheme://host:port origins fetch_url may reach (None:
    any public one), and *internal_origins* those that may also resolve to internal
    addresses. *approve_origin* asks the user about any other origin met on the way.
    """

    cwd: Path = field(default_factory=Path.cwd)
    read_roots: tuple[Path, ...] | None = None
    approved_origins: set[str] | None = None
    internal_origins: set[str] = field(default_factory=set)
    approve_origin: OriginApproval | None = None


class _Refused(Exception):
    """Raised inside a tool to return its message as a ToolFailure."""


# Background tasks by ID, guarded by _background_lock
_background_tasks: dict[str, dict[str, Any]] = {}
_background_lock = threading.Lock()


def execute_tool(
    name: str,
    arguments: str,
    on_output: OutputCallback | None = None,
    *,
    context: ToolContext | None = None,
) -> str:
    """Execute a tool call and return the result, capped to the tool output limit.

    Failures come back as ToolFailure. *on_output* receives shell output as it arrives.
    """
    return cap_output(_dispatch(name, arguments, on_output, context or ToolContext()))


def _dispatch(
    name: str, arguments: str, on_output: OutputCallback | None, context: ToolContext
) -> str:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return ToolFailure(f"Error: invalid JSON arguments: {arguments}")
    if not isinstance(args, dict):
        return ToolFailure(f"Error: arguments must be a JSON object: {arguments}")

    try:
        if name == "read_file":
            return _read_file(args, context)
        elif name == "write_file":
            return _write_file(args, context)
        elif name == "list_files":
            return _list_files(args, context)
        elif name == "shell_execute":
            return _shell_execute(args, on_output)
        elif name == "fetch_url":
            return _fetch_url(args, context)
        elif name == "search_files":
            return _search_files(args, context)
        elif name == "check_background_task":
            return _check_background_task(args)
        else:
            return ToolFailure(f"Error: unknown tool '{name}'")
    except _Refused as refusal:
        return ToolFailure(str(refusal))


def _mark_truncated(text: str) -> str:
    """*text* within the output cap, always ending with the truncation notice."""
    return cap_output(text + TRUNCATION_NOTICE)


def _path_arg(args: dict[str, Any], context: ToolContext, default: str) -> Path:
    value = args.get("path", default)
    if not isinstance(value, str):
        raise _Refused("Error: path must be a string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else context.cwd / path


def _count_arg(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _Refused(f"Error: {key} must be a non-negative integer")
    return int(value)


def _within_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError, RuntimeError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def _check_roots(path: Path, context: ToolContext) -> None:
    """Refuse a target that resolves outside the directories approved for this call."""
    if context.read_roots is not None and not _within_roots(path, context.read_roots):
        raise _Refused(
            f"Error: {path} resolves outside the approved directories;"
            " it was not read. Call the tool on that path directly to ask the user."
        )


def _read_file(args: dict[str, Any], context: ToolContext) -> str:
    path = _path_arg(args, context, "")
    offset = _count_arg(args, "offset", 0)
    limit = _count_arg(args, "limit", _READ_DEFAULT_LINES)
    _check_roots(path, context)  # first, so missing paths outside the roots are not revealed
    if not path.exists():
        return ToolFailure(f"Error: file not found: {path}")
    if not path.is_file():
        return ToolFailure(f"Error: not a file: {path}")

    lines: list[str] = []
    used = read = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for _ in range(offset):
                if not _skip_line(f):
                    break
            while read < limit:
                # Never read more than the cap, even from a file that is one huge line.
                line = f.readline(TOOL_OUTPUT_LIMIT - used + 1)
                if not line:
                    return "".join(lines)
                lines.append(line)
                used += len(line)
                if used > TOOL_OUTPUT_LIMIT:
                    return _mark_truncated("".join(lines))
                if line.endswith("\n"):
                    read += 1
            more = f.read(1) != ""
    except OSError as e:
        return ToolFailure(f"Error reading file: {e}")

    text = "".join(lines)
    if more:
        text += f"\n[more lines: continue with offset={offset + read}]"
    return text


def _skip_line(f: IO[str]) -> bool:
    """Skip one line in bounded chunks; False at end of file."""
    while True:
        chunk = f.readline(_READ_CHUNK)
        if not chunk:
            return False
        if chunk.endswith("\n"):
            return True


def _write_file(args: dict[str, Any], context: ToolContext) -> str:
    path = _path_arg(args, context, "")
    content = args.get("content", "")
    if not isinstance(content, str):
        return ToolFailure("Error: content must be a string")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolFailure(f"Error writing file: {e}")
    return f"Successfully wrote {len(content)} bytes to {path}"


def _list_files(args: dict[str, Any], context: ToolContext) -> str:
    path = _path_arg(args, context, ".")
    recursive = args.get("recursive", False)

    _check_roots(path, context)  # first, so missing paths outside the roots are not revealed
    if not path.exists():
        return ToolFailure(f"Error: path not found: {path}")
    if not path.is_dir():
        return ToolFailure(f"Error: not a directory: {path}")

    entries: list[str] = []
    used = 0
    try:
        # os.walk does not descend into symlinked directories.
        for entry in _walk_entries(path, bool(recursive)):
            entries.append(entry)
            used += len(entry) + 1
            if used > TOOL_OUTPUT_LIMIT:
                return _mark_truncated("\n".join(sorted(entries)))
        return "\n".join(sorted(entries))
    except Exception as e:
        return ToolFailure(f"Error listing files: {e}")


def _walk_entries(path: Path, recursive: bool) -> Iterator[str]:
    if not recursive:
        for entry in path.iterdir():
            yield entry.name + ("/" if entry.is_dir() else "")
        return
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        for d in dirs:
            yield str((root_path / d).relative_to(path)) + "/"
        for f in files:
            yield str((root_path / f).relative_to(path))


def _clamp_timeout(value: object) -> float:
    """A shell timeout within 1..600 seconds; anything not a finite number gives 30."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _SHELL_DEFAULT_TIMEOUT
    if isinstance(value, float) and not math.isfinite(value):
        return _SHELL_DEFAULT_TIMEOUT
    return min(max(value, 1), _SHELL_MAX_TIMEOUT)


def _shell_execute(args: dict[str, Any], on_output: OutputCallback | None = None) -> str:
    command = args.get("command", "")
    timeout = _clamp_timeout(args.get("timeout"))

    if not command or not isinstance(command, str):
        return ToolFailure("Error: no command provided")

    if args.get("background", False):
        return _start_background(command, timeout)

    try:
        stdout, stderr, returncode = _run_streaming(command, timeout, on_output)
    except subprocess.TimeoutExpired:
        return ToolFailure(f"Error: command timed out after {timeout:g} seconds")
    except Exception as e:
        return ToolFailure(f"Error executing command: {e}")
    output = f"STDOUT:\n{cap_output(stdout, _STDOUT_LIMIT)}"
    if stderr:
        output += f"\nSTDERR:\n{cap_output(stderr, _STDERR_LIMIT)}"
    output += f"\nReturn code: {returncode}"
    return output if returncode == 0 else ToolFailure(output)


def _start_background(command: str, timeout: float) -> str:
    with _background_lock:
        _prune_background()
        running = sum(1 for task in _background_tasks.values() if task["status"] == "running")
        if running >= _MAX_BACKGROUND_RUNNING:
            return ToolFailure(
                f"Error: {running} background tasks are already running;"
                " wait for one to finish (check_background_task) before starting another"
            )
        task_id = uuid.uuid4().hex[:8]
        _background_tasks[task_id] = {
            "command": command,
            "status": "running",
            "output": "",
            "error": "",
        }
    threading.Thread(target=_run_background, args=(task_id, command, timeout), daemon=True).start()
    return f"Started background task with ID: {task_id}"


def _run_background(task_id: str, command: str, timeout: float) -> None:
    update: dict[str, Any]
    try:
        stdout, stderr, returncode = _run_streaming(command, timeout, None)
        update = {
            "status": "finished",
            "output": cap_output(stdout, _STDOUT_LIMIT),
            "error": cap_output(stderr, _STDERR_LIMIT),
            "returncode": returncode,
        }
    except subprocess.TimeoutExpired:
        update = {
            "status": "timed out",
            "error": f"Error: command timed out after {timeout:g} seconds",
        }
    except Exception as e:
        update = {"status": "failed", "error": str(e)}
    update["finished_at"] = time.monotonic()
    with _background_lock:
        if task_id in _background_tasks:
            _background_tasks[task_id].update(update)


def _prune_background() -> None:
    """Forget finished tasks past their retention, and all but the newest few.

    The caller holds _background_lock; running tasks are never dropped.
    """
    now = time.monotonic()
    finished = sorted(
        (
            (task["finished_at"], task_id)
            for task_id, task in _background_tasks.items()
            if "finished_at" in task
        ),
        reverse=True,
    )
    for index, (finished_at, task_id) in enumerate(finished):
        expired = now - finished_at > _BACKGROUND_RETENTION_SECONDS
        if expired or index >= _MAX_BACKGROUND_FINISHED:
            del _background_tasks[task_id]


def _run_streaming(
    command: str, timeout: float, on_output: OutputCallback | None
) -> tuple[str, str, int]:
    """Run *command*, draining stdout and stderr together until it exits or *timeout* passes.

    Reads raw chunks rather than lines, so output without a newline still streams. The
    command runs in its own process group, which is killed when it ends or times out, so
    no descendant outlives it. Only the first part of each stream is kept.
    """
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdout is not None and proc.stderr is not None
    streams: dict[IO[bytes], list[str]] = {proc.stdout: [], proc.stderr: []}
    kept = dict.fromkeys(streams, 0)
    decoders = {
        stream: io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True
        )
        for stream in streams
    }
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ, stream)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _ in selector.select(remaining):
                    source: IO[bytes] = key.data
                    chunk = os.read(key.fd, _READ_CHUNK)
                    if not chunk:
                        selector.unregister(source)
                    text = decoders[source].decode(chunk, final=not chunk)
                    if text:
                        # Keep draining past the limit so the command never blocks on a pipe.
                        if kept[source] < TOOL_OUTPUT_LIMIT:
                            streams[source].append(text[: TOOL_OUTPUT_LIMIT - kept[source]])
                            kept[source] += len(text)
                        if on_output is not None:
                            on_output(text)
        returncode = proc.wait(timeout=max(deadline - time.monotonic(), 0))
    except BaseException:
        _kill_group(proc)
        proc.wait()
        raise
    finally:
        _kill_group(proc)  # descendants that let go of the pipes but are still running
        proc.stdout.close()
        proc.stderr.close()
    return "".join(streams[proc.stdout]), "".join(streams[proc.stderr]), returncode


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError, PermissionError:
        pass


def _resolve(host: str, port: int) -> list[str]:
    """Every address *host* resolves to, in resolver order (replaced by tests)."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _fetch_url(args: dict[str, Any], context: ToolContext) -> str:
    url = args.get("url", "")
    if not url or not isinstance(url, str):
        return ToolFailure("Error: no URL provided")
    try:
        current = httpx.URL(url)
    except httpx.InvalidURL as e:
        return ToolFailure(f"Error: invalid URL: {e}")

    approved = None if context.approved_origins is None else set(context.approved_origins)
    internal = set(context.internal_origins)
    try:
        # trust_env=False: an environment proxy would connect on our behalf, unchecked.
        with httpx.Client(
            timeout=_FETCH_TIMEOUT, trust_env=False, transport=_fetch_transport
        ) as client:
            for _ in range(_FETCH_MAX_REDIRECTS + 1):
                host, address = _vet_hop(current, approved, internal, context)
                response = client.send(_pinned_request(client, current, host, address), stream=True)
                try:
                    if response.is_redirect:
                        current = current.join(response.headers["location"])
                        continue
                    if response.is_error:
                        return ToolFailure(
                            f"Error fetching URL: HTTP {response.status_code} from {current}"
                        )
                    return _read_body(response)
                finally:
                    response.close()
    except _Refused:
        raise
    except Exception as e:
        return ToolFailure(f"Error fetching URL: {e}")
    return ToolFailure(f"Error fetching URL: too many redirects (more than {_FETCH_MAX_REDIRECTS})")


def _vet_hop(
    url: httpx.URL, approved: set[str] | None, internal: set[str], context: ToolContext
) -> tuple[str, str]:
    """Check one hop and return its host and the vetted address to connect to.

    Every DNS answer must be public unless the user allowed internal access to that
    origin, and an origin not yet approved (a redirect to another host, scheme or port)
    must be approved now.
    """
    if url.scheme not in {"http", "https"}:
        raise _Refused(f"Error: fetch_url supports http and https URLs only, not {str(url)!r}")
    if url.userinfo:
        raise _Refused("Error: fetch_url does not accept credentials in the URL")
    parsed = canonical_origin(str(url))
    if parsed is None:
        raise _Refused(f"Error: invalid URL: {url}")
    origin, host = parsed
    try:
        addresses = [host] if _is_ip(host) else _resolve(host, url.port or _default_port(url))
    except OSError as e:
        raise _Refused(f"Error fetching URL: cannot resolve {host}: {e}") from e
    if not addresses:
        raise _Refused(f"Error fetching URL: cannot resolve {host}")

    blocked = [address for address in addresses if is_blocked_address(address)]
    needs_origin = approved is not None and origin not in approved
    needs_internal = bool(blocked) and origin not in internal
    if needs_origin or needs_internal:
        if context.approve_origin is None or not context.approve_origin(origin, bool(blocked)):
            if blocked:
                raise _Refused(
                    f"Error: {host} resolves to an internal address ({blocked[0]});"
                    " fetch_url refuses private, loopback and link-local addresses"
                    " unless the user approves that origin"
                )
            raise _Refused(f"Error: fetching from {origin} (a redirect target) was not approved")
        if approved is not None:
            approved.add(origin)
        if blocked:
            internal.add(origin)
    return host, addresses[0]


def _is_ip(host: str) -> bool:
    try:
        is_blocked_address(host)
    except ValueError:
        return False
    return True


def _default_port(url: httpx.URL) -> int:
    return 443 if url.scheme == "https" else 80


def _pinned_request(client: httpx.Client, url: httpx.URL, host: str, address: str) -> httpx.Request:
    """A request to *address* that still presents *host*: in Host, SNI and certificate checks.

    Connecting to the vetted address (rather than resolving again) closes the DNS
    rebinding gap between the check and the connection.
    """
    return client.build_request(
        "GET",
        url.copy_with(host=address),
        headers={"Host": url.netloc.decode("ascii")},
        extensions={"sni_hostname": host},
    )


def _read_body(response: httpx.Response) -> str:
    data = bytearray()
    truncated = False
    for chunk in response.iter_bytes():
        data += chunk
        if len(data) > TOOL_OUTPUT_LIMIT:
            truncated = True
            break
    try:
        text = data.decode(response.charset_encoding or "utf-8", errors="replace")
    except LookupError:
        text = data.decode("utf-8", errors="replace")
    return _mark_truncated(text) if truncated else text


def _search_files(args: dict[str, Any], context: ToolContext) -> str:
    path = _path_arg(args, context, ".")
    name_pattern = args.get("name_pattern")
    content_query = args.get("content_query")
    recursive = args.get("recursive", True)
    for key, value in (("name_pattern", name_pattern), ("content_query", content_query)):
        if value is not None and not isinstance(value, str):
            return ToolFailure(f"Error: {key} must be a string")

    _check_roots(path, context)  # first, so missing paths outside the roots are not revealed
    if not path.exists():
        return ToolFailure(f"Error: path not found: {path}")

    results: list[str] = []
    used = skipped = outside = 0
    truncated = False
    try:
        for p in path.rglob("*") if recursive else path.iterdir():
            # The name is checked first, so non-matching files are never opened.
            if name_pattern and not fnmatch.fnmatch(p.name, name_pattern):
                continue
            if not p.is_file():
                continue
            if content_query:
                if context.read_roots is not None and not _within_roots(p, context.read_roots):
                    outside += 1
                    continue
                content = _searchable_text(p)
                if content is None:
                    skipped += 1
                    continue
                if content_query not in content:
                    continue
            entry = str(p.relative_to(path))
            results.append(entry)
            used += len(entry) + 1
            if used > TOOL_OUTPUT_LIMIT:
                truncated = True
                break
    except Exception as e:
        return ToolFailure(f"Error searching files: {e}")

    text = "\n".join(results) if results else "No matches found."
    if skipped:
        text += f"\n[skipped {skipped} binary, oversized or undecodable files]"
    if outside:
        text += (
            f"\n[skipped {outside} files outside the approved directories;"
            " search them directly to ask the user]"
        )
    return _mark_truncated(text) if truncated else text


def _searchable_text(path: Path) -> str | None:
    """The text of a small UTF-8 file, or None for a large, binary or undecodable one."""
    try:
        if path.stat().st_size > _SEARCH_MAX_FILE_BYTES:
            return None
        with path.open("rb") as f:
            data = f.read(_SEARCH_MAX_FILE_BYTES + 1)  # the file may have grown since
    except OSError:
        return None
    if len(data) > _SEARCH_MAX_FILE_BYTES or b"\0" in data[:_SEARCH_BINARY_SAMPLE]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _check_background_task(args: dict[str, Any]) -> str:
    task_id = args.get("task_id", "")
    with _background_lock:
        _prune_background()
        task = dict(_background_tasks[task_id]) if task_id in _background_tasks else None
    if task is None:
        return ToolFailure(f"Error: task ID '{task_id}' not found")

    status = task["status"]
    res = f"Task {task_id} status: {status}\nCommand: {task['command']}\n"
    if status != "running":
        res += f"Output:\n{task['output']}\n"
        if task["error"]:
            res += f"Error:\n{task['error']}\n"
        if "returncode" in task:
            res += f"Return code: {task['returncode']}\n"
    return res
