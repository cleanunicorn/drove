"""bash / bash_output / bash_kill tools.

Foreground mode runs via `asyncio.create_subprocess_shell`, waits up to
`timeout_ms`, returns merged stdout+stderr plus exit code info. Background
mode delegates to `ctx.bg_procs`, returning a shell_id + pid.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_TIMEOUT_MS = 600_000

_BASH_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command. Foreground (default) blocks until exit or timeout"
            " and returns combined stdout+stderr. Set run_in_background=true to"
            " spawn a long-running process; returns a shell_id you can poll with"
            " bash_output or terminate with bash_kill."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        f"Foreground timeout in milliseconds"
                        f" (default {_DEFAULT_TIMEOUT_MS}, max {_MAX_TIMEOUT_MS})."
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "If true, spawn and return shell_id without waiting.",
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable summary of the command (for UI).",
                },
            },
            "required": ["command"],
        },
    },
}

_OUTPUT_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash_output",
        "description": (
            "Read accumulated stdout+stderr from a backgrounded shell started with"
            " bash(run_in_background=true). Returns any new bytes from `offset` plus"
            " the exit code if the shell has exited."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "shell_id": {"type": "string"},
                "offset": {
                    "type": "integer",
                    "description": "Byte offset into the buffer; default 0.",
                },
            },
            "required": ["shell_id"],
        },
    },
}

_KILL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash_kill",
        "description": (
            "Terminate a backgrounded shell. Sends SIGTERM, waits ~2s, then SIGKILL"
            " if needed. Returns an error if the shell doesn't exist or already exited."
        ),
        "parameters": {
            "type": "object",
            "properties": {"shell_id": {"type": "string"}},
            "required": ["shell_id"],
        },
    },
}


async def _bash_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = args.get("command")
    if not isinstance(command, str) or not command:
        return ToolResult(content="Error: 'command' argument is required", error=True)

    timeout_raw = args.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
    timeout_ms = (
        timeout_raw
        if isinstance(timeout_raw, int) and 0 < timeout_raw <= _MAX_TIMEOUT_MS
        else _DEFAULT_TIMEOUT_MS
    )

    run_bg = bool(args.get("run_in_background", False))

    if run_bg:
        if ctx.bg_procs is None:
            return ToolResult(
                content="Error: background execution not available (no bg_procs in context)",
                error=True,
            )
        shell_id = await ctx.bg_procs.start(command, cwd=ctx.cwd)
        bp = ctx.bg_procs.get(shell_id)
        pid = bp.pid if bp is not None else -1
        return ToolResult(
            content=f"Background shell started. shell_id={shell_id}, pid={pid}",
            meta={"shell_id": shell_id, "pid": pid},
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:
        return ToolResult(content=f"Error launching command: {e}", error=True)

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_ms / 1000)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        partial = b""
        if proc.stdout is not None:
            try:
                partial = await asyncio.wait_for(proc.stdout.read(), timeout=1.0)
            except (TimeoutError, Exception):
                partial = b""
        text = partial.decode("utf-8", errors="replace")
        return ToolResult(
            content=(
                f"Error: command timed out after {timeout_ms} ms and was killed."
                f" Partial output:\n{text}"
            ),
            error=True,
        )

    text = stdout.decode("utf-8", errors="replace")
    exit_code = proc.returncode if proc.returncode is not None else -1
    return ToolResult(content=f"{text}\n[exit {exit_code}]")


async def _bash_output_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    shell_id = args.get("shell_id")
    if not isinstance(shell_id, str) or not shell_id:
        return ToolResult(content="Error: 'shell_id' argument is required", error=True)
    if ctx.bg_procs is None:
        return ToolResult(
            content="Error: background procs not available in this context", error=True
        )
    bp = ctx.bg_procs.get(shell_id)
    if bp is None:
        return ToolResult(content=f"Error: unknown shell_id: {shell_id}", error=True)

    offset_raw = args.get("offset", 0)
    offset = offset_raw if isinstance(offset_raw, int) and offset_raw >= 0 else 0

    data, exit_code = await ctx.bg_procs.output(shell_id, offset=offset)
    text = data.decode("utf-8", errors="replace")
    suffix = f"\n[exit {exit_code}]" if exit_code is not None else "\n[still running]"
    return ToolResult(content=text + suffix)


async def _bash_kill_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    shell_id = args.get("shell_id")
    if not isinstance(shell_id, str) or not shell_id:
        return ToolResult(content="Error: 'shell_id' argument is required", error=True)
    if ctx.bg_procs is None:
        return ToolResult(
            content="Error: background procs not available in this context", error=True
        )
    bp = ctx.bg_procs.get(shell_id)
    if bp is None:
        return ToolResult(content=f"Error: unknown shell_id: {shell_id}", error=True)
    if bp.exit_code is not None:
        return ToolResult(
            content=f"Error: shell {shell_id} already exited (code {bp.exit_code})",
            error=True,
        )
    ok = await ctx.bg_procs.kill(shell_id)
    if not ok:
        return ToolResult(content=f"Error: failed to kill {shell_id}", error=True)
    return ToolResult(content=f"Killed shell {shell_id} (exit code {bp.exit_code})")


register(
    ToolSpec(
        name="bash",
        definition=_BASH_DEFINITION,
        tier="exec",
        handler=_bash_handler,
    )
)

register(
    ToolSpec(
        name="bash_output",
        definition=_OUTPUT_DEFINITION,
        tier="read",
        handler=_bash_output_handler,
    )
)

register(
    ToolSpec(
        name="bash_kill",
        definition=_KILL_DEFINITION,
        tier="mutate",
        handler=_bash_kill_handler,
    )
)
