"""Tests for bash / bash_output / bash_kill tools."""

from __future__ import annotations

import importlib
from pathlib import Path

from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    import vllama.agents.tools.bash as m

    importlib.reload(m)


async def test_bash_foreground_success(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler({"command": "echo hello"}, ctx)
    assert result.error is False
    assert "hello" in result.content


async def test_bash_foreground_nonzero_exit_not_error(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler({"command": "false"}, ctx)
    # Non-zero exit = model's problem, not a tool error.
    assert result.error is False
    assert "exit 1" in result.content.lower() or "exited 1" in result.content.lower()


async def test_bash_foreground_captures_stderr(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler({"command": "echo out; echo err 1>&2"}, ctx)
    assert "out" in result.content
    assert "err" in result.content


async def test_bash_foreground_timeout(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler(
        {"command": "sleep 5", "timeout_ms": 200}, ctx
    )
    assert result.error is True
    assert "timed out" in result.content.lower()


async def test_bash_missing_command_arg(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "command" in result.content.lower()


async def test_bash_background_returns_shell_id(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    result = await spec.handler(
        {"command": "echo hi", "run_in_background": True}, ctx
    )
    assert result.error is False
    assert "shell_id" in result.content.lower()


async def test_bash_output_reads_background(ctx: ToolContext) -> None:
    _load()
    spec_start = get_spec("bash")
    spec_out = get_spec("bash_output")
    assert spec_start is not None and spec_out is not None
    r = await spec_start.handler(
        {"command": "printf hi", "run_in_background": True}, ctx
    )
    assert ctx.bg_procs is not None
    shell_id = r.content.split("shell_id=")[1].split(",")[0].strip()
    await ctx.bg_procs.wait(shell_id, timeout=2.0)
    result = await spec_out.handler({"shell_id": shell_id}, ctx)
    assert result.error is False
    assert "hi" in result.content


async def test_bash_output_unknown_shell(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("bash_output")
    assert spec is not None
    result = await spec.handler({"shell_id": "does-not-exist"}, ctx)
    assert result.error is True
    assert "unknown" in result.content.lower() or "not found" in result.content.lower()


async def test_bash_kill_terminates(ctx: ToolContext) -> None:
    _load()
    spec_start = get_spec("bash")
    spec_kill = get_spec("bash_kill")
    assert spec_start is not None and spec_kill is not None
    r = await spec_start.handler(
        {"command": "sleep 30", "run_in_background": True}, ctx
    )
    shell_id = r.content.split("shell_id=")[1].split(",")[0].strip()
    result = await spec_kill.handler({"shell_id": shell_id}, ctx)
    assert result.error is False
    assert "killed" in result.content.lower() or "terminated" in result.content.lower()


async def test_bash_kill_already_dead(ctx: ToolContext) -> None:
    _load()
    spec_start = get_spec("bash")
    spec_kill = get_spec("bash_kill")
    assert spec_start is not None and spec_kill is not None
    r = await spec_start.handler(
        {"command": "true", "run_in_background": True}, ctx
    )
    shell_id = r.content.split("shell_id=")[1].split(",")[0].strip()
    assert ctx.bg_procs is not None
    await ctx.bg_procs.wait(shell_id, timeout=2.0)
    result = await spec_kill.handler({"shell_id": shell_id}, ctx)
    assert result.error is True
    assert "already" in result.content.lower() or "exited" in result.content.lower()


async def test_bash_no_bg_procs_in_ctx_fails(tmp_path: Path) -> None:
    _load()
    spec = get_spec("bash")
    assert spec is not None
    ctx_no_bg = ToolContext(
        cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768, bg_procs=None
    )
    result = await spec.handler(
        {"command": "echo hi", "run_in_background": True}, ctx_no_bg
    )
    assert result.error is True
    assert "background" in result.content.lower()
