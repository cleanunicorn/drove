"""Tests for ToolRuntime dispatch + output cap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    register,
)


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content=str(args.get("text", "")))


def _reg(name: str, tier: str = "read") -> None:
    register(
        ToolSpec(
            name=name,
            definition={
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            },
            tier=tier,  # type: ignore[arg-type]
            handler=_echo_handler,
        )
    )


async def test_dispatch_unknown_tool(ctx: ToolContext) -> None:
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("nope", "{}")
    assert r.error is True
    assert "unknown" in r.content.lower()


async def test_dispatch_invalid_json(ctx: ToolContext) -> None:
    _reg("echo")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("echo", "{not json")
    assert r.error is True
    assert "json" in r.content.lower()


async def test_dispatch_success(ctx: ToolContext) -> None:
    _reg("echo")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert r.error is False
    assert r.content == "hi"


async def test_dispatch_caps_output(tmp_path: Path) -> None:
    _reg("echo")
    ctx = ToolContext(cwd=tmp_path, cap_bytes=32, cap_bytes_bash=128)
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    payload = "A" * 200
    r = await rt.dispatch("echo", '{"text": "' + payload + '"}')
    assert r.truncated is True
    assert r.content.startswith("A" * 32)
    assert "truncated" in r.content.lower()
    assert "200" in r.content  # total bytes reported


async def test_dispatch_caps_bash_separately(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8, cap_bytes_bash=64)
    _reg("bash", tier="exec")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("bash", '{"text": "' + "x" * 50 + '"}')
    # Bash cap is 64, payload 50 — below cap, no truncate.
    assert r.truncated is False


async def test_dispatch_handler_exception_becomes_error_result(ctx: ToolContext) -> None:
    async def boom(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")

    register(
        ToolSpec(
            name="boom",
            definition={"type": "function", "function": {"name": "boom"}},
            tier="read",
            handler=boom,
        )
    )
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("boom", "{}")
    assert r.error is True
    assert "kaboom" in r.content
