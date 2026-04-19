"""Integration test: config → Policy → ToolRuntime → prompt_hook → dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllama.agents.permissions import AbortTurn, Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    register,
)


async def _echo(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content=str(args.get("text", "")))


def _register_dummy(name: str, tier: str = "mutate") -> None:
    register(
        ToolSpec(
            name=name,
            definition={
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            },
            tier=tier,  # type: ignore[arg-type]
            handler=_echo,
        )
    )


async def test_end_to_end_prompt_then_session_allow(tmp_path: Path) -> None:
    _register_dummy("write_file")
    policy = Policy.from_config({"write_file": "prompt"})
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    calls: list[str] = []

    async def hook(name: str, args: dict[str, object]) -> str:
        calls.append(name)
        return "session_allow"

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=hook)
    r1 = await rt.dispatch("write_file", '{"text": "a"}')
    r2 = await rt.dispatch("write_file", '{"text": "b"}')
    r3 = await rt.dispatch("write_file", '{"text": "c"}')
    assert r1.content == "a"
    assert r2.content == "b"
    assert r3.content == "c"
    assert calls == ["write_file"]  # hooked only once


async def test_end_to_end_abort_turn_propagates(tmp_path: Path) -> None:
    _register_dummy("bash", tier="exec")
    policy = Policy()  # exec tier default = PROMPT
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    async def hook(name: str, args: dict[str, object]) -> str:
        return "deny_abort"

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=hook)
    with pytest.raises(AbortTurn):
        await rt.dispatch("bash", '{"command": "rm -rf /"}')


async def test_end_to_end_deny_config_blocks_without_hook(tmp_path: Path) -> None:
    _register_dummy("bash", tier="exec")
    policy = Policy.from_config({"bash": "deny"})
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=None)
    r = await rt.dispatch("bash", '{"command": "date"}')
    assert r.error is True
    assert "denied" in r.content.lower()
