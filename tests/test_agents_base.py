"""Tests for the tool registry and base types."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    all_specs,
    clear_registry,
    get_spec,
    register,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    clear_registry()


def test_tool_result_defaults() -> None:
    r = ToolResult(content="hello")
    assert r.content == "hello"
    assert r.error is False
    assert r.truncated is False
    assert r.meta is None


async def _noop_handler(args: dict, ctx: ToolContext) -> ToolResult:
    return ToolResult(content="")


def test_register_and_get_spec() -> None:
    spec = ToolSpec(
        name="demo",
        definition={"type": "function", "function": {"name": "demo"}},
        tier="read",
        handler=_noop_handler,
    )
    register(spec)
    assert get_spec("demo") is spec
    assert spec in all_specs()


def test_get_spec_missing_returns_none() -> None:
    assert get_spec("missing") is None


def test_register_replaces_same_name() -> None:
    s1 = ToolSpec(name="demo", definition={}, tier="read", handler=_noop_handler)
    s2 = ToolSpec(name="demo", definition={}, tier="mutate", handler=_noop_handler)
    register(s1)
    register(s2)
    assert get_spec("demo") is s2
    assert len(all_specs()) == 1


def test_tool_context_shape(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    assert ctx.cwd == tmp_path
    assert ctx.cap_bytes == 8192
    assert ctx.cap_bytes_bash == 32768
