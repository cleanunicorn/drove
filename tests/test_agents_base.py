"""Tests for the tool registry and base types."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    all_specs,
    get_spec,
    register,
)


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


def test_tool_context_requires_kwargs(tmp_path: Path) -> None:
    """ToolContext construction must be kw-only (protects against positional drift)."""
    with pytest.raises(TypeError):
        ToolContext(tmp_path, 8192, 32768)  # type: ignore[misc]


def test_tool_context_carries_bg_procs(tmp_path: Path) -> None:
    from vllama.agents.bash_procs import BgProcs

    procs = BgProcs()
    ctx = ToolContext(
        cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768, bg_procs=procs
    )
    assert ctx.bg_procs is procs


def test_package_import_registers_all_tools() -> None:
    # Re-populate by explicitly reloading each child module.
    from vllama.agents.tools import (  # noqa: F401
        bash,
        edit,
        glob,
        grep,
        list,
        read,
        todo,
        webfetch,
        write,
    )

    for mod in (bash, edit, glob, grep, list, read, todo, webfetch, write):
        importlib.reload(mod)

    names = {s.name for s in all_specs()}
    assert names == {
        "read_file",
        "write_file",
        "apply_patch",
        "list_dir",
        "glob_files",
        "grep",
        "bash",
        "bash_output",
        "bash_kill",
        "todo_write",
        "web_fetch",
    }


def test_all_definitions_match_openai_shape() -> None:
    # Re-populate by explicitly reloading each child module.
    from vllama.agents.tools import (  # noqa: F401
        bash,
        edit,
        glob,
        grep,
        list,
        read,
        todo,
        webfetch,
        write,
    )

    for mod in (bash, edit, glob, grep, list, read, todo, webfetch, write):
        importlib.reload(mod)

    for spec in all_specs():
        assert spec.definition.get("type") == "function"
        fn = spec.definition.get("function")
        assert isinstance(fn, dict)
        assert fn.get("name") == spec.name
        assert "description" in fn
        assert "parameters" in fn


def test_tool_context_todos_default_empty(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    assert ctx.todos == []


def test_tool_context_todos_is_mutable(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    ctx.todos.append({"id": "1", "content": "x", "status": "pending"})
    assert len(ctx.todos) == 1
    assert ctx.todos[0]["id"] == "1"


def test_tool_context_todos_isolated_per_instance(tmp_path: Path) -> None:
    a = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    b = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    a.todos.append({"id": "1", "content": "x", "status": "pending"})
    assert b.todos == []
