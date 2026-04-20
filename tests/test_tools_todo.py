"""Tests for todo_write tool."""

from __future__ import annotations

import importlib

from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    import vllama.agents.tools.todo as m

    importlib.reload(m)


async def test_todo_write_replaces_list(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    result = await spec.handler(
        {
            "todos": [
                {"id": "1", "content": "Do X", "status": "pending"},
                {"id": "2", "content": "Then Y", "status": "in_progress"},
            ]
        },
        ctx,
    )
    assert result.error is False
    assert len(ctx.todos) == 2
    assert ctx.todos[0] == {"id": "1", "content": "Do X", "status": "pending"}


async def test_todo_write_renders_checklist(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    result = await spec.handler(
        {
            "todos": [
                {"id": "a", "content": "step", "status": "completed"},
            ]
        },
        ctx,
    )
    assert "[x]" in result.content
    assert "step" in result.content


async def test_todo_write_empty_clears(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    ctx.todos.append({"id": "x", "content": "old", "status": "pending"})
    result = await spec.handler({"todos": []}, ctx)
    assert result.error is False
    assert ctx.todos == []
    assert "cleared" in result.content.lower()


async def test_todo_write_rejects_non_list(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    result = await spec.handler({"todos": "nope"}, ctx)
    assert result.error is True
    assert "list" in result.content.lower()


async def test_todo_write_rejects_bad_status(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    result = await spec.handler(
        {"todos": [{"id": "1", "content": "x", "status": "weird"}]},
        ctx,
    )
    assert result.error is True
    assert "status" in result.content.lower()


async def test_todo_write_rejects_missing_id(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    result = await spec.handler(
        {"todos": [{"content": "x", "status": "pending"}]},
        ctx,
    )
    assert result.error is True
    assert "id" in result.content.lower()


async def test_todo_write_mutation_visible_to_caller(ctx: ToolContext) -> None:
    """Mutation happens in-place via slice assign — same list object shared."""
    _load()
    spec = get_spec("todo_write")
    assert spec is not None
    original = ctx.todos
    await spec.handler(
        {"todos": [{"id": "1", "content": "x", "status": "pending"}]},
        ctx,
    )
    assert ctx.todos is original
    assert len(original) == 1
