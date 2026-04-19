"""Tests for the write_file tool."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    # Import and reload to trigger registration after clear_registry.
    import vllama.agents.tools.write as write_module

    reload(write_module)


async def test_write_new_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.txt"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "hello"}, ctx)
    assert result.error is False
    assert p.read_text(encoding="utf-8") == "hello"


async def test_write_overwrite(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.txt"
    p.write_text("old", encoding="utf-8")
    spec = get_spec("write_file")
    assert spec is not None
    await spec.handler({"path": str(p), "content": "new"}, ctx)
    assert p.read_text(encoding="utf-8") == "new"


async def test_write_creates_parents(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "nested" / "deep" / "a.txt"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "hi"}, ctx)
    assert result.error is False
    assert p.read_text(encoding="utf-8") == "hi"


async def test_write_relative(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    await spec.handler({"path": "a.txt", "content": "rel"}, ctx)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "rel"


async def test_write_missing_path(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"content": "x"}, ctx)
    assert result.error is True
    assert "path" in result.content.lower()


async def test_write_missing_content(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "a")}, ctx)
    assert result.error is True
    assert "content" in result.content.lower()


async def test_write_reports_bytes(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "12345"}, ctx)
    assert "5" in result.content
    assert str(p) in result.content
