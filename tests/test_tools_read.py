"""Tests for the read_file tool."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    # Import and reload to trigger registration after clear_registry.
    import vllama.agents.tools.read as read_module

    reload(read_module)


async def test_read_file_full(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p)}, ctx)
    assert result.error is False
    assert result.content == "hello\nworld\n"


async def test_read_file_offset_limit(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "offset": 1, "limit": 2}, ctx)
    assert result.content == "b\nc\n"


async def test_read_file_offset_only(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "offset": 1}, ctx)
    assert result.content == "b\nc\n"


async def test_read_file_not_found(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "nope.txt")}, ctx)
    assert result.error is True
    assert "not found" in result.content.lower()


async def test_read_file_not_a_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path)}, ctx)
    assert result.error is True
    assert "not a file" in result.content.lower()


async def test_read_file_binary_rejected(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "bin"
    p.write_bytes(b"\x00\x01\x02\x03" * 16)
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p)}, ctx)
    assert result.error is True
    assert "binary" in result.content.lower()


async def test_read_file_relative_to_cwd(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "rel.txt").write_text("content", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": "rel.txt"}, ctx)
    assert result.error is False
    assert result.content == "content"


async def test_read_file_missing_path_arg(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "path" in result.content.lower()
