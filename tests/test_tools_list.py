"""Tests for the list_dir tool."""

from __future__ import annotations

from importlib import reload
from pathlib import Path

import vllama.agents.tools.list as list_module
from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    # Import and reload to trigger registration after clear_registry.
    reload(list_module)


async def test_list_flat(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path)}, ctx)
    assert result.error is False
    lines = result.content.strip().split("\n")
    assert "a.txt" in lines
    assert "sub/" in lines


async def test_list_recursive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path), "recursive": True}, ctx)
    assert "a.txt" in result.content
    assert "sub/" in result.content
    assert "sub/b.txt" in result.content


async def test_list_max_entries(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path), "max_entries": 5}, ctx)
    lines = [ln for ln in result.content.split("\n") if ln.strip()]
    # includes truncation marker line as a non-path line
    file_lines = [ln for ln in lines if ln.endswith(".txt")]
    assert len(file_lines) == 5
    assert "truncated" in result.content.lower()


async def test_list_path_not_found(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "nope")}, ctx)
    assert result.error is True


async def test_list_not_a_dir(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(f)}, ctx)
    assert result.error is True


async def test_list_default_cwd(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert "a.txt" in result.content
