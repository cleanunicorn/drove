"""Tests for the glob_files tool."""

from __future__ import annotations

import importlib
import os
import time
from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.glob as glob_module  # noqa: F401

    importlib.reload(glob_module)


async def test_glob_flat(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    assert result.error is False
    assert "a.py" in result.content
    assert "b.txt" not in result.content


async def test_glob_recursive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "top.py").write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "**/*.py"}, ctx)
    assert "sub/a.py" in result.content.replace("\\", "/")
    assert "top.py" in result.content


async def test_glob_sorts_by_mtime_desc(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    old = tmp_path / "old.py"
    old.write_text("x", encoding="utf-8")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    new = tmp_path / "new.py"
    new.write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    lines = [ln for ln in result.content.split("\n") if ln.strip()]
    assert lines[0].endswith("new.py")
    assert lines[1].endswith("old.py")


async def test_glob_cap(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    for i in range(1100):
        (tmp_path / f"f{i:04d}.py").write_text("x", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    lines = [ln for ln in result.content.split("\n") if ln.strip() and ln.endswith(".py")]
    assert len(lines) == 1000
    assert "truncated" in result.content.lower()


async def test_glob_missing_pattern(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "pattern" in result.content.lower()


async def test_glob_no_matches(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.nothere"}, ctx)
    assert result.error is False
    assert result.content == ""
