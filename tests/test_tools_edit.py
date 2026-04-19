"""Tests for the apply_patch tool."""

from __future__ import annotations

import importlib
from pathlib import Path

import vllama.agents.tools.edit as edit_module
from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    importlib.reload(edit_module)


def _make_diff(old: str, new: str, path: str) -> str:
    """Build a unified diff from two full-file strings (without trailing-newline concerns)."""
    import difflib

    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


async def test_apply_patch_success(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    diff = _make_diff("x = 1\ny = 2\nz = 3\n", "x = 1\ny = 20\nz = 3\n", "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is False, result.content
    assert p.read_text(encoding="utf-8") == "x = 1\ny = 20\nz = 3\n"


async def test_apply_patch_context_mismatch(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("actual\ncontent\n", encoding="utf-8")
    diff = _make_diff("different\ncontent\n", "different\nnew\n", "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is True
    assert "hunk" in result.content.lower() or "mismatch" in result.content.lower()
    # File left untouched.
    assert p.read_text(encoding="utf-8") == "actual\ncontent\n"


async def test_apply_patch_missing_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("apply_patch")
    assert spec is not None
    diff = _make_diff("a\n", "b\n", "missing.py")
    result = await spec.handler({"path": str(tmp_path / "missing.py"), "diff": diff}, ctx)
    assert result.error is True
    assert "not found" in result.content.lower()


async def test_apply_patch_missing_args(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("apply_patch")
    assert spec is not None
    r1 = await spec.handler({"diff": "x"}, ctx)
    r2 = await spec.handler({"path": "/tmp/x"}, ctx)
    assert r1.error is True
    assert r2.error is True


async def test_apply_patch_malformed_diff(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("x\n", encoding="utf-8")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": "not a diff"}, ctx)
    assert result.error is True


async def test_apply_patch_multi_hunk(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    original = "a\nb\nc\nd\ne\nf\ng\nh\n"
    modified = "a\nB\nc\nd\ne\nf\nG\nh\n"
    p.write_text(original, encoding="utf-8")
    diff = _make_diff(original, modified, "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is False, result.content
    assert p.read_text(encoding="utf-8") == modified
