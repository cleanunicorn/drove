"""Tests for the grep tool."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, get_spec


@pytest.fixture(autouse=True)
def _force_python_impl(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default test config: pretend rg is not installed so we hit the Python path."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    yield


def _load() -> None:
    import vllama.agents.tools.grep as grep_module  # noqa: F401

    importlib.reload(grep_module)


async def test_grep_basic_match(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("hello\nworld\nhello again\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello"}, ctx)
    assert result.error is False
    assert "a.py:1:hello" in result.content
    assert "a.py:3:hello again" in result.content


async def test_grep_regex(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("foo1\nbar\nfoo42\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": r"foo\d+"}, ctx)
    assert "foo1" in result.content
    assert "foo42" in result.content
    assert "bar" not in result.content


async def test_grep_case_insensitive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("Hello\nHELLO\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello", "case_insensitive": True}, ctx)
    assert "Hello" in result.content
    assert "HELLO" in result.content


async def test_grep_glob_filter(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("match\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("match\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "match", "glob": "*.py"}, ctx)
    assert "a.py" in result.content
    assert "b.txt" not in result.content


async def test_grep_context_lines(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("line1\nline2\ntarget\nline4\nline5\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "target", "context_lines": 1}, ctx)
    assert "line2" in result.content
    assert "target" in result.content
    assert "line4" in result.content
    assert "line1" not in result.content
    assert "line5" not in result.content


async def test_grep_max_matches(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("m\n" * 500, encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "m", "max_matches": 10}, ctx)
    match_lines = [ln for ln in result.content.split("\n") if "a.py:" in ln]
    assert len(match_lines) == 10
    assert "truncated" in result.content.lower()


async def test_grep_missing_pattern(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "pattern" in result.content.lower()


async def test_grep_uses_rg_when_present(
    tmp_path: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ripgrep is on PATH, grep shells out to it. We assert the code path ran."""
    _load()
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")

    import vllama.agents.tools.grep as grep_mod

    called = {"rg": False}

    async def fake_rg(
        pattern: str,
        root: Path,
        glob: str | None,
        case_insensitive: bool,
        context_lines: int,
    ) -> list[str]:
        called["rg"] = True
        return [f"{root}/a.py:1:hello"]

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/rg" if name == "rg" else None)
    monkeypatch.setattr(grep_mod, "_run_ripgrep", fake_rg)

    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello"}, ctx)
    assert called["rg"] is True
    assert "hello" in result.content
