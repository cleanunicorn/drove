"""Tests for /bg listing helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.bash_procs import BgProcs


@pytest.fixture
def procs() -> BgProcs:
    return BgProcs()


async def test_render_bg_listing_empty(procs: BgProcs) -> None:
    from vllama.tui import render_bg_listing

    text = render_bg_listing(procs)
    assert "no background" in text.lower() or "none" in text.lower()


async def test_render_bg_listing_includes_ids(procs: BgProcs, tmp_path: Path) -> None:
    from vllama.tui import render_bg_listing

    a = await procs.start("sleep 5", cwd=tmp_path)
    b = await procs.start("sleep 5", cwd=tmp_path)
    text = render_bg_listing(procs)
    assert a in text
    assert b in text
    await procs.shutdown()


async def test_render_bg_listing_shows_status(procs: BgProcs, tmp_path: Path) -> None:
    from vllama.tui import render_bg_listing

    done = await procs.start("true", cwd=tmp_path)
    await procs.wait(done, timeout=2.0)
    text = render_bg_listing(procs)
    assert "exit" in text.lower() or "done" in text.lower()
