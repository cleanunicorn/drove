"""Shared fixtures for the agents test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Clear the global tool registry before every test in tests/."""
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """Default ToolContext for tool handlers: cwd=tmp_path, caps set to real defaults."""
    from vllama.agents.bash_procs import BgProcs

    return ToolContext(
        cwd=tmp_path,
        cap_bytes=8192,
        cap_bytes_bash=32768,
        bg_procs=BgProcs(),
    )
