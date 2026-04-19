"""Tests for /permits slash command handler logic (extracted, UI-free)."""

from __future__ import annotations

from pathlib import Path

from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import ToolContext


def _make_runtime() -> ToolRuntime:
    ctx = ToolContext(cwd=Path("/tmp"), cap_bytes=8192, cap_bytes_bash=32768)
    pol = Policy.from_config({"write_file": "auto", "bash": "deny"})
    return ToolRuntime(policy=pol, ctx=ctx)


def test_permits_summary_includes_overrides_and_session() -> None:
    from vllama.tui import render_permits_summary

    rt = _make_runtime()
    rt.session_permits.add("apply_patch")
    text = render_permits_summary(rt)
    assert "write_file" in text and "auto" in text
    assert "bash" in text and "deny" in text
    assert "apply_patch" in text
    # Tier defaults reminder present:
    assert "read" in text.lower()
    assert "mutate" in text.lower() or "prompt" in text.lower()


def test_permits_summary_no_overrides_no_session() -> None:
    from vllama.tui import render_permits_summary

    rt = ToolRuntime(
        policy=Policy(),
        ctx=ToolContext(cwd=Path("/tmp"), cap_bytes=8192, cap_bytes_bash=32768),
    )
    text = render_permits_summary(rt)
    assert "no overrides" in text.lower() or "default" in text.lower()
    assert "no session permits" in text.lower() or "none" in text.lower()


def test_permits_summary_trust_mode_labeled() -> None:
    from vllama.tui import render_permits_summary

    rt = ToolRuntime(
        policy=Policy.trust_mode(),
        ctx=ToolContext(cwd=Path("/tmp"), cap_bytes=8192, cap_bytes_bash=32768),
    )
    text = render_permits_summary(rt)
    assert "trust" in text.lower()
