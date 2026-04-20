"""Tests for history compaction."""

from __future__ import annotations

from typing import Any

from vllama.agents.compaction import estimate_tokens, maybe_compact
from vllama.config import CompactionConfig


async def _never_called(_messages: list[dict[str, Any]]) -> str:
    raise AssertionError("LLM should not be called")


async def test_estimate_tokens_monotonic() -> None:
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "x" * 10_000}]
    assert estimate_tokens(small) < estimate_tokens(big)


async def test_disabled_returns_history_as_is() -> None:
    cfg = CompactionConfig(enabled=False)
    history = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": "x" * 10_000}
    ] * 20
    out = await maybe_compact(
        history=history, ctx_size=1024, llm_call=_never_called, config=cfg
    )
    assert out == history


async def test_under_threshold_returns_history_as_is() -> None:
    cfg = CompactionConfig(enabled=True, threshold=0.7, keep_tail_messages=6)
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi back"},
    ]
    out = await maybe_compact(
        history=history, ctx_size=1_000_000, llm_call=_never_called, config=cfg
    )
    assert out == history


async def test_over_threshold_summarizes() -> None:
    cfg = CompactionConfig(enabled=True, threshold=0.7, keep_tail_messages=2)
    history: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
    ]
    for i in range(10):
        history.append({"role": "user", "content": f"msg {i} " + "x" * 4000})
        history.append({"role": "assistant", "content": f"reply {i} " + "y" * 4000})

    async def llm(_messages: list[dict[str, Any]]) -> str:
        return "SUMMARY: user asked 10 questions."

    out = await maybe_compact(
        history=history, ctx_size=10_000, llm_call=llm, config=cfg
    )
    # Must be shorter than original.
    assert len(out) < len(history)
    # Head preserved.
    assert out[0]["content"] == "sys"
    # Summary injected.
    assert "SUMMARY" in out[1]["content"]
    assert "Earlier conversation summary" in out[1]["content"]
    # Tail preserved.
    assert out[-1] == history[-1]
    assert out[-2] == history[-2]


async def test_fail_open_on_llm_exception() -> None:
    cfg = CompactionConfig(enabled=True, threshold=0.0, keep_tail_messages=2)
    history: list[dict[str, Any]] = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": "big " + "x" * 5000} for _ in range(10)
    ]

    async def llm(_messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("down")

    out = await maybe_compact(
        history=history, ctx_size=100, llm_call=llm, config=cfg
    )
    assert out == history  # unchanged


async def test_empty_summary_returns_history() -> None:
    cfg = CompactionConfig(enabled=True, threshold=0.0, keep_tail_messages=2)
    history: list[dict[str, Any]] = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": "big " + "x" * 5000} for _ in range(10)
    ]

    async def llm(_messages: list[dict[str, Any]]) -> str:
        return "   "

    out = await maybe_compact(
        history=history, ctx_size=100, llm_call=llm, config=cfg
    )
    assert out == history


async def test_short_history_skips_even_when_above_threshold() -> None:
    """No middle to summarize."""
    cfg = CompactionConfig(enabled=True, threshold=0.0, keep_tail_messages=6)
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    out = await maybe_compact(
        history=history,
        ctx_size=1,
        llm_call=_never_called,
        config=cfg,
    )
    assert out == history
