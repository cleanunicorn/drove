"""Tests for done-evaluator."""

from __future__ import annotations

from typing import Any

from vllama.agents.evaluator import Verdict, check_done
from vllama.config import EvaluatorConfig


async def test_check_done_disabled_returns_done_true() -> None:
    cfg = EvaluatorConfig(enabled=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("should not be called")

    v = await check_done(history=[], llm_call=llm, config=cfg)
    assert v.done is True


async def test_check_done_parses_json_object() -> None:
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return '{"done": false, "reason": "still stub"}'

    v = await check_done(
        history=[
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "I'll start."},
        ],
        llm_call=llm,
        config=cfg,
    )
    assert v.done is False
    assert "stub" in v.reason.lower()


async def test_check_done_fail_open_on_parse_error() -> None:
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return "not json"

    v = await check_done(history=[], llm_call=llm, config=cfg)
    assert v.done is True


async def test_check_done_fail_open_on_exception() -> None:
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("network down")

    v = await check_done(history=[], llm_call=llm, config=cfg)
    assert v.done is True


async def test_check_done_skip_when_no_todos_and_long_reply() -> None:
    """No todos yet in Phase 4, and a substantial reply → treat as done."""
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=True)
    calls: list[int] = []

    async def llm(messages: list[dict[str, Any]]) -> str:
        calls.append(1)
        return '{"done": false, "reason": "x"}'

    long_reply = "A" * 250
    v = await check_done(
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": long_reply},
        ],
        llm_call=llm,
        config=cfg,
    )
    assert v.done is True
    assert calls == []


async def test_check_done_short_reply_triggers_llm() -> None:
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=True)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return '{"done": true, "reason": "ok"}'

    v = await check_done(
        history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "k"},
        ],
        llm_call=llm,
        config=cfg,
    )
    assert v.done is True


async def test_verdict_defaults() -> None:
    v = Verdict(done=True, reason="ok")
    assert v.done is True
    assert v.reason == "ok"
