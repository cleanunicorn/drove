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


async def test_pending_todos_bypass_long_reply_skip() -> None:
    """Any non-completed todo forces the LLM call even with a long reply."""
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=True)
    called: list[int] = []

    async def llm(_messages: list[dict[str, Any]]) -> str:
        called.append(1)
        return '{"done": false, "reason": "still pending"}'

    v = await check_done(
        history=[
            {"role": "user", "content": "plan it"},
            {"role": "assistant", "content": "A" * 300},
        ],
        llm_call=llm,
        config=cfg,
        todos=[{"id": "1", "content": "x", "status": "pending"}],
    )
    assert called == [1]
    assert v.done is False


async def test_completed_todos_preserve_long_reply_skip() -> None:
    """All-completed todos behave the same as no todos."""
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=True)
    called: list[int] = []

    async def llm(_messages: list[dict[str, Any]]) -> str:
        called.append(1)
        return '{"done": false, "reason": "x"}'

    v = await check_done(
        history=[
            {"role": "user", "content": "plan it"},
            {"role": "assistant", "content": "A" * 300},
        ],
        llm_call=llm,
        config=cfg,
        todos=[{"id": "1", "content": "x", "status": "completed"}],
    )
    assert called == []
    assert v.done is True


async def test_todos_appear_in_prompt() -> None:
    cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)
    seen: dict[str, Any] = {}

    async def llm(messages: list[dict[str, Any]]) -> str:
        seen["messages"] = messages
        return '{"done": true, "reason": "ok"}'

    await check_done(
        history=[
            {"role": "user", "content": "do stuff"},
            {"role": "assistant", "content": "I did."},
        ],
        llm_call=llm,
        config=cfg,
        todos=[{"id": "1", "content": "write code", "status": "completed"}],
    )
    prompt_text = " ".join(m["content"] for m in seen["messages"])
    assert "write code" in prompt_text
    assert "completed" in prompt_text.lower()
