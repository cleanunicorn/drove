"""Tests for the tool router."""

from __future__ import annotations

from typing import Any

from vllama.agents.router import select_tools
from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec
from vllama.config import RouterConfig


async def _dummy_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content="")


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        definition={
            "type": "function",
            "function": {"name": name, "description": f"{name} desc"},
        },
        tier="read",
        handler=_dummy_handler,
    )


async def test_select_skips_when_disabled() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(enabled=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("should not be called")

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=3,
    )
    assert result == specs


async def test_select_skips_on_first_iteration() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(skip_on_first_iteration=True)

    async def llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("should not be called")

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=1,
    )
    assert result == specs


async def test_select_filters_by_llm_response() -> None:
    specs = [_spec("a"), _spec("b"), _spec("c")]
    cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return '["a", "c"]'

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    names = {s.name for s in result}
    assert names == {"a", "c"}


async def test_select_fail_open_on_bad_json() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return "not json at all"

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    assert result == specs


async def test_select_fail_open_on_llm_exception() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("network down")

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    assert result == specs


async def test_select_permissive_prompt_says_err_toward_inclusion() -> None:
    specs = [_spec("a")]
    cfg = RouterConfig(enabled=True, permissive=True, skip_on_first_iteration=False)
    seen: dict[str, Any] = {}

    async def llm(messages: list[dict[str, Any]]) -> str:
        seen["messages"] = messages
        return '["a"]'

    await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    prompt_text = " ".join(m["content"] for m in seen["messages"])
    assert "inclu" in prompt_text.lower() or "err" in prompt_text.lower()


async def test_select_unknown_names_dropped() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return '["a", "nonexistent", "b"]'

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    names = {s.name for s in result}
    assert names == {"a", "b"}


async def test_select_empty_response_returns_empty() -> None:
    specs = [_spec("a"), _spec("b")]
    cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)

    async def llm(messages: list[dict[str, Any]]) -> str:
        return "[]"

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=llm,
        config=cfg,
        iteration=2,
    )
    assert result == []
