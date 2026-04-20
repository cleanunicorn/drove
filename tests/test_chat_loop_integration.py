"""Integration: router + evaluator cooperate over a scripted LLM."""

from __future__ import annotations

from typing import Any

from vllama.agents.evaluator import check_done
from vllama.agents.router import select_tools
from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec
from vllama.config import EvaluatorConfig, RouterConfig


async def _dummy_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content="")


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        definition={
            "type": "function",
            "function": {"name": name, "description": f"{name}"},
        },
        tier="read",
        handler=_dummy_handler,
    )


async def test_router_picks_file_tools_then_evaluator_says_done() -> None:
    specs = [_spec("read_file"), _spec("bash"), _spec("grep")]
    r_cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)
    e_cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)

    async def router_llm(_messages: list[dict[str, Any]]) -> str:
        return '["read_file","grep"]'

    async def evaluator_llm(_messages: list[dict[str, Any]]) -> str:
        return '{"done": true, "reason": "file read"}'

    history: list[dict[str, Any]] = [
        {"role": "user", "content": "read the README for me"},
    ]

    selected = await select_tools(
        history=history,
        all_specs=specs,
        llm_call=router_llm,
        config=r_cfg,
        iteration=2,
    )
    assert {s.name for s in selected} == {"read_file", "grep"}

    history.append({"role": "assistant", "content": "Here's the README content..."})
    verdict = await check_done(
        history=history,
        llm_call=evaluator_llm,
        config=e_cfg,
    )
    assert verdict.done is True


async def test_evaluator_nudge_then_done() -> None:
    """First eval says not done with a reason; second says done."""
    e_cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)
    responses = iter(
        [
            '{"done": false, "reason": "you only said you would"}',
            '{"done": true, "reason": "actually did it"}',
        ]
    )

    async def evaluator_llm(_messages: list[dict[str, Any]]) -> str:
        return next(responses)

    history: list[dict[str, Any]] = [
        {"role": "user", "content": "write a poem"},
        {"role": "assistant", "content": "Sure, I'll write one."},
    ]
    v1 = await check_done(history=history, llm_call=evaluator_llm, config=e_cfg)
    assert v1.done is False
    assert "only said you would" in v1.reason

    history.append({"role": "assistant", "content": "Roses are red..."})
    v2 = await check_done(history=history, llm_call=evaluator_llm, config=e_cfg)
    assert v2.done is True


async def test_router_skip_first_iter_keeps_all_specs() -> None:
    specs = [_spec("a"), _spec("b"), _spec("c")]
    r_cfg = RouterConfig(enabled=True, skip_on_first_iteration=True)

    async def never(_messages: list[dict[str, Any]]) -> str:
        raise AssertionError("router should not be called on iter 1")

    result = await select_tools(
        history=[{"role": "user", "content": "hi"}],
        all_specs=specs,
        llm_call=never,
        config=r_cfg,
        iteration=1,
    )
    assert result == specs


async def test_router_then_both_fail_open() -> None:
    specs = [_spec("a"), _spec("b")]
    r_cfg = RouterConfig(enabled=True, skip_on_first_iteration=False)
    e_cfg = EvaluatorConfig(enabled=True, skip_when_no_todos_and_long_reply=False)

    async def llm_boom(_messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("down")

    selected = await select_tools(
        history=[], all_specs=specs, llm_call=llm_boom, config=r_cfg, iteration=2
    )
    assert selected == specs

    verdict = await check_done(history=[], llm_call=llm_boom, config=e_cfg)
    assert verdict.done is True
