"""Tests for SubagentRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.subagent import SubagentDepthExceeded, SubagentRunner
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    register,
)


def _make_runtime(tmp_path: Path) -> ToolRuntime:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    return ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)


def _mock_responses(scripts: list[dict[str, Any]]) -> httpx.MockTransport:
    """Return an httpx MockTransport that serves each scripted message in order.
    It returns a JSON array for router calls.
    """
    import json as _json

    idx = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json_internal

        body = _json_internal.loads(request.read())
        # Router call: requests json format and has system prompt mentioning tool names
        is_router = body.get("response_format", {}).get("type") == "json_object"
        if is_router:
            # Return all tools mentioned in the system prompt
            system_prompt = body["messages"][0]["content"]
            # Extract tool names from "- name: desc" lines
            import re

            tool_names = re.findall(r"^- ([a-z_0-9]+):", system_prompt, re.MULTILINE)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": _json_internal.dumps(tool_names)}}]},
            )

        msg = scripts[idx["i"]]
        idx["i"] += 1
        return httpx.Response(200, json={"choices": [{"message": msg}]})

    return httpx.MockTransport(handler)


def _install_transport(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> None:
    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_subagent_single_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _mock_responses([{"content": "done!"}])
    _install_transport(monkeypatch, transport)

    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
    )
    result = await runner.run(
        description="greet", prompt="say hi", allowed_tools=None, depth=0
    )
    assert result == "done!"


async def test_subagent_tool_call_then_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Register a fake tool.
    async def echo(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(content=f"echoed: {args.get('x', '')}")

    register(
        ToolSpec(
            name="echo",
            definition={
                "type": "function",
                "function": {"name": "echo", "description": "", "parameters": {}},
            },
            tier="read",
            handler=echo,
        )
    )

    transport = _mock_responses(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"x": "hi"}'},
                    }
                ],
            },
            {"content": "tool reported: echoed: hi"},
        ]
    )
    _install_transport(monkeypatch, transport)

    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
    )
    result = await runner.run(
        description="use tool", prompt="call echo", allowed_tools=None, depth=0
    )
    assert "echoed: hi" in result


async def test_subagent_depth_exceeded_raises(tmp_path: Path) -> None:
    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
        depth_cap=2,
    )
    with pytest.raises(SubagentDepthExceeded):
        await runner.run(
            description="nested", prompt="go", allowed_tools=None, depth=2
        )


async def test_subagent_max_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model keeps emitting tool calls; subagent bails after max_iter."""

    async def noop(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(content="...")

    register(
        ToolSpec(
            name="noop",
            definition={
                "type": "function",
                "function": {"name": "noop", "description": "", "parameters": {}},
            },
            tier="read",
            handler=noop,
        )
    )

    tool_call_msg: dict[str, Any] = {
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
            }
        ],
    }

    # Endless tool calls.
    transport = _mock_responses([tool_call_msg] * 20)
    _install_transport(monkeypatch, transport)

    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
        max_iterations=3,
    )
    result = await runner.run(
        description="x", prompt="x", allowed_tools=None, depth=0
    )
    assert "max iterations" in result.lower()


async def test_subagent_allowed_tools_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """allowed_tools restricts what the subagent sees."""

    async def a_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(content="a")

    async def b_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(content="b")

    for name, handler in [("tool_a", a_handler), ("tool_b", b_handler)]:
        register(
            ToolSpec(
                name=name,
                definition={
                    "type": "function",
                    "function": {"name": name, "description": "", "parameters": {}},
                },
                tier="read",
                handler=handler,
            )
        )

    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["payload"] = req.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
    )
    await runner.run(
        description="x",
        prompt="x",
        allowed_tools=["tool_a"],
        depth=0,
    )
    body = captured["payload"].decode()
    assert "tool_a" in body
    assert "tool_b" not in body


async def test_subagent_http_error_returned_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=_make_runtime(tmp_path),
    )
    result = await runner.run(
        description="x", prompt="x", allowed_tools=None, depth=0
    )
    assert "error" in result.lower()
