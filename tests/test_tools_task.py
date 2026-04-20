"""Tests for task tool (subagent delegation)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.subagent import SubagentRunner
from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    import vllama.agents.tools.task as m

    importlib.reload(m)


def _install(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": response}]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def _make_ctx_with_runner(tmp_path: Path, depth: int = 0) -> ToolContext:
    inner_ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    runtime = ToolRuntime(policy=Policy.trust_mode(), ctx=inner_ctx)
    runner = SubagentRunner(
        base_url="http://llm.test",
        model="m",
        api_key=None,
        runtime=runtime,
        depth_cap=2,
    )
    return ToolContext(
        cwd=tmp_path,
        cap_bytes=8192,
        cap_bytes_bash=32768,
        subagent_runner=runner,
        depth=depth,
    )


async def test_task_delegates_and_returns_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()
    _install(monkeypatch, {"content": "sub-reply"})
    ctx = _make_ctx_with_runner(tmp_path)
    spec = get_spec("task")
    assert spec is not None
    result = await spec.handler(
        {"description": "greet", "prompt": "say hi"}, ctx
    )
    assert result.error is False
    assert "sub-reply" in result.content


async def test_task_missing_args(tmp_path: Path) -> None:
    _load()
    ctx = _make_ctx_with_runner(tmp_path)
    spec = get_spec("task")
    assert spec is not None
    r1 = await spec.handler({"prompt": "x"}, ctx)
    r2 = await spec.handler({"description": "x"}, ctx)
    assert r1.error is True and "description" in r1.content.lower()
    assert r2.error is True and "prompt" in r2.content.lower()


async def test_task_no_runner_in_ctx(tmp_path: Path) -> None:
    _load()
    ctx = ToolContext(
        cwd=tmp_path,
        cap_bytes=8192,
        cap_bytes_bash=32768,
        subagent_runner=None,
    )
    spec = get_spec("task")
    assert spec is not None
    result = await spec.handler(
        {"description": "x", "prompt": "x"}, ctx
    )
    assert result.error is True
    assert "subagent" in result.content.lower()


async def test_task_depth_cap_hit(tmp_path: Path) -> None:
    _load()
    # depth=1, depth_cap=2 → runner.run is called with depth=2, which >= cap.
    ctx = _make_ctx_with_runner(tmp_path, depth=1)
    spec = get_spec("task")
    assert spec is not None
    result = await spec.handler(
        {"description": "x", "prompt": "x"}, ctx
    )
    assert result.error is True
    assert "depth" in result.content.lower()


async def test_task_allowed_tools_validation(tmp_path: Path) -> None:
    _load()
    ctx = _make_ctx_with_runner(tmp_path)
    spec = get_spec("task")
    assert spec is not None
    result = await spec.handler(
        {"description": "x", "prompt": "x", "allowed_tools": "read_file"}, ctx
    )
    assert result.error is True
    assert "allowed_tools" in result.content.lower()
