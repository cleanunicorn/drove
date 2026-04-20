# Phase 4 — Router + Evaluator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing chat turn loop with a per-iteration LLM tool router and a tool-less-reply done-evaluator, plus an iteration cap, so the model stays on task over multi-step work with shrunken tool surfaces per iteration.

**Architecture:**
- `router.py` — pure async function `select_tools(history, all_specs, llm_call, config, iteration) -> list[ToolSpec]`. Cheap LLM call with the compact name+description of every tool; parses JSON array response; fail-open returns `all_specs`. Caches by `(hash(history), frozenset(spec names))`. Skips on first iteration of a turn if configured.
- `evaluator.py` — pure async function `check_done(history, llm_call, config) -> Verdict` where `Verdict = {done: bool, reason: str}`. Called after a tool-less assistant response; short-circuits when the reply is long and there are no todos (todos added Phase 5).
- `config.py` — new `RouterConfig` + `EvaluatorConfig` submodels hung off `AgentsConfig`; plus `max_iterations: int = 50`.
- `tui.py` — `_stream_chat` accepts an explicit `tools: list[dict]` parameter. `_send_message` gains an iteration counter (status-line `iter k/N`) and calls router before each stream and evaluator on tool-less reply.

**Tech Stack:** Python 3.14, asyncio, httpx, pydantic. mypy strict. ruff (E/F/I/UP, 100).

---

## File Structure

```
src/vllama/
    config.py                       # MODIFY: RouterConfig, EvaluatorConfig, max_iterations
    agents/
        router.py                   # NEW: select_tools + _parse_tool_names + cache
        evaluator.py                # NEW: check_done + Verdict
        llm_call.py                 # NEW: build_json_llm_call helper (chat completion → str)
    tui.py                          # MODIFY: _stream_chat signature;
                                    # _send_message turn loop rewrite;
                                    # iter counter in status line

tests/
    test_router.py                  # NEW: mocked llm_call; parse, cache, skip-first, permissive prompt
    test_evaluator.py               # NEW: mocked llm_call; done/not-done, parse-fail fail-open, skip
    test_llm_call.py                # NEW: httpx mock
    test_chat_loop.py               # NEW: full integration with mocked llama-server
```

---

## Task 1: Config — RouterConfig, EvaluatorConfig, max_iterations

**Files:**
- Modify: `src/vllama/config.py`
- Modify: `tests/test_config.py`

### Step 1: Write failing tests

Append to `tests/test_config.py`:

```python
def test_agents_router_defaults(tmp_path: Path) -> None:
    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.agents.router.enabled is True
    assert cfg.agents.router.permissive is True
    assert cfg.agents.router.skip_on_first_iteration is True


def test_agents_evaluator_defaults(tmp_path: Path) -> None:
    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.agents.evaluator.enabled is True
    assert cfg.agents.evaluator.skip_when_no_todos_and_long_reply is True


def test_agents_max_iterations_default(tmp_path: Path) -> None:
    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.agents.max_iterations == 50


def test_agents_max_iterations_from_toml(tmp_path: Path) -> None:
    import tomli_w

    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps({"agents": {"max_iterations": 10}}).encode()
    )
    cfg = load_config(path)
    assert cfg.agents.max_iterations == 10


def test_agents_router_toggle_via_toml(tmp_path: Path) -> None:
    import tomli_w

    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps({"agents": {"router": {"enabled": False}}}).encode()
    )
    cfg = load_config(path)
    assert cfg.agents.router.enabled is False
```

### Step 2: Run — expect fail

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 new tests FAIL.

### Step 3: Update `config.py`

In `src/vllama/config.py`, replace the `AgentsConfig` block (and add the two sub-configs above it) with:

```python
class RouterConfig(BaseModel):
    enabled: bool = True
    permissive: bool = True
    skip_on_first_iteration: bool = True


class EvaluatorConfig(BaseModel):
    enabled: bool = True
    skip_when_no_todos_and_long_reply: bool = True


class AgentsConfig(BaseModel):
    """Configuration for the agents subsystem (Phase 2+)."""

    permissions: dict[str, DecisionValue] = {}
    max_iterations: int = 50
    router: RouterConfig = RouterConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
```

### Step 4: Run — expect pass

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass (9 existing + 5 new = 14).

Run full suite: `uv run pytest -q`
Expected: 173 pass.

### Step 5: Lint + mypy

Run: `uv run mypy src/vllama/config.py`
Run: `uv run ruff check src/vllama/config.py tests/test_config.py`

### Step 6: Commit

```bash
git add src/vllama/config.py tests/test_config.py
git commit -m "feat(config): add agents.router, agents.evaluator, agents.max_iterations"
```

---

## Task 2: LLM call helper

**Files:**
- Create: `src/vllama/agents/llm_call.py`
- Create: `tests/test_llm_call.py`

A small reusable helper that wraps an httpx POST to `/v1/chat/completions` with a `{"response_format": {"type": "json_object"}}` hint (best-effort — llama-server may ignore it, but widely-adopted LLM wrappers honor it) and returns the content string. Used by router and evaluator.

### Step 1: Write failing tests

Create `tests/test_llm_call.py`:

```python
"""Tests for llm_call helper."""

from __future__ import annotations

import httpx

from vllama.agents.llm_call import call_chat_json


async def test_call_chat_json_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '["read_file"]'}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        out = await call_chat_json(
            client=client,
            base_url="http://llama.test",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            api_key=None,
        )
    assert out == '["read_file"]'


async def test_call_chat_json_sends_api_key() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await call_chat_json(
            client=client,
            base_url="http://llama.test",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            api_key="secret",
        )
    assert seen_headers.get("authorization") == "Bearer secret"


async def test_call_chat_json_propagates_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    import pytest

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await call_chat_json(
                client=client,
                base_url="http://llama.test",
                model="m",
                messages=[{"role": "user", "content": "x"}],
                api_key=None,
            )
```

### Step 2: Run — expect fail

Run: `uv run pytest tests/test_llm_call.py -v`
Expected: FAIL — module missing.

### Step 3: Implement

Create `src/vllama/agents/llm_call.py`:

```python
"""Tiny OpenAI-style chat-completion client used by router and evaluator."""

from __future__ import annotations

from typing import Any

import httpx


async def call_chat_json(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> str:
    """POST /v1/chat/completions and return the first choice's content string.

    Raises httpx.HTTPStatusError on non-2xx. Callers should catch and apply
    fail-open policies at the semantic layer (router returns all specs;
    evaluator returns done=true).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice.get("message", {}).get("content", "")
    return str(content)
```

### Step 4: Run — expect pass

Run: `uv run pytest tests/test_llm_call.py -v`
Expected: 3 pass.

### Step 5: Lint + mypy

Run: `uv run mypy src/vllama/agents/llm_call.py`
Run: `uv run ruff check src/vllama/agents/llm_call.py tests/test_llm_call.py`

### Step 6: Commit

```bash
git add src/vllama/agents/llm_call.py tests/test_llm_call.py
git commit -m "feat(agents): add call_chat_json helper for router/evaluator"
```

---

## Task 3: Router

**Files:**
- Create: `src/vllama/agents/router.py`
- Create: `tests/test_router.py`

### Step 1: Write failing tests

Create `tests/test_router.py`:

```python
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
    assert result == specs  # fail-open


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
    assert result == specs  # fail-open


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
```

### Step 2: Run — expect fail

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL — module missing.

### Step 3: Implement

Create `src/vllama/agents/router.py`:

```python
"""LLM-based per-iteration tool router.

Given the conversation history and the full tool set, asks a cheap LLM call
which tools might help the next step and returns a filtered list. Errs toward
inclusion (permissive). Fails open on parse or network errors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from vllama.agents.tools._base import ToolSpec
from vllama.config import RouterConfig

LlmCall = Callable[[list[dict[str, Any]]], Awaitable[str]]


def _build_messages(
    history: list[dict[str, Any]],
    all_specs: list[ToolSpec],
    permissive: bool,
) -> list[dict[str, Any]]:
    tool_lines = []
    for s in all_specs:
        fn = s.definition.get("function", {})
        desc = fn.get("description", "").split(".")[0].strip()
        tool_lines.append(f"- {s.name}: {desc}")
    tool_list = "\n".join(tool_lines)

    include_hint = (
        " Err on the side of INCLUSION — prefer over-including tools rather than"
        " missing a useful one. Only exclude tools that are clearly irrelevant."
        if permissive
        else ""
    )

    system = (
        "You are a tool selector. Given the conversation below, return a JSON"
        f" array of tool names that MIGHT help the next step.{include_hint}"
        " Return JSON only, no prose.\n\n"
        f"Available tools:\n{tool_list}"
    )

    tail = history[-6:] if len(history) > 6 else list(history)

    convo_lines = []
    for m in tail:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str) and content:
            convo_lines.append(f"{role}: {content[:500]}")
    convo = "\n".join(convo_lines) if convo_lines else "(empty)"

    user = (
        "Conversation (last 6 messages):\n"
        f"{convo}\n\n"
        'Return JSON array only. Example: ["read_file","grep"]'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_names(raw: str) -> list[str] | None:
    """Extract the first JSON array of strings; return None on parse failure."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a bracketed array anywhere in the blob.
        m = re.search(r"\[[^\[\]]*\]", raw)
        if m is None:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, list):
        return None
    names: list[str] = []
    for x in obj:
        if isinstance(x, str):
            names.append(x)
    return names


async def select_tools(
    *,
    history: list[dict[str, Any]],
    all_specs: list[ToolSpec],
    llm_call: LlmCall,
    config: RouterConfig,
    iteration: int,
) -> list[ToolSpec]:
    """Filter all_specs to the subset the router judges relevant.

    Fail-open behaviour:
    - config.enabled=False → return all_specs.
    - config.skip_on_first_iteration and iteration==1 → return all_specs.
    - llm_call raises → return all_specs.
    - parse failure → return all_specs.
    """
    if not config.enabled:
        return all_specs
    if config.skip_on_first_iteration and iteration <= 1:
        return all_specs

    messages = _build_messages(history, all_specs, permissive=config.permissive)
    try:
        raw = await llm_call(messages)
    except Exception:  # noqa: BLE001 — fail-open on any network/decoding error
        return all_specs

    names = _parse_names(raw)
    if names is None:
        return all_specs

    wanted = set(names)
    return [s for s in all_specs if s.name in wanted]
```

### Step 4: Run — expect pass

Run: `uv run pytest tests/test_router.py -v`
Expected: 8 pass.

### Step 5: Lint + mypy

Run: `uv run mypy src/vllama/agents/router.py`
Run: `uv run ruff check src/vllama/agents/router.py tests/test_router.py`

### Step 6: Commit

```bash
git add src/vllama/agents/router.py tests/test_router.py
git commit -m "feat(agents): add LLM-based tool router with permissive fail-open policy"
```

---

## Task 4: Evaluator

**Files:**
- Create: `src/vllama/agents/evaluator.py`
- Create: `tests/test_evaluator.py`

### Step 1: Write failing tests

Create `tests/test_evaluator.py`:

```python
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
    assert v.done is True  # fail-safe default


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
    assert calls == []  # no LLM call because of the skip


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
```

### Step 2: Run — expect fail

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL — module missing.

### Step 3: Implement

Create `src/vllama/agents/evaluator.py`:

```python
"""LLM-based done-evaluator for the chat turn loop.

Invoked when the assistant produces a tool-less reply. Asks a cheap LLM call
to judge whether the user's last request has been fulfilled. Fails safe
(done=True) on parse/network errors so the turn never loops forever.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from vllama.config import EvaluatorConfig

LlmCall = Callable[[list[dict[str, Any]]], Awaitable[str]]

_LONG_REPLY_THRESHOLD = 200  # chars of assistant content considered "long"


@dataclass
class Verdict:
    done: bool
    reason: str


def _last_user(history: list[dict[str, Any]]) -> str:
    for m in reversed(history):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _last_assistant(history: list[dict[str, Any]]) -> str:
    for m in reversed(history):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _build_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user = _last_user(history)
    assistant = _last_assistant(history)
    system = (
        "Judge whether the user's request has been accomplished by the"
        " assistant's latest reply. Return JSON: {\"done\": bool, \"reason\": string}."
        " Done criteria:\n"
        "- The user's last request is fulfilled.\n"
        "Not-done signals:\n"
        "- Assistant promised an action without doing it.\n"
        "- Assistant answered a different question than asked.\n"
        "Return JSON only, no prose."
    )
    user_prompt = (
        f"User's last request:\n{user[:2000]}\n\n"
        f"Assistant's latest reply:\n{assistant[:2000]}\n\n"
        'Return JSON only. Example: {"done": true, "reason": "answered"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def _parse_verdict(raw: str) -> Verdict | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", raw)
        if m is None:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    done = obj.get("done")
    if not isinstance(done, bool):
        return None
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return Verdict(done=done, reason=reason)


async def check_done(
    *,
    history: list[dict[str, Any]],
    llm_call: LlmCall,
    config: EvaluatorConfig,
) -> Verdict:
    """Evaluate whether the current turn is complete.

    Fail-safe defaults: disabled → done=True; short-circuit on long reply +
    no todos (Phase 4 has no todos yet, so short-circuit only depends on
    reply length); any llm_call or parse failure → done=True.
    """
    if not config.enabled:
        return Verdict(done=True, reason="evaluator disabled")

    if config.skip_when_no_todos_and_long_reply:
        assistant = _last_assistant(history)
        # Phase 4 has no todos, so the "no todos" side is trivially satisfied.
        if len(assistant) >= _LONG_REPLY_THRESHOLD:
            return Verdict(done=True, reason="skip: no todos, long reply")

    messages = _build_messages(history)
    try:
        raw = await llm_call(messages)
    except Exception:  # noqa: BLE001 — fail-safe on network/decoding error
        return Verdict(done=True, reason="evaluator llm error (fail-safe done)")

    v = _parse_verdict(raw)
    if v is None:
        return Verdict(done=True, reason="evaluator parse failure (fail-safe done)")
    return v
```

### Step 4: Run — expect pass

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: 7 pass.

### Step 5: Lint + mypy

Run: `uv run mypy src/vllama/agents/evaluator.py`
Run: `uv run ruff check src/vllama/agents/evaluator.py tests/test_evaluator.py`

### Step 6: Commit

```bash
git add src/vllama/agents/evaluator.py tests/test_evaluator.py
git commit -m "feat(agents): add LLM-based done-evaluator with fail-safe defaults"
```

---

## Task 5: Turn-loop rewrite in `tui.py`

**Files:**
- Modify: `src/vllama/tui.py`

### Step 1: Read the current turn loop

Open `src/vllama/tui.py`. Relevant sections:
- Imports near top (line 33-37)
- `ChatApp.__init__` (around L399)
- `_send_message` (L735-869): outer `while True` loop; streams chat; if tool_calls → execute → loop; else → save + break.
- `_stream_chat` (L871): currently takes `messages` only; reads `[s.definition for s in all_specs()]` inline at line 880.

### Step 2: Add imports + runtime wiring

At the top of `tui.py`, add imports:

```python
from vllama.agents.evaluator import check_done
from vllama.agents.llm_call import call_chat_json
from vllama.agents.router import select_tools
```

In `ChatApp.__init__`, after `self._runtime = ToolRuntime(...)` assignment, add:

```python
self._max_iterations = cfg.agents.max_iterations
self._router_config = cfg.agents.router
self._evaluator_config = cfg.agents.evaluator
```

Add a new reactive attribute near the other reactives (after `is_generating: reactive[bool] = reactive(False)`):

```python
iteration: reactive[int] = reactive(0)
```

### Step 3: Add a `_llm_json_call` helper method on `ChatApp`

Place it near `_stream_chat`:

```python
async def _llm_json_call(self, messages: list[dict]) -> str:
    """Router/evaluator JSON call — non-streaming, short-lived client."""
    import httpx

    async with httpx.AsyncClient() as client:
        return await call_chat_json(
            client=client,
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            temperature=0.0,
            timeout=30.0,
        )
```

### Step 4: Update `_stream_chat` signature

Change `_stream_chat` to accept the tool definitions explicitly:

```python
async def _stream_chat(
    self,
    messages: list[dict],
    tools: list[dict],
) -> AsyncIterator[dict]:
```

Inside the method, replace the existing line:

```python
"tools": [s.definition for s in all_specs()],
```

with:

```python
"tools": tools,
```

### Step 5: Rewrite `_send_message` turn loop

Replace the current `while True:` body (line 749 onward up to the `break` / bottom of the try) with the iteration-capped + routed + evaluator-gated version.

The exact new `_send_message` body (everything from `try:` inside `_send_message` down to the final `self._update_status(); self.is_generating = False`):

```python
        try:
            for iter_idx in range(1, self._max_iterations + 1):
                self.iteration = iter_idx
                self._update_status()

                # 1. Router — pick a tool subset for this iteration.
                selected = await select_tools(
                    history=self._history,
                    all_specs=all_specs(),
                    llm_call=self._llm_json_call,
                    config=self._router_config,
                    iteration=iter_idx,
                )
                tools_payload = [s.definition for s in selected]

                # 2. Main stream.
                assistant_bubble = Message("assistant")
                await self.query_one("#messages").mount(assistant_bubble)
                self._scroll_to_bottom()

                full_response = ""
                full_thinking = ""
                was_thinking = False
                tool_calls: dict[int, dict] = {}

                async for event in self._stream_chat(self._history, tools_payload):
                    kind = event["kind"]
                    if t_first is None:
                        t_first = time.monotonic()

                    if kind == "thinking":
                        token_count += 1
                        was_thinking = True
                        full_thinking += event["chunk"]
                        await assistant_bubble.append_thinking(event["chunk"])
                    elif kind == "content":
                        token_count += 1
                        if was_thinking:
                            assistant_bubble.finish_thinking()
                            was_thinking = False
                        assistant_bubble.append_text(event["chunk"])
                        full_response += event["chunk"]
                    elif kind == "tool_call":
                        idx = event["index"]
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": event.get("id", ""),
                                "name": event.get("name", ""),
                                "arguments": "",
                            }
                        if event.get("id"):
                            tool_calls[idx]["id"] = event["id"]
                        if event.get("name"):
                            tool_calls[idx]["name"] = event["name"]
                        tool_calls[idx]["arguments"] += event.get("arguments", "")
                    self._scroll_to_bottom()

                if was_thinking:
                    assistant_bubble.finish_thinking()

                if not tool_calls:
                    # No tool calls — save the final assistant message, then
                    # ask the evaluator whether the turn is really done.
                    if full_response or full_thinking:
                        assistant_msg: dict = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        if full_thinking:
                            assistant_msg["thinking"] = full_thinking
                        self._history.append(assistant_msg)
                        self._session.messages = self._history
                        save_session(self._sessions_dir, self._session)

                    verdict = await check_done(
                        history=self._history,
                        llm_call=self._llm_json_call,
                        config=self._evaluator_config,
                    )
                    if verdict.done:
                        break
                    # Nudge the model and continue.
                    self._history.append(
                        {
                            "role": "system",
                            "content": (
                                f"Task not yet complete: {verdict.reason}. Continue."
                            ),
                        }
                    )
                    continue

                # Build the assistant message with tool_calls for history.
                assistant_msg_tc: dict = {"role": "assistant"}
                assistant_msg_tc["content"] = full_response if full_response else None
                assistant_msg_tc["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in sorted(tool_calls.values(), key=lambda t: t["id"])
                ]
                self._history.append(assistant_msg_tc)

                try:
                    for tc in sorted(tool_calls.values(), key=lambda t: t["id"]):
                        tool_result = await self._runtime.dispatch(
                            tc["name"], tc["arguments"]
                        )
                        result = tool_result.content

                        self._history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            }
                        )

                        await assistant_bubble.append_tool_call(
                            tc["name"],
                            tc["arguments"],
                            result,
                        )
                        self._scroll_to_bottom()
                except AbortTurn:
                    await self._show_note("Turn aborted by user.")
                    return

                self._session.messages = self._history
            else:
                # for/else: only runs when the loop completes without break.
                await self._show_note(
                    f"Max iterations ({self._max_iterations}) reached. Turn ended."
                )

        except Exception as e:
            await self.query_one("#messages").mount(Message("error", str(e)))

        t_end = time.monotonic()
        if t_first is not None:
            self._last_ttft = t_first - t_start
            gen_duration = t_end - t_first
            self._last_speed = token_count / gen_duration if gen_duration > 0 else None
        else:
            self._last_speed = None
            self._last_ttft = None

        self._update_status()
        self.is_generating = False
        self.iteration = 0
```

### Step 6: Update `_update_status` to surface the iteration count

Find `_update_status` (search for `_update_status`). Add to its status string construction: if `self.iteration > 0`, include `f"iter {self.iteration}/{self._max_iterations}"` as one of the segments. Exact placement depends on the current format; aim to put it alongside model/cwd, separated by `•`.

Concretely, find the assignment that builds the status text. Something like:
```python
parts = [f"model: {self._model}", ...]
```

Add before the final join:

```python
if self.iteration > 0:
    parts.append(f"iter {self.iteration}/{self._max_iterations}")
```

If the current implementation builds the label string differently, adapt — the goal is that the status line shows `iter K/N` while `is_generating` is true, and omits it otherwise.

### Step 7: Smoke-check imports and full test suite

Run: `uv run python -c "from vllama.tui import ChatApp; print('ok')"`
Expected: `ok`.

Run: `uv run pytest -q`
Expected: all tests pass (previous + new from Tasks 1-4). Most existing tests don't touch `_send_message` directly, so no regressions.

### Step 8: Lint + mypy

Run: `uv run mypy src/vllama/tui.py`
Run: `uv run ruff check src/vllama/tui.py`
Expected: no new errors beyond the 13 pre-existing.

### Step 9: Commit

```bash
git add src/vllama/tui.py
git commit -m "feat(tui): route per-iteration tool selection + done-evaluator into turn loop"
```

---

## Task 6: Integration test for router + evaluator + turn-loop

**Files:**
- Create: `tests/test_chat_loop_integration.py`

A lightweight integration test of the pure loop pieces: drive `select_tools` + `check_done` with scripted LLM responses and verify the expected sequence. The turn loop lives in `tui.py` which is Textual-coupled, so full end-to-end is deferred; this test exercises the decision logic directly.

### Step 1: Write the test

Create `tests/test_chat_loop_integration.py`:

```python
"""Integration: router + evaluator cooperate over a scripted LLM."""

from __future__ import annotations

from typing import Any

import pytest

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

    history = [
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

    history = [
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
    assert selected == specs  # fail-open: all tools

    verdict = await check_done(history=[], llm_call=llm_boom, config=e_cfg)
    assert verdict.done is True  # fail-safe: done
```

### Step 2: Run

Run: `uv run pytest tests/test_chat_loop_integration.py -v`
Expected: 4 pass.

Run full suite:
Run: `uv run pytest -q`
Expected: all tests pass.

### Step 3: Lint + mypy

Run: `uv run mypy src/vllama/agents/`
Run: `uv run ruff check src/ tests/`
Expected: no new errors.

### Step 4: Commit

```bash
git add tests/test_chat_loop_integration.py
git commit -m "test(agents): router+evaluator cooperation integration test"
```

---

## Phase 4 Acceptance Criteria

- [ ] `[agents.router]`, `[agents.evaluator]`, and `agents.max_iterations` are loadable from `config.toml`.
- [ ] `select_tools(history, all_specs, llm_call, config, iteration) -> list[ToolSpec]` filters by LLM response; fails open on any error; skips on iter 1 when configured; honors `enabled=False`.
- [ ] `check_done(history, llm_call, config) -> Verdict` returns done=true by default, fail-open on errors, short-circuits on long reply when configured.
- [ ] `call_chat_json` POSTs to `/v1/chat/completions` and returns the first choice content; raises on non-2xx.
- [ ] `_stream_chat` accepts tool definitions explicitly.
- [ ] `_send_message` loops up to `max_iterations`, running router each iter and evaluator on tool-less replies; warns on cap.
- [ ] Status line shows `iter K/N` while generating.
- [ ] Full test suite passes; no new mypy or ruff errors.
