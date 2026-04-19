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
    - config.skip_on_first_iteration and iteration<=1 → return all_specs.
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
