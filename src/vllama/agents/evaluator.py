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
_TODOS_PROMPT_CAP = 2000  # chars of serialized todos included in prompt


def _has_pending_todos(todos: list[dict[str, Any]] | None) -> bool:
    if not todos:
        return False
    return any(t.get("status") != "completed" for t in todos)


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


def _build_messages(
    history: list[dict[str, Any]],
    todos: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    user = _last_user(history)
    assistant = _last_assistant(history)
    system = (
        "Judge whether the user's request has been accomplished by the"
        ' assistant\'s latest reply. Return JSON: {"done": bool, "reason": string}.'
        " Done criteria:\n"
        "- The user's last request is fulfilled.\n"
        "- Any todos are all marked completed.\n"
        "Not-done signals:\n"
        "- Assistant promised an action without doing it.\n"
        "- Assistant answered a different question than asked.\n"
        "- Todos remain in pending or in_progress state.\n"
        "Return JSON only, no prose."
    )
    todos_block = ""
    if todos:
        todos_json = json.dumps(todos)[:_TODOS_PROMPT_CAP]
        todos_block = f"\n\nTodos:\n{todos_json}"
    user_prompt = (
        f"User's last request:\n{user[:2000]}\n\n"
        f"Assistant's latest reply:\n{assistant[:2000]}"
        f"{todos_block}\n\n"
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
    todos: list[dict[str, Any]] | None = None,
) -> Verdict:
    """Evaluate whether the current turn is complete.

    Fail-safe defaults: disabled → done=True; any llm_call or parse failure →
    done=True.

    Todos: any non-completed entry bypasses the long-reply short-circuit
    (so the model gets nudged back to work) and is serialized into the prompt
    so the LLM can check them directly.
    """
    if not config.enabled:
        return Verdict(done=True, reason="evaluator disabled")

    pending = _has_pending_todos(todos)

    if config.skip_when_no_todos_and_long_reply and not pending:
        assistant = _last_assistant(history)
        if len(assistant) >= _LONG_REPLY_THRESHOLD:
            return Verdict(done=True, reason="skip: no todos, long reply")

    messages = _build_messages(history, todos)
    try:
        raw = await llm_call(messages)
    except Exception:  # noqa: BLE001 — fail-safe on network/decoding error
        return Verdict(done=True, reason="evaluator llm error (fail-safe done)")

    v = _parse_verdict(raw)
    if v is None:
        return Verdict(done=True, reason="evaluator parse failure (fail-safe done)")
    return v
