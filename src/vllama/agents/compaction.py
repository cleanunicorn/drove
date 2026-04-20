"""History compaction: summarize older turns into one message when over budget.

Keeps the original system prompt (history[0]) in place, replaces the middle
with a generated summary, preserves the last N messages verbatim.

Fails open — on LLM error or unexpected shape, returns the original history.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from vllama.config import CompactionConfig

LlmCall = Callable[[list[dict[str, Any]]], Awaitable[str]]


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Crude 4-chars-per-token estimator that works well enough for thresholds."""
    total = 0
    for m in messages:
        try:
            total += len(json.dumps(m, default=str))
        except (TypeError, ValueError):
            total += len(str(m))
    return total // 4


def _summarize_prompt(head: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = (
        "Summarize the conversation below into a concise brief preserving:"
        " decisions, file paths, tool results, and open tasks. Omit verbose"
        " tool output. Under 300 words."
    )
    blob = json.dumps(head, default=str)[:16000]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Conversation so far:\n{blob}"},
    ]


async def maybe_compact(
    *,
    history: list[dict[str, Any]],
    ctx_size: int,
    llm_call: LlmCall,
    config: CompactionConfig,
) -> list[dict[str, Any]]:
    """Compact `history` if it exceeds `ctx_size * threshold` tokens.

    Returns a new list; does not mutate the input.
    """
    if not config.enabled:
        return history
    if len(history) <= config.keep_tail_messages + 1:
        return history  # nothing meaningful to compact

    tokens = estimate_tokens(history)
    if tokens < int(ctx_size * config.threshold):
        return history

    keep_tail = config.keep_tail_messages
    # Keep index 0 (assumed system prompt) and the last keep_tail.
    head_end = len(history) - keep_tail
    head = history[1:head_end] if head_end > 1 else []
    tail = history[head_end:]
    if not head:
        return history  # nothing worth summarizing in the middle

    try:
        summary = await llm_call(_summarize_prompt(head))
    except Exception:  # noqa: BLE001 — fail-open
        return history

    if not isinstance(summary, str) or not summary.strip():
        return history

    summary_msg: dict[str, Any] = {
        "role": "system",
        "content": f"[Earlier conversation summary]\n{summary.strip()}",
    }
    return [history[0], summary_msg, *tail]
