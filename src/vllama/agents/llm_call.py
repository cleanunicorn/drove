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
