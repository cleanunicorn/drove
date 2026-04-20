"""Tiny OpenAI-style chat-completion client used by router and evaluator."""

from __future__ import annotations

from typing import Any

import httpx


async def call_chat_completion(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str | None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST /v1/chat/completions and return the first choice's message dict.

    Unlike call_chat_json this does NOT request response_format=json_object,
    because the caller may need normal text or tool_calls. Returns the
    `message` sub-dict (with `content` and optionally `tool_calls`).
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
    }
    if tools:
        payload["tools"] = tools
    resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        return {"content": str(message)}
    return message


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
