"""Tests for llm_call helper."""

from __future__ import annotations

import httpx
import pytest

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
