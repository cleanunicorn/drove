"""Tests for web_fetch tool (httpx.MockTransport)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from vllama.agents.tools._base import ToolContext, get_spec


def _load() -> None:
    import vllama.agents.tools.webfetch as m

    importlib.reload(m)


@contextmanager
def _patched_client(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> Any:
    """Patch httpx.AsyncClient so web_fetch uses a MockTransport."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    yield


async def test_web_fetch_extracts_html_text(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()
    html = (
        b"<!DOCTYPE html><html><head><title>Hello Page</title></head>"
        b"<body><article><p>This is the main content paragraph.</p>"
        b"<p>Second paragraph with stuff.</p></article></body></html>"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=html, headers={"content-type": "text/html"})

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/"}, ctx)
    assert result.error is False
    assert "main content paragraph" in result.content


async def test_web_fetch_non_html_returns_raw(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"k":"v"}',
            headers={"content-type": "application/json"},
        )

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/x.json"}, ctx)
    assert result.error is False
    assert "non-HTML" in result.content
    assert '"k":"v"' in result.content


async def test_web_fetch_http_error(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, content=b"not found", headers={"content-type": "text/plain"}
        )

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/missing"}, ctx)
    assert result.error is True
    assert "404" in result.content


async def test_web_fetch_size_cap(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()
    big = b"A" * (16_000_000)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "text/html"})

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/big"}, ctx)
    assert result.error is True
    assert "too large" in result.content.lower()


async def test_web_fetch_timeout(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/slow"}, ctx)
    assert result.error is True
    assert "timed out" in result.content.lower()


async def test_web_fetch_connection_error(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load()

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    with _patched_client(monkeypatch, handler):
        spec = get_spec("web_fetch")
        assert spec is not None
        result = await spec.handler({"url": "https://example.test/"}, ctx)
    assert result.error is True
    assert "connection" in result.content.lower()


async def test_web_fetch_missing_url(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("web_fetch")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "url" in result.content.lower()


async def test_web_fetch_rejects_non_http(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("web_fetch")
    assert spec is not None
    result = await spec.handler({"url": "file:///etc/passwd"}, ctx)
    assert result.error is True
    assert "http" in result.content.lower()
