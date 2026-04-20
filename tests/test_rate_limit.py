"""Tests for RateLimitedClient."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from vllama.agents.rate_limit import RateLimitedClient, RateLimitExceeded
from vllama.config import RateLimitSettings


class _FakeClock:
    """Deterministic clock. monotonic() advances only via sleep()."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds
        await asyncio.sleep(0)  # yield


def _transport(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


async def test_allows_within_limits() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(requests_per_minute=3)},
            clock=clock,
        )
        for _ in range(3):
            r = await rlc.post("http://x/", json={}, model="m")
            assert r.status_code == 200
        # Three calls within allowance; no sleep beyond base_delay (which is 0).
        assert clock.sleeps == []


async def test_minute_window_enforced() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(requests_per_minute=2)},
            clock=clock,
        )
        await rlc.post("http://x/", json={}, model="m")
        await rlc.post("http://x/", json={}, model="m")
        # Third call should sleep ~60s because window is full.
        await rlc.post("http://x/", json={}, model="m")
    assert clock.sleeps
    assert clock.sleeps[-1] >= 59


async def test_base_delay_applied() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(base_delay_ms=500)},
            clock=clock,
        )
        await rlc.post("http://x/", json={}, model="m")
    assert 0.5 in clock.sleeps


async def test_429_with_retry_after() -> None:
    clock = _FakeClock()
    hits = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(max_retries=3)},
            clock=clock,
        )
        r = await rlc.post("http://x/", json={}, model="m")
    assert r.status_code == 200
    assert 3.0 in clock.sleeps


async def test_429_without_retry_after_uses_expo_backoff() -> None:
    clock = _FakeClock()
    hits = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(max_retries=5)},
            clock=clock,
        )
        r = await rlc.post("http://x/", json={}, model="m")
    assert r.status_code == 200
    # Expected sleeps: 1.0 (2^0), 2.0 (2^1).
    assert 1.0 in clock.sleeps
    assert 2.0 in clock.sleeps


async def test_retries_exhausted_raises() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={"m": RateLimitSettings(max_retries=2)},
            clock=clock,
        )
        with pytest.raises(RateLimitExceeded):
            await rlc.post("http://x/", json={}, model="m")


async def test_per_model_isolation() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(
            client=raw,
            settings={
                "a": RateLimitSettings(requests_per_minute=1),
                "b": RateLimitSettings(requests_per_minute=1),
            },
            clock=clock,
        )
        await rlc.post("http://x/", json={}, model="a")
        await rlc.post("http://x/", json={}, model="b")  # should NOT sleep
    assert clock.sleeps == []


async def test_no_limits_no_throttle() -> None:
    clock = _FakeClock()

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=_transport(handler)) as raw:
        rlc = RateLimitedClient(client=raw, settings={}, clock=clock)
        for _ in range(10):
            await rlc.post("http://x/", json={}, model="unrated")
    assert clock.sleeps == []
