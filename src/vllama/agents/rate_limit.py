"""Per-model rate limiting: token bucket + 429 retry.

Wraps `httpx.AsyncClient.post` so a single `RateLimitedClient` instance can
respect different per-model quotas when a caller may hit several backends.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from vllama.config import RateLimitSettings


class RateLimitExceeded(Exception):
    pass


@dataclass
class _ModelState:
    minute_window: deque[float] = field(default_factory=deque)
    hour_window: deque[float] = field(default_factory=deque)


def _parse_retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class RateLimitedClient:
    """httpx.AsyncClient wrapper that enforces per-model quotas + 429 retry."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        settings: dict[str, RateLimitSettings] | None = None,
        clock: Any = None,
    ) -> None:
        self._client = client
        self._settings = settings or {}
        self._states: dict[str, _ModelState] = {}
        # Injectable clock for tests. Must expose monotonic() and async sleep.
        self._clock = clock
        self._lock = asyncio.Lock()

    def _cfg(self, model: str) -> RateLimitSettings:
        return self._settings.get(model, RateLimitSettings())

    def _state(self, model: str) -> _ModelState:
        st = self._states.get(model)
        if st is None:
            st = _ModelState()
            self._states[model] = st
        return st

    def _now(self) -> float:
        if self._clock is not None:
            return float(self._clock.monotonic())
        return time.monotonic()

    async def _sleep(self, seconds: float) -> None:
        if self._clock is not None and hasattr(self._clock, "sleep"):
            await self._clock.sleep(seconds)
        else:
            await asyncio.sleep(seconds)

    async def _wait_for_slot(self, model: str) -> None:
        cfg = self._cfg(model)
        if cfg.requests_per_minute is None and cfg.requests_per_hour is None:
            return
        st = self._state(model)
        while True:
            now = self._now()
            # Evict expired timestamps.
            while st.minute_window and now - st.minute_window[0] >= 60:
                st.minute_window.popleft()
            while st.hour_window and now - st.hour_window[0] >= 3600:
                st.hour_window.popleft()

            min_wait = 0.0
            if (
                cfg.requests_per_minute is not None
                and len(st.minute_window) >= cfg.requests_per_minute
            ):
                min_wait = max(
                    min_wait, 60 - (now - st.minute_window[0])
                )
            if (
                cfg.requests_per_hour is not None
                and len(st.hour_window) >= cfg.requests_per_hour
            ):
                min_wait = max(
                    min_wait, 3600 - (now - st.hour_window[0])
                )

            if min_wait <= 0:
                st.minute_window.append(now)
                st.hour_window.append(now)
                return
            await self._sleep(min_wait)

    async def post(
        self,
        url: str,
        *,
        json: Any,
        headers: dict[str, str] | None = None,
        model: str,
        timeout: float | None = None,
    ) -> httpx.Response:
        cfg = self._cfg(model)
        if cfg.base_delay_ms > 0:
            await self._sleep(cfg.base_delay_ms / 1000)

        for attempt in range(cfg.max_retries):
            async with self._lock:
                await self._wait_for_slot(model)

            resp = await self._client.post(
                url, json=json, headers=headers, timeout=timeout
            )
            if resp.status_code != 429:
                return resp

            retry_after = _parse_retry_after(resp)
            if retry_after is None:
                retry_after = min(
                    2.0 ** attempt, float(cfg.retry_max_backoff_s)
                )
            else:
                retry_after = min(retry_after, float(cfg.retry_max_backoff_s))

            await self._sleep(retry_after)

        raise RateLimitExceeded(
            f"rate limit exceeded for model {model!r} after {cfg.max_retries} retries"
        )


__all__ = ["RateLimitExceeded", "RateLimitedClient"]
