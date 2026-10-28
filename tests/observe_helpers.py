"""Shared builders for observe tests: a request context and a pre-#122 legacy record."""

from __future__ import annotations

from typing import Any

from drove.observe import ObserveContext


def make_ctx(**overrides: Any) -> ObserveContext:
    """An ObserveContext for a chat completion; pass only what a test cares about."""
    fields: dict[str, Any] = {
        "timestamp": "2026-04-08T12:00:00",
        "model": "testmodel",
        "endpoint": "v1/chat/completions",
        "method": "POST",
        "request_headers": {},
        "request_body": b"{}",
    }
    return ObserveContext(**(fields | overrides))


def legacy_record(**overrides: Any) -> dict[str, Any]:
    """A record as written before #122: no raw body, hex request body."""
    return {
        "id": "20250101-000000-deadbeef",
        "timestamp": "2025-01-01T00:00:00",
        "model": "testmodel",
        "endpoint": "v1/audio/transcriptions",
        "method": "POST",
        "request_headers": {},
        "request_body": "fffe00",
        "response_status": 200,
        "response_headers": {},
        "response_body": "{}",
        **overrides,
    }
