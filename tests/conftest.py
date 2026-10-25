"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _offline_context_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the chat's startup context-size lookup off the network; meter tests override it."""

    async def unknown(base_url: str, api_key: str | None, model: str) -> None:
        return None

    monkeypatch.setattr("drove.tui.fetch_context_length", unknown)
