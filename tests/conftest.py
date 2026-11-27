"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _offline_context_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the chat's startup context-size lookup off the network; meter tests override it."""

    async def unknown(base_url: str, api_key: str | None, model: str) -> None:
        return None

    monkeypatch.setattr("drove.tui.fetch_context_length", unknown)


@pytest.fixture(autouse=True)
def _config_in_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """No test reads the real config file: its saved tool approvals would leak into chat tests."""
    config = tmp_path_factory.mktemp("config") / "config.toml"
    monkeypatch.setenv("DROVE_CONFIG", str(config))
