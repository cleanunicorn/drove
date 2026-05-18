"""Tests for observe integration in the proxy."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from drove.config import Config
from drove.observe import list_records
from drove.proxy import create_app


def make_config(tmp_path: Path, observe: bool = False) -> Config:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    observe_dir = tmp_path / "observe"
    return Config(
        models_dir=models_dir,
        listen_port=8080,
        observe=observe,
        observe_dir=observe_dir,
    )


def test_observe_enabled_writes_log(
    tmp_path: Path, aiter: Callable[[list[bytes]], AsyncIterator[bytes]]
) -> None:
    config = make_config(tmp_path, observe=True)
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    fake_response.aiter_raw = lambda: aiter(
        [b'{"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}']
    )

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        pass

    with (
        patch.object(manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", new_callable=AsyncMock, return_value=fake_response),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "testmodel", "messages": []},
            )

    assert resp.status_code == 200

    # The fire-and-forget task needs a moment to complete in the test environment.
    # Since TestClient runs the event loop, the task should already be done,
    # but let's check with a small sleep via asyncio just in case.
    records = list_records(config.observe_dir)
    assert len(records) == 1

    _path, record = records[0]
    assert record.model == "testmodel"
    assert record.endpoint == "v1/chat/completions"
    assert record.response_status == 200
    assert record.tokens_prompt == 10
    assert record.tokens_completion == 5


def test_observe_disabled_writes_nothing(
    tmp_path: Path, aiter: Callable[[list[bytes]], AsyncIterator[bytes]]
) -> None:
    config = make_config(tmp_path, observe=False)
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    fake_response.aiter_raw = lambda: aiter([b'{"choices": []}'])

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        pass

    with (
        patch.object(manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", new_callable=AsyncMock, return_value=fake_response),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/v1/chat/completions",
                json={"model": "testmodel", "messages": []},
            )

    assert not config.observe_dir.exists() or list_records(config.observe_dir) == []


def test_observe_record_contains_request_body(
    tmp_path: Path, aiter: Callable[[list[bytes]], AsyncIterator[bytes]]
) -> None:
    config = make_config(tmp_path, observe=True)
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    fake_response.aiter_raw = lambda: aiter([b'{"choices": []}'])

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        pass

    request_body = {"model": "testmodel", "messages": [{"role": "user", "content": "hello"}]}

    with (
        patch.object(manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", new_callable=AsyncMock, return_value=fake_response),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/v1/chat/completions", json=request_body)

    records = list_records(config.observe_dir)
    assert len(records) == 1
    _path, record = records[0]

    # The request body should contain the sent JSON
    assert record.request_body is not None
    parsed = json.loads(record.request_body)
    assert parsed["model"] == "testmodel"
    assert parsed["messages"][0]["content"] == "hello"
