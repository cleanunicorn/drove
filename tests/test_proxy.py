"""Tests for the FastAPI proxy app."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from drove.config import Config
from drove.proxy import create_app


def make_config(tmp_path: Path) -> Config:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return Config(
        models_dir=models_dir,
        listen_port=8080,
    )


def test_proxy_no_model_returns_400(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    app = create_app(config)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400
    assert "model" in resp.json()["detail"].lower()


def test_proxy_unknown_model_returns_404(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    app = create_app(config)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ghost-model", "messages": []},
        )
    assert resp.status_code == 404


def test_proxy_forwards_when_server_running(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    # Create a fake model directory
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    fake_response.aiter_raw = AsyncMock(return_value=aiter([b'{"ok": true}']))

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


def test_health_endpoint_no_model(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    app = create_app(config)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models"] == []
    assert body["server_running"] is False


def test_status_endpoint_no_model(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    app = create_app(config)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == []
    assert body["requests"]["total"] == 0
    assert body["tokens"]["total"] == 0
    assert "uptime_seconds" in body["server"]


def test_status_tracks_requests(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    # Use a lambda so aiter_raw() returns an async generator directly
    # (AsyncMock wraps in a coroutine which isn't async-iterable)
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
            client.post(
                "/v1/chat/completions",
                json={"model": "testmodel", "messages": []},
            )
            resp = client.get("/status")

    body = resp.json()
    assert body["requests"]["total"] == 1
    assert body["tokens"]["prompt"] == 10
    assert body["tokens"]["completion"] == 5
    assert body["tokens"]["total"] == 15


# ── multipart model extraction (audio endpoints) ────────────────────────────


def test_proxy_extracts_model_from_multipart_form(tmp_path: Path) -> None:
    """Audio requests carry 'model' as a multipart form field, not JSON."""
    config = make_config(tmp_path)
    app = create_app(config)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"RIFFxxxx", "audio/wav")},
            data={"model": "ghost-asr-model"},
        )
    # 404 (model not found) proves the model name was extracted from the
    # multipart body — otherwise the proxy would answer 400 "no model loaded".
    assert resp.status_code == 404


def test_proxy_forwards_multipart_body_intact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.models_dir / "asrmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "asrmodel" / "encoder-model.onnx").write_bytes(b"")

    app = create_app(config)
    manager = app.state.manager

    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "application/json"})
    fake_response.aiter_raw = AsyncMock(return_value=aiter([b'{"text": "hi"}']))

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        pass

    sent_requests: list[httpx.Request] = []

    async def fake_send(request: httpx.Request, **kwargs: object) -> httpx.Response:
        sent_requests.append(request)
        return fake_response

    with (
        patch.object(manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", side_effect=fake_send),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"RIFF-audio-bytes", "audio/wav")},
                data={"model": "asrmodel"},
            )

    assert resp.status_code == 200
    # The original multipart body (including the audio) reached the upstream
    assert len(sent_requests) == 1
    assert b"RIFF-audio-bytes" in sent_requests[0].content


# Helper for async generator mock
async def aiter(items: list[bytes]):  # type: ignore[return]
    for item in items:
        yield item
