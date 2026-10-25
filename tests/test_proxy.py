"""Tests for the FastAPI proxy app."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
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


def test_proxy_forwards_speech_json_and_binary_audio_intact(tmp_path: Path) -> None:
    import json

    config = make_config(tmp_path)
    app = create_app(config)
    manager = app.state.manager
    audio = b"RIFF\x00\xff\x80WAVE\x00\x01"
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.headers = httpx.Headers({"content-type": "audio/wav"})
    fake_response.aiter_raw = lambda: aiter([audio[:7], audio[7:]])
    sent_requests: list[httpx.Request] = []

    async def fake_send(request: httpx.Request, **kwargs: object) -> httpx.Response:
        sent_requests.append(request)
        return fake_response

    body = {"model": "kokoro", "input": "Hello.", "voice": "alloy", "response_format": "wav"}
    with (
        patch.object(manager, "ensure_running", new_callable=AsyncMock) as ensure,
        patch("httpx.AsyncClient.send", side_effect=fake_send),
    ):
        with TestClient(app) as client:
            response = client.post("/v1/audio/speech", json=body)
    ensure.assert_awaited_once_with("kokoro", claim=True)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == audio
    assert len(sent_requests) == 1
    assert sent_requests[0].url.path == "/v1/audio/speech"
    assert json.loads(sent_requests[0].content) == body


@pytest.mark.parametrize(
    "endpoint,status",
    [("audio/speech", 200), ("audio/transcriptions", 502), ("chat/completions", 502)],
)
async def test_speech_can_outlast_the_regular_upstream_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, status: int
) -> None:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    config = make_config(tmp_path)
    app = create_app(config)
    # Use the current config, as a running proxy does after a config reload.
    app.state.config = config.model_copy(update={"speech_timeout_seconds": 2})
    # Scale the ordinary 300-second timeout down to keep this network test fast.
    app.state.client.timeout = httpx.Timeout(0.05)
    requests: list[httpx.Request] = []

    async def capture(request: httpx.Request) -> None:
        requests.append(request)

    app.state.client.event_hooks["request"] = [capture]
    audio = b"RIFF-slow-speech"
    finished = asyncio.Event()

    async def worker(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            # A buffered engine sends no response headers until synthesis finishes.
            await asyncio.sleep(0.15)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\n"
                + f"Content-Length: {len(audio)}\r\nConnection: close\r\n\r\n".encode()
                + audio
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    server = await asyncio.start_server(worker, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    manager = app.state.manager
    try:
        with (
            patch.object(manager, "ensure_running", new_callable=AsyncMock) as ensure,
            patch.object(manager, "base_url_for", return_value=f"http://127.0.0.1:{port}"),
            patch.object(manager, "request_finished") as release,
        ):
            async with (
                server,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://proxy"
                ) as client,
            ):
                response = await client.post(f"/v1/{endpoint}", json={"model": "kitten"})
            ensure.assert_awaited_once_with("kitten", claim=True)
            release.assert_called_once_with("kitten")
        assert response.status_code == status
        assert requests[0].extensions["timeout"] == {
            "connect": 0.05,
            "read": 2 if endpoint == "audio/speech" else 0.05,
            "write": 0.05,
            "pool": 0.05,
        }
        if status == 200:
            assert response.content == audio
        else:
            assert "ReadTimeout" in response.json()["detail"]
        assert app.state.stats.active_requests == 0
    finally:
        await app.state.client.aclose()
        await asyncio.wait_for(finished.wait(), timeout=2)


@pytest.mark.parametrize("endpoint", ["transcriptions", "speech"])
@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize("duration", ["2.5", "nan", "-1", "invalid", None])
def test_audio_status(tmp_path: Path, endpoint: str, status: int, duration: str | None) -> None:
    app = create_app(make_config(tmp_path))
    headers = {"content-type": "audio/wav" if endpoint == "speech" else "application/json"}
    if duration is not None:
        headers["x-drove-audio-seconds"] = duration

    async def send(request: httpx.Request, **kwargs: object) -> httpx.Response:
        # Advance the clock before returning headers: inference happens before streaming.
        clock[0] += 4
        response = MagicMock(spec=httpx.Response)
        response.status_code = status
        response.headers = httpx.Headers(headers)
        response.aiter_raw = lambda: aiter([b"audio or transcript"])
        return response

    clock = [100.0]
    with (
        patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock),
        patch("httpx.AsyncClient.send", side_effect=send),
        patch("drove.proxy.time.monotonic", side_effect=lambda: clock[0]),
        TestClient(app) as client,
    ):
        if endpoint == "speech":
            response = client.post("/v1/audio/speech", json={"model": "voice", "input": "hi"})
        else:
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", b"audio")},
                data={"model": "asr"},
            )
        data = client.get("/status").json()
    assert response.status_code == status
    assert response.content == b"audio or transcript"
    metrics = data["audio"][endpoint]
    assert metrics["requests"] == (1 if status == 200 else 0)
    assert metrics["audio_seconds"] == (2.5 if status == 200 and duration == "2.5" else 0)
    assert metrics["processing_seconds"] == (4 if status == 200 else 0)
    assert data["tokens"]["total"] == 0
    assert data["tokens"]["ttft"]["last_seconds"] is None
    assert data["requests"]["active"] == 0


def test_model_listing_keeps_core_fields_and_adds_trustworthy_chat_metadata(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.models_dir / "chat.gguf").write_bytes(b"chat")
    (config.models_dir / "vision.gguf").write_bytes(b"vision")
    (config.models_dir / "vision.toml").write_text('mmproj = "projector.gguf"\n')
    for name, extension in (("speech", ".onnx"), ("voice", ".tts")):
        folder = config.models_dir / name
        folder.mkdir()
        (folder / (name + extension)).write_bytes(b"data")
    (config.models_dir / "override.gguf").write_bytes(b"data")
    (config.models_dir / "override.toml").write_text('backend = "asr"\n')
    (config.models_dir / "broken.gguf").write_bytes(b"data")
    (config.models_dir / "broken.toml").write_text('ctx_size = "bad"\n')
    from unittest.mock import PropertyMock

    with (
        patch(
            "drove.server_manager.ServerManager.loaded_models",
            new_callable=PropertyMock,
            return_value=["chat"],
        ),
        TestClient(create_app(config)) as client,
    ):
        response = client.get("/v1/models")
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()["data"]}
    assert set(rows) == {"chat", "vision", "speech", "voice", "override", "broken"}
    for name, row in rows.items():
        assert {key: row[key] for key in ("id", "object", "created", "owned_by")} == {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }
    assert rows["chat"]["drove"] == {
        "chat": True,
        "loaded": True,
        "size_bytes": 4,
        "capabilities": ["text", "tools"],
    }
    assert "vision" in rows["vision"]["drove"]["capabilities"]
    assert rows["vision"]["drove"]["loaded"] is False
    assert all(rows[name]["drove"]["chat"] is False for name in ("speech", "voice", "override"))
    assert rows["broken"]["drove"]["chat"] is None


@pytest.mark.parametrize(
    "default,explicit,loaded,expected",
    [
        ("preferred", None, None, "preferred"),
        ("preferred", None, "other", "preferred"),
        ("missing", "chosen", "other", "chosen"),
        (None, None, "other", "other"),
        ("org/model:Q4_K_M", None, None, "org/model:Q4_K_M"),
        ("preferred", "", None, ""),
    ],
)
def test_default_model_routing_and_claims(
    tmp_path: Path,
    default: str | None,
    explicit: str | None,
    loaded: str | None,
    expected: str,
) -> None:
    config = make_config(tmp_path).model_copy(update={"default_model": default})
    app = create_app(config)
    manager = app.state.manager
    payload = {"messages": []}
    if explicit is not None:
        payload["model"] = explicit
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.headers = httpx.Headers({"content-type": "application/json"})

    async def response_body():
        yield b'{"ok": true}'

    response.aiter_raw = response_body
    with (
        patch.object(manager, "ensure_running", new_callable=AsyncMock) as ensure,
        patch.object(
            type(manager), "is_running", new_callable=PropertyMock, return_value=loaded is not None
        ),
        patch.object(
            type(manager), "current_model", new_callable=PropertyMock, return_value=loaded
        ),
        patch.object(manager, "base_url_for", return_value="http://backend:9999") as url,
        patch.object(manager, "request_started") as started,
        patch.object(manager, "request_finished") as finished,
        patch.object(
            app.state.client, "send", new_callable=AsyncMock, return_value=response
        ) as send,
        TestClient(app) as client,
    ):
        result = client.post("/v1/chat/completions", json=payload)
        assert result.status_code == 200
        upstream = send.call_args.args[0]
        assert upstream.content == result.request.content
        assert str(upstream.url) == "http://backend:9999/v1/chat/completions"
        url.assert_called_once_with(expected)
        if default is None and explicit is None:
            ensure.assert_not_awaited()
            started.assert_called_once_with(expected)
        else:
            ensure.assert_awaited_once_with(expected, claim=True)
            started.assert_not_called()
        finished.assert_called_once_with(expected)
        assert app.state.stats.active_requests == 0
        assert app.state.stats.error_count == 0


@pytest.mark.parametrize("loaded", [False, True])
def test_missing_default_returns_actionable_404(tmp_path: Path, loaded: bool) -> None:
    config = make_config(tmp_path).model_copy(update={"default_model": "missing"})
    app = create_app(config)
    with (
        patch.object(
            type(app.state.manager), "is_running", new_callable=PropertyMock, return_value=loaded
        ),
        TestClient(app) as client,
    ):
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 404
        assert "missing" in response.json()["detail"]
        assert "drove models list" in response.json()["detail"]
        assert app.state.stats.active_requests == 0
        assert app.state.stats.error_count == 1


@pytest.mark.parametrize("error", [TimeoutError("startup timed out"), RuntimeError("failed")])
def test_default_startup_error_returns_503(tmp_path: Path, error: Exception) -> None:
    config = make_config(tmp_path).model_copy(update={"default_model": "preferred"})
    app = create_app(config)
    with (
        patch.object(app.state.manager, "ensure_running", side_effect=error),
        TestClient(app) as client,
    ):
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 503
        assert app.state.stats.active_requests == 0
        assert app.state.stats.error_count == 1


def test_default_model_multipart_preserves_body(tmp_path: Path) -> None:
    config = make_config(tmp_path).model_copy(update={"default_model": "speech"})
    app = create_app(config)
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.headers = httpx.Headers({"content-type": "application/json"})

    async def response_body():
        yield b'{"text": "hello"}'

    response.aiter_raw = response_body
    with (
        patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock) as ensure,
        patch.object(
            app.state.client, "send", new_callable=AsyncMock, return_value=response
        ) as send,
        TestClient(app) as client,
    ):
        result = client.post("/v1/audio/transcriptions", files={"file": ("audio.wav", b"RIFF")})
        assert result.status_code == 200
        ensure.assert_awaited_once_with("speech", claim=True)
        assert send.call_args.args[0].content == result.request.content


def test_default_model_reload_change_and_removal(tmp_path: Path) -> None:
    from drove.proxy import _reload_config

    config = make_config(tmp_path).model_copy(update={"default_model": "first"})
    app = create_app(config)
    path = tmp_path / "config.toml"
    with (
        patch.object(
            app.state.manager, "ensure_running", side_effect=FileNotFoundError("not downloaded")
        ) as ensure,
        TestClient(app) as client,
    ):
        assert client.post("/v1/chat/completions", json={}).status_code == 404
        ensure.assert_awaited_with("first", claim=True)
        config.model_copy(update={"default_model": "second"}).save(path)
        _reload_config(app, path)
        assert client.post("/v1/chat/completions", json={}).status_code == 404
        ensure.assert_awaited_with("second", claim=True)
        config.model_copy(update={"default_model": None}).save(path)
        _reload_config(app, path)
        assert client.post("/v1/chat/completions", json={}).status_code == 400
        assert ensure.await_count == 2
        assert app.state.stats.active_requests == 0


@pytest.mark.parametrize("selection", ["omitted", "null", "explicit", "unset"])
async def test_default_speech_reaches_real_worker(tmp_path: Path, selection: str) -> None:
    from drove.workers.tts_runtime.worker import create_tts_app

    class FakeEngine:
        native_speed = True

        def synthesize(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
            assert (text, voice, speed) == ("Hello", "alloy", 1.0)
            return b"\x00\x00" * 240, 24000

    default = None if selection == "unset" else "preferred"
    model = "chosen" if selection == "explicit" else "preferred"
    config = make_config(tmp_path).model_copy(update={"default_model": default})
    app = create_app(config)
    worker = create_tts_app(FakeEngine(), model)
    app.state.client._transport = httpx.ASGITransport(app=worker)
    requests: list[httpx.Request] = []

    async def capture(request: httpx.Request) -> None:
        requests.append(request)

    app.state.client.event_hooks["request"] = [capture]
    payload = {"input": "Hello", "voice": "alloy", "response_format": "wav"}
    if selection == "explicit":
        payload["model"] = model
    elif selection == "null":
        payload["model"] = None
    try:
        with (
            patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock) as ensure,
            patch.object(
                type(app.state.manager), "is_running", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                type(app.state.manager),
                "current_model",
                new_callable=PropertyMock,
                return_value="preferred",
            ),
            patch.object(app.state.manager, "base_url_for", return_value="http://worker"),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy"
            ) as client:
                response = await client.post("/v1/audio/speech", json=payload)
            if selection == "unset":
                assert response.status_code == 400
                ensure.assert_not_awaited()
            else:
                assert response.status_code == 200
                assert response.headers["content-type"] == "audio/wav"
                assert response.content.startswith(b"RIFF")
                ensure.assert_awaited_once_with(model, claim=True)
            upstream = requests[0]
            assert int(upstream.headers["content-length"]) == len(upstream.content)
            if selection in {"explicit", "unset"}:
                assert upstream.content == response.request.content
            else:
                import json

                assert json.loads(upstream.content) == {**payload, "model": model}
            assert app.state.stats.active_requests == 0
    finally:
        await app.state.client.aclose()
