"""Tests for the FastAPI proxy app."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
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
            release.assert_called_once_with("kitten", ensure.return_value)
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
    (config.models_dir / "vision.toml").write_text('mmproj = "projector.gguf"\nctx_size = 8192\n')
    for name, extension in (("speech", ".onnx"), ("voice", ".tts")):
        folder = config.models_dir / name
        folder.mkdir()
        (folder / (name + extension)).write_bytes(b"data")
    (config.models_dir / "override.gguf").write_bytes(b"data")
    (config.models_dir / "override.toml").write_text('backend = "asr"\n')
    (config.models_dir / "broken.gguf").write_bytes(b"data")
    (config.models_dir / "broken.toml").write_text('ctx_size = "bad"\n')
    from unittest.mock import AsyncMock, PropertyMock

    with (
        patch(
            "drove.server_manager.ServerManager.loaded_models",
            new_callable=PropertyMock,
            return_value=["chat"],
        ),
        patch(
            "drove.server_manager.ServerManager.ensure_running", new_callable=AsyncMock
        ) as ensure_running,
        TestClient(create_app(config)) as client,
    ):
        response = client.get("/v1/models")
    assert response.status_code == 200
    ensure_running.assert_not_awaited()
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
        "context_length": None,
    }
    assert "vision" in rows["vision"]["drove"]["capabilities"]
    assert rows["vision"]["drove"]["context_length"] == 8192
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
            slot = started.return_value
        else:
            ensure.assert_awaited_once_with(expected, claim=True)
            started.assert_not_called()
            slot = ensure.return_value
        finished.assert_called_once_with(expected, slot)
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


def _write_config(config: Config, path: Path) -> None:
    import tomli_w

    path.write_text(tomli_w.dumps(config.model_dump(mode="json", exclude_none=True)))


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
        _write_config(config.model_copy(update={"default_model": "second"}), path)
        _reload_config(app, path)
        assert client.post("/v1/chat/completions", json={}).status_code == 404
        ensure.assert_awaited_with("second", claim=True)
        _write_config(config.model_copy(update={"default_model": None}), path)
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
            if selection == "null":
                # An explicit null is a non-string model, not an omitted one (#110).
                assert response.status_code == 400
                ensure.assert_not_awaited()
                assert requests == []
                assert app.state.stats.active_requests == 0
                return
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


# ── model name validation (#110) ─────────────────────────────────────────────

_INVALID_JSON_MODELS = ["/", "../x", "../../some/dir", "/abs", "", "./m", "m/", "a\\b", "a/b/c"]


def _forbid_disk_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("model lookup touched the disk")

    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)


@pytest.mark.parametrize(
    "model", [*_INVALID_JSON_MODELS, None, 1, 123, 1.5, True, [], ["m"], {"name": "m"}]
)
@pytest.mark.parametrize("default", [None, "preferred"])
def test_proxy_rejects_invalid_json_model_with_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: object, default: str | None
) -> None:
    config = make_config(tmp_path).model_copy(update={"default_model": default})
    app = create_app(config)
    manager = app.state.manager
    with (
        patch.object(manager, "ensure_running", new_callable=AsyncMock) as ensure,
        patch.object(type(manager), "is_running", new_callable=PropertyMock, return_value=True),
        patch.object(
            type(manager), "current_model", new_callable=PropertyMock, return_value="loaded"
        ),
        patch.object(app.state.client, "send", new_callable=AsyncMock) as send,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        _forbid_disk_walk(monkeypatch)
        resp = client.post("/v1/chat/completions", json={"model": model, "messages": []})
        status = client.get("/status").json()
    assert resp.status_code == 400, resp.text
    assert "model" in resp.json()["detail"].lower()
    ensure.assert_not_awaited()
    send.assert_not_awaited()
    assert status["requests"]["active"] == 0
    assert status["requests"]["errors"] == 1


@pytest.mark.parametrize("model", ["", "../x", "/abs", "a//b"])
def test_proxy_rejects_invalid_multipart_model_with_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    app = create_app(make_config(tmp_path).model_copy(update={"default_model": "preferred"}))
    with (
        patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock) as ensure,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        _forbid_disk_walk(monkeypatch)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"RIFFxxxx", "audio/wav")},
            data={"model": model},
        )
    assert resp.status_code == 400, resp.text
    ensure.assert_not_awaited()


def test_proxy_rejects_file_valued_multipart_model_with_400(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path).model_copy(update={"default_model": "preferred"}))
    with (
        patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock) as ensure,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={
                "file": ("audio.wav", b"RIFFxxxx", "audio/wav"),
                "model": ("model.txt", b"preferred", "text/plain"),
            },
        )
    assert resp.status_code == 400, resp.text
    ensure.assert_not_awaited()


def test_proxy_rejects_invalid_default_model_with_400(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path).model_copy(update={"default_model": "../escape"}))
    with (
        patch.object(app.state.manager, "ensure_running", new_callable=AsyncMock) as ensure,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400, resp.text
    ensure.assert_not_awaited()


def test_proxy_treats_traversal_to_a_real_gguf_as_invalid(tmp_path: Path) -> None:
    """Even when the traversal target holds a real model, nothing is loaded."""
    config = make_config(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "model.gguf").write_bytes(b"GGUF")
    app = create_app(config)
    with (
        patch.object(app.state.manager, "_start", new_callable=AsyncMock) as start,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        resp = client.post("/v1/chat/completions", json={"model": "../elsewhere", "messages": []})
    assert resp.status_code == 400
    start.assert_not_awaited()


def test_models_endpoint_hides_names_the_api_would_reject(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for name in ("good", "bad name"):
        (config.models_dir / name).mkdir()
        (config.models_dir / name / "model.gguf").write_bytes(b"")
    with TestClient(create_app(config)) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["good"]


# ── Startup failures: deliberate statuses, balanced accounting, private stderr ─────

_LIST_HINT = "Run 'drove models list' to see available models."
_SENTINEL = "SECRET-STDERR-/home/user/.cache/model.gguf"


def _toml_error() -> Exception:
    import tomllib

    try:
        tomllib.loads("ctx_size = = 1")
    except tomllib.TOMLDecodeError as e:
        return e
    raise AssertionError("expected a TOML error")


def _validation_error() -> Exception:
    from pydantic import ValidationError

    from drove.model_config import ModelConfig

    try:
        ModelConfig(ctx_size="bad")  # type: ignore[arg-type]
    except ValidationError as e:
        return e
    raise AssertionError("expected a validation error")


class _OddError(Exception):
    pass


def _startup_cases() -> list[Any]:
    from drove.server_manager import BackendUnavailableError

    config_detail = "invalid model config for 'm'"
    failed = "model failed to start; see server logs"
    return [
        (FileNotFoundError(_SENTINEL), 404, "Model 'm' not found. " + _LIST_HINT),
        (TypeError(_SENTINEL), 500, "model startup failed; see server logs"),
        (ValueError(_SENTINEL), 500, config_detail),
        (_toml_error(), 500, config_detail),
        (_validation_error(), 500, config_detail),
        (OSError(8, _SENTINEL), 503, failed),
        (RuntimeError(_SENTINEL), 503, failed),
        (TimeoutError(_SENTINEL), 503, failed),
        (BackendUnavailableError(_SENTINEL), 503, "backend unavailable; see server logs"),
        (_OddError(_SENTINEL), 500, "model startup failed; see server logs"),
    ]


@pytest.mark.parametrize(("error", "status", "detail"), _startup_cases())
@pytest.mark.parametrize("baseline", [0, 2])
def test_startup_failure_maps_to_deliberate_status_and_balances_stats(
    tmp_path: Path, error: Exception, status: int, detail: str, baseline: int
) -> None:
    app = create_app(make_config(tmp_path))
    stats = app.state.stats
    for _ in range(baseline):
        stats.request_started()
    errors_before = stats.error_count
    with (
        patch.object(app.state.manager, "ensure_running", side_effect=error),
        patch.object(app.state.client, "send", new_callable=AsyncMock) as send,
        TestClient(app) as client,
    ):
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        reported = client.get("/status").json()["requests"]
    assert resp.status_code == status
    assert resp.json() == {"detail": detail}
    assert _SENTINEL not in resp.text
    send.assert_not_awaited()
    assert reported["active"] == baseline
    assert stats.error_count == errors_before + 1


def test_startup_failure_leaves_other_models_claims_alone(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    manager = app.state.manager
    other = _live_instance(tmp_path, "other")
    other.active_requests = 1
    manager._instances["other"] = other
    with (
        patch.object(manager, "ensure_running", side_effect=_OddError("boom")),
        TestClient(app) as client,
    ):
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert resp.status_code == 500
        assert other.active_requests == 1
        manager._instances.clear()  # keep the lifespan shutdown away from the fake process


def test_unknown_model_still_returns_404_with_balanced_stats(tmp_path: Path) -> None:
    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "ghost", "messages": []})
        reported = client.get("/status").json()["requests"]
    assert resp.status_code == 404
    assert reported["active"] == 0
    assert reported["errors"] == 1
    # The store's own message names models_dir; clients get only the model name.
    assert resp.json() == {"detail": "Model 'ghost' not found. " + _LIST_HINT}
    assert str(app.state.config.models_dir) not in resp.text


def _real_manager_app(tmp_path: Path, **updates: Any) -> tuple[Any, Path]:
    config = make_config(tmp_path).model_copy(update=updates)
    model_dir = config.models_dir / "m"
    model_dir.mkdir()
    model_path = model_dir / "m.gguf"
    model_path.write_bytes(b"")
    return create_app(config), model_path


def _post_chat(app: Any) -> tuple[Any, dict[str, int]]:
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        return resp, client.get("/status").json()["requests"]


def _assert_clean_failure(app: Any, requests: dict[str, int], errors_before: int = 0) -> None:
    assert requests["active"] == 0
    assert requests["errors"] == errors_before + 1
    assert not app.state.manager._instances
    assert not app.state.manager._idle_tasks


def test_missing_llama_server_binary_returns_503(tmp_path: Path) -> None:
    app, _ = _real_manager_app(tmp_path)
    with patch("drove.server_manager.shutil.which", return_value=None):
        resp, requests = _post_chat(app)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "backend unavailable; see server logs"}
    _assert_clean_failure(app, requests)


@pytest.mark.parametrize("error", [OSError(8, "Exec format error"), FileNotFoundError("gone")])
def test_exec_failure_returns_503(tmp_path: Path, error: OSError) -> None:
    app, _ = _real_manager_app(tmp_path)
    with (
        patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
        patch(
            "drove.server_manager.asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=error),
        ),
    ):
        resp, requests = _post_chat(app)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "backend unavailable; see server logs"}
    _assert_clean_failure(app, requests)


def _live_instance(tmp_path: Path, name: str) -> Any:
    from drove.server_manager import _ModelInstance

    process = MagicMock()
    process.returncode = None
    process.pid = 4242
    return _ModelInstance(name, process, 9999, tmp_path / f"{name}.gguf", (0.0, 0.0))


def _healthy_loaded_instance(app: Any, tmp_path: Path) -> Any:
    healthy = _live_instance(tmp_path, "healthy")
    app.state.manager._instances["healthy"] = healthy
    return healthy


@pytest.mark.parametrize(
    "sidecar",
    ['backend = "nope"\n', "ctx_size = = 1\n", 'ctx_size = "bad"\n'],
    ids=["unknown-backend", "malformed-toml", "wrong-type"],
)
def test_broken_sidecar_returns_500_without_evicting_loaded_model(
    tmp_path: Path, sidecar: str
) -> None:
    app, model_path = _real_manager_app(tmp_path, max_loaded_models=1)
    model_path.with_suffix(".toml").write_text(sidecar)
    manager = app.state.manager
    healthy = _healthy_loaded_instance(app, tmp_path)
    with (
        patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
        patch.object(manager, "_stop_instance", new_callable=AsyncMock) as stop,
        patch(
            "drove.server_manager.asyncio.create_subprocess_exec", new_callable=AsyncMock
        ) as spawn,
        TestClient(app) as client,
    ):
        resp = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        requests = client.get("/status").json()["requests"]
        stop.assert_not_awaited()  # checked before the lifespan shutdown stops everything
        spawn.assert_not_awaited()
        assert manager._instances == {"healthy": healthy}
        manager._instances.clear()
    assert resp.status_code == 500
    assert resp.json() == {"detail": "invalid model config for 'm'"}
    assert requests["active"] == 0
    assert requests["errors"] == 1


def _fake_backend(returncode: int | None, stderr: bytes) -> MagicMock:
    process = MagicMock()
    process.returncode = returncode
    process.pid = 4242
    process.wait = AsyncMock(return_value=0)
    process.stderr.read = AsyncMock(side_effect=[stderr, b""])

    def exit_on_signal(_sig: int) -> None:
        process.returncode = 0

    process.send_signal.side_effect = exit_on_signal
    return process


@pytest.mark.parametrize("mode", ["exit", "timeout"])
def test_health_failure_returns_503_without_stderr_but_logs_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, mode: str
) -> None:
    app, _ = _real_manager_app(tmp_path, startup_timeout_seconds=0.3)
    process = _fake_backend(1 if mode == "exit" else None, _SENTINEL.encode())
    with (
        patch("drove.server_manager.shutil.which", return_value="/usr/bin/llama-server"),
        patch(
            "drove.server_manager.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ),
        caplog.at_level("ERROR", logger="drove"),
    ):
        resp, requests = _post_chat(app)
    assert resp.status_code == 503
    assert resp.json() == {"detail": "model failed to start; see server logs"}
    assert _SENTINEL not in resp.text
    assert _SENTINEL in caplog.text
    _assert_clean_failure(app, requests)


# --- Client disconnects before the response starts (#112) -------------------


class _Connection:
    """A raw ASGI client: sends the whole body, then disconnects when told to."""

    def __init__(self, body: bytes) -> None:
        self._pending: list[dict[str, object]] = [
            {"type": "http.request", "body": body, "more_body": False}
        ]
        self.gone = asyncio.Event()
        self.body_sent = asyncio.Event()
        self.status: int | None = None

    async def receive(self) -> dict[str, object]:
        if self._pending:
            return self._pending.pop(0)
        await self.gone.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, object]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]  # type: ignore[assignment]
        elif message["type"] == "http.response.body" and message.get("body"):
            self.body_sent.set()


def _scope(path: str, content_type: str, body: bytes) -> dict[str, object]:
    return {
        "type": "http",
        # 2.3: StreamingResponse watches receive() for http.disconnect, as under uvicorn.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/{path}",
        "raw_path": f"/{path}".encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8080),
    }


def _multipart_body() -> tuple[bytes, str]:
    request = httpx.Request(
        "POST",
        "http://drove/v1/audio/transcriptions",
        data={"model": "m"},
        files={"file": ("a.wav", b"RIFF....WAVE", "audio/wav")},
    )
    request.read()
    return request.content, request.headers["content-type"]


_ENDPOINTS = {
    "chat-stream": (
        "v1/chat/completions",
        "application/json",
        b'{"model": "m", "stream": true, "messages": []}',
    ),
    "chat-buffered": (
        "v1/chat/completions",
        "application/json",
        b'{"model": "m", "stream": false, "messages": []}',
    ),
    "asr": ("v1/audio/transcriptions", *reversed(_multipart_body())),
    "tts": ("v1/audio/speech", "application/json", b'{"model": "m", "input": "hi"}'),
}


class _Abort(BaseException):
    """A BaseException that is neither an Exception nor a cancellation."""


class _Harness:
    """A loaded model 'm' that already serves one other request, plus fakes.

    The other request (instance count 1, global active count 1) is a sentinel:
    a double release would drop it to 0 instead of hiding behind the floor at 0.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from drove.server_manager import _ModelInstance

        self.app = create_app(make_config(tmp_path))
        self.manager = self.app.state.manager
        self.stats = self.app.state.stats
        self.instance = _ModelInstance(
            "m", MagicMock(returncode=None), 9999, tmp_path / "m.gguf", (0.0, 0.0)
        )
        self.instance.active_requests = 1
        self.manager._instances["m"] = self.instance
        self.stats.request_started()

        self.finished = MagicMock(wraps=self.manager.request_finished)
        monkeypatch.setattr(self.manager, "request_finished", self.finished)

        self.starting = asyncio.Event()
        self.started = asyncio.Event()
        self.started.set()

        async def ensure_running(model_name: str, *, claim: bool = False) -> object | None:
            self.starting.set()
            await self.started.wait()
            return self.manager._claim_slot(model_name)

        monkeypatch.setattr(self.manager, "ensure_running", ensure_running)

        self.sending = asyncio.Event()
        self.send_cancelled = asyncio.Event()
        self.answer = asyncio.Event()  # never set by the disconnect tests
        self.send_error: BaseException | None = None
        self.upstream: list[httpx.Request] = []
        self.response = MagicMock(spec=httpx.Response)
        self.response.status_code = 200
        self.response.headers = httpx.Headers({"content-type": "application/json"})
        self.response.aiter_raw = lambda: aiter([b'{"ok": true}'])
        self.response.aclose = AsyncMock()

        async def send(request: httpx.Request, *, stream: bool = False) -> httpx.Response:
            self.upstream.append(request)
            self.sending.set()
            if self.send_error is not None:
                raise self.send_error
            try:
                await self.answer.wait()
            except asyncio.CancelledError:
                self.send_cancelled.set()
                raise
            return self.response

        self.send = AsyncMock(side_effect=send)
        monkeypatch.setattr(self.app.state.client, "send", self.send)

    def call(self, endpoint: str) -> tuple[asyncio.Task[None], _Connection]:
        path, content_type, body = _ENDPOINTS[endpoint]
        conn = _Connection(body)
        task = asyncio.create_task(
            self.app(_scope(path, content_type, body), conn.receive, conn.send)
        )
        return task, conn

    def assert_released_once(self) -> None:
        self.finished.assert_called_once_with("m", self.instance)
        assert self.instance.active_requests == 1
        assert self.stats.active_requests == 1


@pytest.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    h = _Harness(tmp_path, monkeypatch)
    yield h
    h.manager._instances.clear()
    await h.app.state.client.aclose()


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
async def test_disconnect_during_cold_start_is_never_forwarded(
    harness: _Harness, endpoint: str
) -> None:
    harness.started.clear()
    task, conn = harness.call(endpoint)
    await asyncio.wait_for(harness.starting.wait(), 5)
    conn.gone.set()
    harness.started.set()
    await asyncio.wait_for(task, 5)

    harness.send.assert_not_awaited()
    harness.assert_released_once()
    assert harness.stats.error_count == 0
    assert conn.status == 499


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
async def test_disconnect_during_upstream_send_cancels_it(harness: _Harness, endpoint: str) -> None:
    task, conn = harness.call(endpoint)
    await asyncio.wait_for(harness.sending.wait(), 5)
    conn.gone.set()
    # The fake upstream never answers: the handler only finishes by cancelling it.
    await asyncio.wait_for(task, 5)

    assert harness.send_cancelled.is_set()
    assert not harness.answer.is_set()
    harness.assert_released_once()
    assert harness.stats.error_count == 0
    assert conn.status == 499


async def test_mid_stream_disconnect_releases_once(harness: _Harness) -> None:
    more = asyncio.Event()  # never set: the second chunk never arrives

    async def chunks():  # type: ignore[no-untyped-def]
        yield b'data: {"choices": []}\n\n'
        await more.wait()
        yield b"data: [DONE]\n\n"

    harness.response.aiter_raw = chunks
    harness.answer.set()
    task, conn = harness.call("chat-stream")
    await asyncio.wait_for(conn.body_sent.wait(), 5)
    conn.gone.set()
    await asyncio.wait_for(task, 5)

    assert conn.status == 200
    harness.assert_released_once()


async def test_connected_client_still_gets_the_upstream_response(harness: _Harness) -> None:
    harness.answer.set()
    task, conn = harness.call("chat-buffered")
    await asyncio.wait_for(task, 5)

    assert conn.status == 200
    assert conn.body_sent.is_set()
    harness.assert_released_once()


async def test_fallback_model_forwards_a_non_json_body_intact(harness: _Harness) -> None:
    harness.answer.set()
    conn = _Connection(b"plain text body")
    scope = _scope("v1/completions", "text/plain", b"plain text body")
    await asyncio.wait_for(harness.app(scope, conn.receive, conn.send), 5)

    assert conn.status == 200
    assert harness.upstream[0].content == b"plain text body"
    harness.assert_released_once()


async def test_cancel_during_cold_start_settles_without_a_release(harness: _Harness) -> None:
    harness.started.clear()
    task, _ = harness.call("chat-stream")
    await asyncio.wait_for(harness.starting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    harness.finished.assert_not_called()
    assert harness.instance.active_requests == 1
    assert harness.stats.active_requests == 1


async def test_cancel_during_upstream_send_releases_once(harness: _Harness) -> None:
    before = asyncio.all_tasks()
    task, _ = harness.call("chat-stream")
    await asyncio.wait_for(harness.sending.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.send_cancelled.is_set()
    harness.assert_released_once()
    assert harness.stats.error_count == 0
    leftover = [t for t in asyncio.all_tasks() - before if not t.done()]
    assert leftover == []


async def test_base_exception_from_upstream_send_releases_once(harness: _Harness) -> None:
    harness.send_error = _Abort()
    task, _ = harness.call("tts")
    with pytest.raises(_Abort):
        await asyncio.wait_for(task, 5)

    harness.assert_released_once()


async def test_send_race_prefers_disconnect_on_a_tie() -> None:
    from drove.proxy import _send_unless_disconnected

    response = MagicMock(spec=httpx.Response)
    response.aclose = AsyncMock()
    client = MagicMock(spec=httpx.AsyncClient)
    client.send = AsyncMock(return_value=response)

    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    assert await _send_unless_disconnected(client, MagicMock(), receive) is None
    response.aclose.assert_awaited_once()


async def test_send_race_ignores_non_disconnect_messages() -> None:
    from drove.proxy import _send_unless_disconnected

    response = MagicMock(spec=httpx.Response)
    answer = asyncio.Event()

    async def send(request: object, *, stream: bool = False) -> httpx.Response:
        await answer.wait()
        return response

    client = MagicMock(spec=httpx.AsyncClient)
    client.send = AsyncMock(side_effect=send)
    messages = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive() -> dict[str, object]:
        if messages:
            message = messages.pop()
            answer.set()
            return message
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    assert await _send_unless_disconnected(client, MagicMock(), receive) is response


def _race_parts() -> tuple[MagicMock, MagicMock, asyncio.Event, asyncio.Event]:
    """A client whose send answers at once, and a response whose close can be held."""
    closing = asyncio.Event()
    may_close = asyncio.Event()
    response = MagicMock(spec=httpx.Response)

    async def aclose() -> None:
        closing.set()
        await may_close.wait()

    response.aclose = AsyncMock(side_effect=aclose)
    client = MagicMock(spec=httpx.AsyncClient)
    client.send = AsyncMock(return_value=response)
    return client, response, closing, may_close


def _cancel_on_watcher_cleanup(parent: list[asyncio.Task[object]]):  # type: ignore[no-untyped-def]
    """A receive() that blocks, and cancels the handler once the watcher is cancelled.

    That is the moment the send has won and the helper is winding the watcher down.
    """

    async def receive() -> dict[str, object]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            parent[0].cancel()
            raise
        raise AssertionError("unreachable")

    return receive


async def test_send_race_closes_the_response_when_cancelled_during_handoff() -> None:
    from drove.proxy import _send_unless_disconnected

    client, response, closing, may_close = _race_parts()
    may_close.set()
    parent: list[asyncio.Task[object]] = []
    before = asyncio.all_tasks()
    receive = _cancel_on_watcher_cleanup(parent)
    parent.append(asyncio.create_task(_send_unless_disconnected(client, MagicMock(), receive)))
    with pytest.raises(asyncio.CancelledError):
        await parent[0]

    await asyncio.wait_for(closing.wait(), 5)
    response.aclose.assert_awaited_once()
    assert [t for t in asyncio.all_tasks() - before if not t.done()] == []


async def test_send_race_finishes_closing_when_cancelled_again() -> None:
    from drove.proxy import _send_unless_disconnected

    client, response, closing, may_close = _race_parts()
    parent: list[asyncio.Task[object]] = []
    receive = _cancel_on_watcher_cleanup(parent)
    parent.append(asyncio.create_task(_send_unless_disconnected(client, MagicMock(), receive)))
    await asyncio.wait_for(closing.wait(), 5)
    parent[0].cancel()  # a second cancellation must not abandon the close
    with pytest.raises(asyncio.CancelledError):
        await parent[0]

    from drove.proxy import _background_tasks

    assert _background_tasks  # the shielded cleanup is still closing the response
    may_close.set()
    await asyncio.wait_for(_drain_background_tasks(), 5)
    response.aclose.assert_awaited_once()


async def _drain_background_tasks() -> None:
    from drove.proxy import _background_tasks

    while _background_tasks:
        await asyncio.gather(*_background_tasks)


def test_unservable_and_incomplete_downloads_are_hidden_and_never_started(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    models = config.models_dir
    (models / "ready").mkdir()
    (models / "ready" / "model.gguf").write_bytes(b"GGUF")
    (models / "org" / "cache-only" / ".cache" / "huggingface").mkdir(parents=True)
    (models / "org" / "cache-only" / ".cache" / "huggingface" / "x.incomplete").write_bytes(b"")
    (models / "raw").mkdir()
    (models / "raw" / "model.safetensors").write_bytes(b"weights")
    (models / "split").mkdir()
    (models / "split" / "split-00001-of-00002.gguf").write_bytes(b"GGUF")
    app = create_app(config)
    with (
        patch.object(app.state.manager, "_start", new_callable=AsyncMock) as start,
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        resp = client.post("/v1/chat/completions", json={"model": "split", "messages": []})
    assert ids == ["ready"]
    assert resp.status_code == 404
    assert "incomplete" in resp.json()["detail"]
    start.assert_not_awaited()
