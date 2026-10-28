"""Tests for observe integration in the proxy."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import drove.proxy as proxy_module
from drove.config import Config
from drove.observe import (
    MAX_BODY_BYTES,
    MAX_RECORD_BYTES,
    REDACTED,
    ObserveContext,
    ObserveRecord,
    list_records,
)
from drove.proxy import create_app
from drove.stats import ProxyStats


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


async def aiter(items: list[bytes]):  # type: ignore[return]
    for item in items:
        yield item


def test_observe_enabled_writes_log(tmp_path: Path) -> None:
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


def test_observe_disabled_writes_nothing(tmp_path: Path) -> None:
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


def test_observe_record_contains_request_body(tmp_path: Path) -> None:
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


# ── #122: redaction, caps, off-loop writes, retention, bounded buffering ─────


def _setup(tmp_path: Path, observe: bool = True, **overrides: object):  # type: ignore[no-untyped-def]
    config = make_config(tmp_path, observe=observe)
    if overrides:
        config = config.model_copy(update=overrides)
    (config.models_dir / "testmodel").mkdir(parents=True, exist_ok=True)
    (config.models_dir / "testmodel" / "testmodel.gguf").write_bytes(b"")
    app = create_app(config)

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        pass

    return config, app, fake_ensure_running


def _fake_response(chunks: list[bytes], headers: dict[str, str] | None = None) -> MagicMock:
    fake = MagicMock(spec=httpx.Response)
    fake.status_code = 200
    fake.headers = httpx.Headers(headers or {"content-type": "application/json"})
    fake.aiter_raw = lambda: aiter(chunks)
    return fake


def test_credential_headers_are_redacted_in_saved_record(tmp_path: Path) -> None:
    config, app, fake_ensure_running = _setup(tmp_path)
    secrets = {
        "AUTHORIZATION": "Bearer sk-request-secret",
        "Cookie": "session=cookie-secret",
        "x-API-key": "api-key-secret",
    }
    fake = _fake_response(
        [b'{"choices": []}'],
        {
            "content-type": "application/json",
            "Set-Cookie": "session=set-cookie-secret",
            "X-Api-Key": "response-key-secret",
            "x-request-id": "req-1",
        },
    )
    send = AsyncMock(return_value=fake)
    with (
        patch.object(app.state.manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", send),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "testmodel", "messages": []},
                headers=secrets,
            )

    assert resp.status_code == 200
    # Forwarding is untouched: the backend still receives the real credential.
    forwarded = send.call_args.args[0]
    assert forwarded.headers["authorization"] == "Bearer sk-request-secret"

    [(path, record)] = list_records(config.observe_dir)
    text = path.read_text()
    for secret in (*secrets.values(), "set-cookie-secret", "response-key-secret"):
        assert secret not in text
    lowered = {k.lower(): v for k, v in record.request_headers.items()}
    assert lowered["authorization"] == REDACTED
    assert lowered["cookie"] == REDACTED
    assert lowered["x-api-key"] == REDACTED
    resp_lowered = {k.lower(): v for k, v in record.response_headers.items()}
    assert resp_lowered["set-cookie"] == REDACTED
    assert resp_lowered["x-api-key"] == REDACTED
    assert resp_lowered["x-request-id"] == "req-1"


def test_binary_multipart_upload_is_stored_as_placeholder(tmp_path: Path) -> None:
    config, app, fake_ensure_running = _setup(tmp_path)
    audio = bytes(range(256)) * 8192  # 2 MiB of binary
    fake = _fake_response([b'{"text": "hi"}'])
    send = AsyncMock(return_value=fake)
    with (
        patch.object(app.state.manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", send),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/audio/transcriptions",
                data={"model": "testmodel"},
                files={"file": ("a.wav", audio, "audio/wav")},
            )

    assert resp.status_code == 200
    assert audio in send.call_args.args[0].content  # forwarded byte-for-byte
    [(path, record)] = list_records(config.observe_dir)
    assert path.stat().st_size < MAX_RECORD_BYTES
    assert record.request_body is not None
    assert record.request_body.startswith("[binary body omitted:")
    assert "multipart/form-data" in record.request_body
    assert audio[:64].hex() not in path.read_text()
    assert record.response_body_raw == '{"text": "hi"}'


def test_record_is_built_and_written_off_the_event_loop(tmp_path: Path) -> None:
    config, app, _ = _setup(tmp_path)
    threads: dict[str, int] = {}

    async def fake_ensure_running(model_name: str, *, claim: bool = False) -> None:
        threads["loop"] = threading.get_ident()

    real_to_record = ObserveContext.to_record
    real_save = proxy_module.save_record

    def spy_to_record(self: ObserveContext) -> ObserveRecord:
        threads["to_record"] = threading.get_ident()
        return real_to_record(self)

    def spy_save(observe_dir: Path, record: ObserveRecord) -> Path:
        threads["save"] = threading.get_ident()
        return real_save(observe_dir, record)

    with (
        patch.object(app.state.manager, "ensure_running", side_effect=fake_ensure_running),
        patch("httpx.AsyncClient.send", AsyncMock(return_value=_fake_response([b"{}"]))),
        patch.object(ObserveContext, "to_record", spy_to_record),
        patch.object(proxy_module, "save_record", spy_save),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/v1/chat/completions", json={"model": "testmodel", "messages": []})

    assert set(threads) == {"loop", "to_record", "save"}
    assert threads["to_record"] != threads["loop"]
    assert threads["save"] != threads["loop"]
    [(path, _)] = list_records(config.observe_dir)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _aged_record(observe_dir: Path, name: str, age_days: float) -> Path:
    path = observe_dir / "testmodel" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


def test_startup_prunes_expired_records_with_default_30_days(tmp_path: Path) -> None:
    config, app, _ = _setup(tmp_path)
    assert config.observe_max_age_days == 30
    old = _aged_record(config.observe_dir, "old", 31)
    new = _aged_record(config.observe_dir, "new", 29)

    with TestClient(app):
        assert not old.exists()
        assert new.exists()


def test_startup_keeps_records_when_disabled(tmp_path: Path) -> None:
    for observe, days in ((True, 0), (False, 30)):
        root = tmp_path / f"{observe}-{days}"
        root.mkdir()
        config, app, _ = _setup(root, observe=observe, observe_max_age_days=days)
        old = _aged_record(config.observe_dir, "old", 365)
        with TestClient(app):
            pass
        assert old.exists()


def test_failing_retention_pass_leaves_the_proxy_serving(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, app, _ = _setup(tmp_path)
    boom = MagicMock(side_effect=PermissionError("observe dir unreadable"))

    with (
        patch.object(proxy_module, "prune_records", boom),
        caplog.at_level("WARNING", logger="drove.proxy"),
        TestClient(app) as client,
    ):
        resp = client.get("/v1/models")

    boom.assert_called_once()
    assert resp.status_code == 200
    assert "Observe retention pass failed: observe dir unreadable" in caplog.text


async def test_periodic_pass_uses_current_config(tmp_path: Path) -> None:
    config, app, _ = _setup(tmp_path, observe_max_age_days=0)
    old = _aged_record(config.observe_dir, "old", 10)

    await proxy_module._prune_observe_once(app)
    assert old.exists()

    app.state.config = config.model_copy(update={"observe_max_age_days": 7})
    await proxy_module._prune_observe_once(app)
    assert not old.exists()


async def test_retention_loop_runs_passes_on_its_interval(tmp_path: Path) -> None:
    _, app, _ = _setup(tmp_path)
    passes = 0
    done = asyncio.Event()

    async def fake_prune(_app: object) -> None:
        nonlocal passes
        passes += 1
        if passes == 3:
            done.set()

    with (
        patch.object(proxy_module, "_OBSERVE_PRUNE_SECONDS", 0),
        patch.object(proxy_module, "_prune_observe_once", fake_prune),
    ):
        task = asyncio.create_task(proxy_module._observe_retention(app))
        await asyncio.wait_for(done.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert passes >= 3


def _usage_sse(n_chunks: int, chunk_text: str) -> list[bytes]:
    delta = json.dumps({"choices": [{"delta": {"content": chunk_text}}]})
    chunks = [f"data: {delta}\n\n".encode() for _ in range(n_chunks)]
    usage = json.dumps({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    chunks.append(f"data: {usage}\n\ndata: [DONE]\n\n".encode())
    return chunks


async def _drain(stream: AsyncIterator[bytes]) -> bytes:
    out = bytearray()
    async for chunk in stream:
        out += chunk
    return bytes(out)


async def test_observe_off_keeps_only_a_bounded_usage_tail(tmp_path: Path) -> None:
    chunks = _usage_sse(5000, "x" * 1000)  # ~5 MiB of SSE
    seen: list[int] = []
    real_record_usage = proxy_module._record_usage

    def spy(body: bytes, stats: ProxyStats, duration: float = 0.0):  # type: ignore[no-untyped-def]
        seen.append(len(body))
        return real_record_usage(body, stats, duration)

    stats = ProxyStats()
    manager = MagicMock()
    with patch.object(proxy_module, "_record_usage", spy):
        forwarded = await _drain(
            proxy_module._counting_stream(aiter(chunks), stats, manager, "testmodel")
        )

    assert forwarded == b"".join(chunks)
    assert seen and seen[0] <= proxy_module._USAGE_TAIL_BYTES
    assert stats.tokens_prompt == 7
    assert stats.tokens_completion == 3


async def test_observe_on_captures_bounded_prefix(tmp_path: Path) -> None:
    chunks = _usage_sse(5000, "y" * 1000)
    total = sum(len(c) for c in chunks)
    stats = ProxyStats()
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={},
        request_body=b"{}",
        response_headers={"content-type": "text/event-stream"},
    )
    writer = MagicMock()
    await _drain(
        proxy_module._counting_stream(
            aiter(chunks),
            stats,
            MagicMock(),
            "testmodel",
            ctx,
            tmp_path,
            endpoint="v1/chat/completions",
            observe_writer=writer,
        )
    )

    assert len(ctx.response_body) <= MAX_BODY_BYTES
    assert ctx.response_body_size == total
    assert ctx.tokens_completion == 3
    writer.submit.assert_called_once_with(ctx, tmp_path)


async def test_audio_responses_are_not_buffered(tmp_path: Path) -> None:
    chunks = [b"\x00\x01" * 32768 for _ in range(64)]  # 4 MiB of audio
    stats = ProxyStats()
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/audio/speech",
        method="POST",
        request_headers={},
        request_body=b"{}",
        response_headers={"content-type": "audio/wav"},
    )
    with patch.object(proxy_module, "_record_usage") as record_usage:
        forwarded = await _drain(
            proxy_module._counting_stream(
                aiter(chunks),
                stats,
                MagicMock(),
                "testmodel",
                ctx,
                tmp_path,
                endpoint="v1/audio/speech",
                audio_duration="1.5",
                observe_writer=MagicMock(),
            )
        )

    assert forwarded == b"".join(chunks)
    record_usage.assert_not_called()
    assert ctx.response_body == b""
    assert ctx.response_body_size == len(forwarded)
    assert stats.speech.requests == 1
    assert stats.speech.audio_seconds == 1.5
    assert "audio/wav" in ctx.to_record().response_body


async def test_observe_writer_drops_when_full_and_drains(tmp_path: Path) -> None:
    release = threading.Event()
    written: list[str] = []

    def slow_write(ctx: ObserveContext, observe_dir: Path) -> None:
        release.wait(timeout=5)
        written.append(ctx.endpoint)

    writer = proxy_module._ObserveWriter(limit=2)
    ctxs = [
        ObserveContext(
            timestamp="2026-04-08T12:00:00",
            model="testmodel",
            endpoint=f"e{i}",
            method="POST",
            request_headers={},
            request_body=b"",
        )
        for i in range(3)
    ]
    with patch.object(proxy_module, "_write_observe_record", slow_write):
        for ctx in ctxs:
            writer.submit(ctx, tmp_path)
        heartbeat = await asyncio.wait_for(asyncio.sleep(0, result="alive"), timeout=1)
        release.set()
        await writer.drain()

    assert heartbeat == "alive"
    assert sorted(written) == ["e0", "e1"]


@pytest.mark.parametrize("observe", [False, True])
async def test_large_non_streaming_json_keeps_usage(tmp_path: Path, observe: bool) -> None:
    body = json.dumps(
        {
            "choices": [{"message": {"content": "z" * (proxy_module._USAGE_TAIL_BYTES + 500_000)}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "timings": {"prompt_n": 7, "predicted_n": 3, "predicted_per_second": 30.0},
        }
    ).encode()
    # Chunk boundaries fall inside the usage and timings objects.
    cuts = sorted({body.index(b'"usage"') + 9, body.index(b'"timings"') + 12, len(body)})
    chunks, start = [], 0
    for offset in range(65536, cuts[0], 65536):
        chunks.append(body[start:offset])
        start = offset
    for cut in cuts:
        chunks.append(body[start:cut])
        start = cut
    assert b"".join(chunks) == body

    stats = ProxyStats()
    ctx = (
        ObserveContext(
            timestamp="2026-04-08T12:00:00",
            model="testmodel",
            endpoint="v1/chat/completions",
            method="POST",
            request_headers={},
            request_body=b"{}",
            response_headers={"content-type": "application/json"},
        )
        if observe
        else None
    )
    forwarded = await _drain(
        proxy_module._counting_stream(
            aiter(chunks),
            stats,
            MagicMock(),
            "testmodel",
            ctx,
            tmp_path,
            endpoint="v1/chat/completions",
            observe_writer=MagicMock(),
        )
    )

    assert forwarded == body
    assert stats.tokens_prompt == 7
    assert stats.tokens_completion == 3
    assert stats.avg_tokens_per_second == pytest.approx(30.0)
    if ctx is not None:
        assert ctx.tokens_completion == 3
