"""Tests for the built-in ASR worker."""

from __future__ import annotations

import io
import subprocess
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from drove.workers.asr import TARGET_RATE, create_asr_app, normalize_audio


class FakeEngine:
    def __init__(self, text: str = "hello world") -> None:
        self.text = text
        self.paths: list[str] = []

    def recognize(self, wav_path: str) -> str:
        self.paths.append(wav_path)
        return self.text


def make_wav(rate: int = TARGET_RATE, channels: int = 1, seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    n_frames = int(rate * seconds)
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * n_frames * channels)
    return buf.getvalue()


def test_health() -> None:
    app = create_asr_app(FakeEngine(), model_name="nemo-parakeet-tdt-0.6b-v3")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model"] == "nemo-parakeet-tdt-0.6b-v3"


def test_transcribe_json_default() -> None:
    engine = FakeEngine("the quick brown fox")
    app = create_asr_app(engine)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", make_wav(), "audio/wav")},
            data={"model": "parakeet"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"text": "the quick brown fox"}
    assert len(engine.paths) == 1


def test_transcribe_text_format() -> None:
    app = create_asr_app(FakeEngine("plain text result"))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", make_wav(), "audio/wav")},
            data={"response_format": "text"},
        )
    assert resp.status_code == 200
    assert resp.text == "plain text result"


def test_transcribe_verbose_json_includes_duration() -> None:
    app = create_asr_app(FakeEngine())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", make_wav(seconds=1.0), "audio/wav")},
            data={"response_format": "verbose_json"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "transcribe"
    assert body["text"] == "hello world"
    assert body["duration"] == pytest.approx(1.0, abs=0.01)


def test_transcribe_unsupported_format_returns_400() -> None:
    app = create_asr_app(FakeEngine())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", make_wav(), "audio/wav")},
            data={"response_format": "srt"},
        )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["detail"]


def test_transcribe_empty_file_returns_400() -> None:
    app = create_asr_app(FakeEngine())
    with TestClient(app) as client:
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", b"", "audio/wav")},
        )
    assert resp.status_code == 400


def test_normalize_audio_conforming_wav_passthrough(tmp_path: Path) -> None:
    data = make_wav(rate=TARGET_RATE, channels=1, seconds=0.25)
    out, duration = normalize_audio(data, tmp_path)
    assert out.read_bytes() == data
    assert duration == pytest.approx(0.25, abs=0.01)


def test_normalize_audio_resamples_stereo_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("drove.workers.asr.shutil.which", lambda _: None)
    data = make_wav(rate=44_100, channels=2, seconds=0.25)
    out, duration = normalize_audio(data, tmp_path)
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == TARGET_RATE
        assert w.getsampwidth() == 2
    assert duration == pytest.approx(0.25, abs=0.02)


def test_normalize_audio_non_wav_without_ffmpeg_raises_415(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("drove.workers.asr.shutil.which", lambda _: None)
    with pytest.raises(HTTPException) as excinfo:
        normalize_audio(b"\xffnot audio at all", tmp_path)
    assert excinfo.value.status_code == 415


def test_main_wires_arguments_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    from drove.workers import asr as asr_module

    engine_args: dict[str, tuple[str, Path, str | None]] = {}

    class FakeEngineCls:
        def __init__(self, model_type: str, model_dir: Path, quantization: str | None) -> None:
            engine_args["init"] = (model_type, model_dir, quantization)

        def recognize(self, wav_path: str) -> str:
            return ""

    monkeypatch.setattr(asr_module, "OnnxAsrEngine", FakeEngineCls)
    runs: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: runs.update({"app": app, **kw}))

    asr_module.main(
        [
            "--model-dir",
            str(tmp_path),
            "--model-type",
            "nemo-parakeet-tdt-0.6b-v3",
            "--quantization",
            "int8",
            "--port",
            "9123",
        ]
    )

    assert engine_args["init"] == ("nemo-parakeet-tdt-0.6b-v3", tmp_path, "int8")
    assert runs["port"] == 9123
    assert runs["host"] == "127.0.0.1"


def test_read_wav_returns_none_for_corrupt_input() -> None:
    """Truncated (EOFError) and malformed (wave.Error) WAV bytes both yield None."""
    from drove.workers.asr import _read_wav

    truncated = make_wav()[:12]
    not_wave = b"RIFF\x10\x00\x00\x00WAVXgarbage!"
    bogus_chunk = b"RIFF\x10\x00\x00\x00WAVEgarbage!"  # chunk seek past EOF → RuntimeError
    assert _read_wav(truncated) is None
    assert _read_wav(not_wave) is None
    assert _read_wav(bogus_chunk) is None


def test_normalize_audio_corrupt_wav_without_ffmpeg_raises_415(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("drove.workers.asr.shutil.which", lambda _: None)
    with pytest.raises(HTTPException) as excinfo:
        normalize_audio(make_wav()[:12], tmp_path)
    assert excinfo.value.status_code == 415


def test_normalize_audio_8bit_wav_treated_as_non_conforming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-16-bit WAV skips the pure-Python path and needs the ffmpeg decoder."""
    monkeypatch.setattr("drove.workers.asr.shutil.which", lambda _: None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(TARGET_RATE)
        w.writeframes(b"\x80" * 1000)
    with pytest.raises(HTTPException) as excinfo:
        normalize_audio(buf.getvalue(), tmp_path)
    assert excinfo.value.status_code == 415


def test_normalize_audio_ffmpeg_failure_raises_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("drove.workers.asr.shutil.which", lambda _: "/usr/bin/ffmpeg")
    proc = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=b"",
        stderr=b"pipe:0: Invalid data found when processing input\n",
    )
    monkeypatch.setattr("drove.workers.asr.subprocess.run", lambda *a, **kw: proc)
    with pytest.raises(HTTPException) as excinfo:
        normalize_audio(b"\xffnot audio at all", tmp_path)
    assert excinfo.value.status_code == 400
    assert "Invalid data found" in excinfo.value.detail
