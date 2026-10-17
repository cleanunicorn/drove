"""OpenAI speech API over Cathy's local engines, in an isolated Python 3.12 env.

Heavy imports happen only at worker startup, before the health port is bound.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import shutil
import subprocess
import threading
import wave
from importlib import import_module
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

FORMATS = {
    "mp3": ("mp3", "libmp3lame", "audio/mpeg"),
    "opus": ("ogg", "libopus", "audio/ogg"),
    "aac": ("adts", "aac", "audio/aac"),
    "flac": ("flac", "flac", "audio/flac"),
    "wav": ("wav", "pcm_s16le", "audio/wav"),
    "pcm": ("s16le", "pcm_s16le", "audio/pcm"),
}


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, strict=True)
    input: str = Field(min_length=1, max_length=4096, strict=True)
    voice: str = Field(min_length=1, max_length=256, strict=True)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0, allow_inf_nan=False)
    instructions: str | None = None
    stream_format: Literal["audio"] = "audio"


class SpeechEngine(Protocol):
    native_speed: bool

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        """Return mono signed little-endian 16-bit PCM and sample rate."""
        ...


class CathyEngine:
    """Keep one loaded engine; serialize requests so voice changes cannot race."""

    def __init__(
        self,
        name: str,
        voice: str,
        device: str,
        language: str | None = None,
        max_chunk_chars: int | None = None,
    ) -> None:
        self.engines = import_module(f"{__package__}.engines" if __package__ else "engines")
        torch = import_module("torch")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = name
        self.default_voice = voice
        self.native_speed = name == "kokoro"
        self.engine: Any = self.engines.build_engine(
            name, voice, 1.0, device, language, max_chunk_chars,
        )

    def _voice(self, voice: str) -> str:
        # These aliases select local voices; they do not replicate OpenAI voices.
        aliases = {
            "alloy": self.default_voice, "default": self.default_voice,
            "echo": "male", "onyx": "male", "ash": "male", "ballad": "male",
            "cedar": "male", "verse": "male", "fable": "british-male",
            "nova": "female", "shimmer": "female", "coral": "female",
            "sage": "female", "marin": "female",
        }
        voice = aliases.get(voice, voice)
        if self.name == "kokoro":
            voice = self.engines.KOKORO_ALIASES.get(voice, voice)
            known = self.engines.KOKORO_VOICES | self.engines.KOKORO_INTL
            for part in voice.split(","):
                preset, _, weight = part.partition(":")
                if preset not in known:
                    raise ValueError("Unknown Kokoro voice; use a documented preset or alias.")
                if weight and (not math.isfinite(float(weight)) or float(weight) <= 0):
                    raise ValueError("Voice blend weights must be positive finite numbers.")
                if preset[0] != self.engine.pipeline.lang_code:
                    raise ValueError(
                        "This voice uses a different language; register a separate model."
                    )
            return voice
        if self.name == "qwen":
            voice = self.engines.QWEN_ALIASES.get(voice, voice)
            if voice not in self.engines.QWEN_SPEAKERS:
                raise ValueError("Unknown Qwen speaker; use a documented speaker or male/female.")
            return voice
        # Only trusted local config may select a reference file on this server.
        if voice != self.default_voice:
            raise ValueError(
                "Cloning engines use the configured reference voice; use voice='alloy'."
            )
        return voice

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        voice = self._voice(voice)
        if self.name == "kokoro":
            self.engine.voice = self.engine._resolve(voice)
            self.engine.speed = speed
        elif self.name == "qwen":
            self.engine.voice = voice
        np = import_module("numpy")
        chunks = [np.asarray(chunk).reshape(-1) for chunk in self.engine.paragraph(text)]
        if not chunks:
            raise RuntimeError("The speech engine produced no audio.")
        samples = np.concatenate(chunks)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("The speech engine produced invalid audio.")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return bytes(pcm), int(self.engine.sr)


def _tempo_filters(speed: float) -> str:
    factors = []
    while speed < 0.5:
        factors.append("atempo=0.5")
        speed /= 0.5
    while speed > 2:
        factors.append("atempo=2")
        speed /= 2
    factors.append(f"atempo={speed}")
    return ",".join(factors)


def encode_audio(pcm: bytes, rate: int, fmt: str, speed: float = 1.0) -> bytes:
    """Encode engine audio, keeping WAV independent of optional system tools."""
    if fmt == "pcm" and rate == 24000 and speed == 1.0:
        return pcm
    if fmt == "wav" and speed == 1.0:
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm)
        return out.getvalue()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "This output format or speed requires ffmpeg on the server; use WAV at speed 1."
        )
    container, codec, _ = FORMATS[fmt]
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(rate),
        "-ac", "1", "-i", "pipe:0",
    ]
    if speed != 1.0:
        cmd.extend(["-af", _tempo_filters(speed)])
    if fmt == "pcm":
        cmd.extend(["-ar", "24000"])
    cmd.extend(["-c:a", codec, "-f", container, "pipe:1"])
    proc = subprocess.run(cmd, input=pcm, capture_output=True, timeout=120)
    if proc.returncode or not proc.stdout:
        logger.error("Speech encoding failed: %s", proc.stderr.decode(errors="replace")[-1000:])
        raise RuntimeError("Failed to encode speech audio.")
    return proc.stdout


def create_tts_app(engine: SpeechEngine, model_name: str) -> FastAPI:
    app = FastAPI(title="drove-tts-worker")
    lock = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {
            "message": "Invalid speech request: " + "; ".join(e["msg"] for e in exc.errors()),
            "type": "invalid_request_error", "param": None, "code": None,
        }})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/v1/audio/speech")
    def speech(body: SpeechRequest) -> Response:
        def error(message: str, status: int = 400) -> JSONResponse:
            return JSONResponse(status_code=status, content={"error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": None, "code": None,
            }})

        if body.model != model_name:
            return error("This worker does not serve the requested model.", 404)
        if not body.input.strip():
            return error("Input must contain text.")
        if body.instructions:
            return error("This speech engine does not support instructions.")
        needs_ffmpeg = body.response_format not in {"wav", "pcm"} or (
            not engine.native_speed and body.speed != 1
        )
        if needs_ffmpeg and not shutil.which("ffmpeg"):
            return error("Install ffmpeg on the server for compressed audio or speed conversion.")
        try:
            with lock:
                pcm, rate = engine.synthesize(body.input, body.voice, body.speed)
                data = encode_audio(
                    pcm, rate, body.response_format, 1.0 if engine.native_speed else body.speed,
                )
        except ValueError as exc:
            return error(str(exc))
        except Exception:
            logger.exception("Speech synthesis failed")
            return error("Speech synthesis failed; check the server logs.", 500)
        return Response(data, media_type=FORMATS[body.response_format][2])

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="drove local text-to-speech worker")
    parser.add_argument("--engine", choices=("kokoro", "qwen", "chatterbox", "fish"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--voice", default="female")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--language")
    parser.add_argument("--max-chunk-chars", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    engine = CathyEngine(args.engine, args.voice, args.device, args.language, args.max_chunk_chars)
    app = create_tts_app(engine, args.model_name)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
