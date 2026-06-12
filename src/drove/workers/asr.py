"""Built-in speech-to-text worker with an OpenAI-compatible API.

Spawned by ServerManager the same way llama-server is::

    python -m drove.workers.asr --model-dir <dir> --model-type <type> --port <port>

Loads an ONNX ASR model (e.g. NVIDIA Parakeet TDT) via the optional
``onnx-asr`` package and exposes:

- ``GET /health`` — readiness probe (the port is only bound after the model
  has finished loading, so health polling doubles as a load barrier)
- ``POST /v1/audio/transcriptions`` — OpenAI-compatible transcription

Audio is normalized to 16 kHz mono 16-bit PCM before recognition: conforming
WAV uploads pass through untouched, other inputs are converted with ``ffmpeg``
when available, with a pure-Python fallback for non-conforming WAV files.
"""

from __future__ import annotations

import argparse
import array
import io
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

logger = logging.getLogger(__name__)

TARGET_RATE = 16_000

# Memory-exhaustion guard for uploads (the file is read fully into memory),
# not an audio-length limit; ~100 MB fits well over an hour of 16-bit WAV.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

_RESPONSE_FORMATS = frozenset({"json", "text", "verbose_json"})


class AsrEngine(Protocol):
    """Minimal interface the worker needs from a recognition engine."""

    def recognize(self, wav_path: str) -> str: ...


class OnnxAsrEngine:
    """Recognition engine backed by onnx-asr (imported lazily)."""

    def __init__(self, model_type: str, model_dir: Path, quantization: str | None = None) -> None:
        try:
            import onnx_asr
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "The 'onnx-asr' package is required for speech-to-text models. "
                "Install drove with the asr extra: pip install 'drove[asr]'"
            ) from e
        logger.info("Loading ASR model %s from %s", model_type, model_dir)
        self._model = onnx_asr.load_model(model_type, str(model_dir), quantization=quantization)
        logger.info("ASR model loaded")

    def recognize(self, wav_path: str) -> str:
        return str(self._model.recognize(wav_path))


def create_asr_app(engine: AsrEngine, model_name: str = "asr") -> FastAPI:
    app = FastAPI(title="drove-asr-worker")

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "model": model_name})

    @app.post("/v1/audio/transcriptions")
    def transcribe(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "",
        response_format: Annotated[str, Form()] = "json",
        language: Annotated[str | None, Form()] = None,
        temperature: Annotated[float, Form()] = 0.0,
    ) -> Response:
        if response_format not in _RESPONSE_FORMATS:
            supported = ", ".join(sorted(_RESPONSE_FORMATS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format '{response_format}'. Supported: {supported}",
            )

        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )

        with tempfile.TemporaryDirectory(prefix="drove-asr-") as tmp:
            wav_path, duration = normalize_audio(data, Path(tmp))
            text = engine.recognize(str(wav_path)).strip()

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "verbose_json":
            return JSONResponse(
                {
                    "task": "transcribe",
                    "language": language,
                    "duration": round(duration, 3),
                    "text": text,
                    "segments": [],
                }
            )
        return JSONResponse({"text": text})

    return app


def normalize_audio(data: bytes, tmp_dir: Path) -> tuple[Path, float]:
    """Write *data* as a 16 kHz mono 16-bit PCM WAV file in *tmp_dir*.

    Returns (path, duration_seconds).  Conforming WAV input is written as-is;
    everything else goes through ffmpeg when available.  Non-conforming WAV
    files fall back to a pure-Python conversion when ffmpeg is missing.
    """
    out = tmp_dir / "audio.wav"
    parsed = _read_wav(data)

    if parsed is not None:
        samples, channels, rate = parsed
        if channels == 1 and rate == TARGET_RATE:
            out.write_bytes(data)
            return out, len(samples) / TARGET_RATE
        if shutil.which("ffmpeg"):
            _convert_with_ffmpeg(data, out)
            return out, _wav_duration(out)
        mono = _to_mono(samples, channels)
        resampled = _resample(mono, rate, TARGET_RATE)
        _write_wav(resampled, out)
        return out, len(resampled) / TARGET_RATE

    # Not a (supported) WAV file — ffmpeg is the only decoder we have.
    _convert_with_ffmpeg(data, out)
    return out, _wav_duration(out)


def _read_wav(data: bytes) -> tuple[array.array[int], int, int] | None:
    """Parse 16-bit PCM WAV bytes into (samples, channels, rate), or None."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                return None
            samples: array.array[int] = array.array("h")
            samples.frombytes(w.readframes(w.getnframes()))
            return samples, w.getnchannels(), w.getframerate()
    # RuntimeError: the chunk reader behind wave raises it for bogus chunk
    # sizes that seek past the end of the data.
    except wave.Error, EOFError, RuntimeError:
        return None


def _to_mono(samples: array.array[int], channels: int) -> array.array[int]:
    if channels == 1:
        return samples
    frames = len(samples) // channels
    mono: array.array[int] = array.array("h", bytes(2 * frames))
    for i in range(frames):
        base = i * channels
        mono[i] = sum(samples[base : base + channels]) // channels
    return mono


def _resample(samples: array.array[int], src_rate: int, dst_rate: int) -> array.array[int]:
    """Linear-interpolation resampler (pure Python fallback)."""
    if src_rate == dst_rate or not samples:
        return samples
    n_out = max(1, int(len(samples) * dst_rate / src_rate))
    out: array.array[int] = array.array("h", bytes(2 * n_out))
    step = (len(samples) - 1) / (n_out - 1) if n_out > 1 else 0.0
    for i in range(n_out):
        pos = i * step
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out[i] = int(samples[lo] * (1 - frac) + samples[hi] * frac)
    return out


def _write_wav(samples: array.array[int], path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(samples.tobytes())


def _wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        return w.getnframes() / rate if rate else 0.0


def _convert_with_ffmpeg(data: bytes, out: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(
            status_code=415,
            detail=(
                "Audio is not 16-bit PCM WAV and 'ffmpeg' is not on PATH. "
                "Install ffmpeg or upload a WAV file."
            ),
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            str(TARGET_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "-y",
            str(out),
        ],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise HTTPException(
            status_code=400, detail="ffmpeg failed to decode audio: " + " | ".join(tail)
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m drove.workers.asr",
        description="drove built-in speech-to-text worker (onnx-asr).",
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Model directory.")
    parser.add_argument(
        "--model-type", required=True, help="onnx-asr model type, e.g. nemo-parakeet-tdt-0.6b-v3."
    )
    parser.add_argument("--quantization", default=None, help="Optional quantization, e.g. int8.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load the model before binding the port: drove polls /health until the
    # socket answers, so a late bind acts as the readiness barrier.
    engine = OnnxAsrEngine(args.model_type, args.model_dir, args.quantization)
    app = create_asr_app(engine, model_name=args.model_type)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
