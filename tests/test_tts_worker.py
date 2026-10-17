"""Speech API tests without model downloads, torch, or GPU."""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import unittest
import wave
from unittest.mock import MagicMock, patch

import httpx

from drove.workers.tts_runtime import engines
from drove.workers.tts_runtime.worker import CathyEngine, create_tts_app, encode_audio

PCM = struct.pack("<2400h", *([1000, -1000] * 1200))


class FakeEngine:
    native_speed = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def synthesize(self, text: str, voice: str, speed: float) -> tuple[bytes, int]:
        self.calls.append((text, voice, speed))
        return PCM, 24000


class SpeechApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = FakeEngine()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_tts_app(self.engine, "kokoro")),
            base_url="http://worker",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def speech(self, **overrides: object) -> httpx.Response:
        return await self.client.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Hello world.",
                "voice": "alloy",
                "response_format": "wav",
                **overrides,
            },
        )

    async def test_wav_is_binary_and_preserves_samples(self) -> None:
        response = await self.speech(speed=1.5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        with wave.open(io.BytesIO(response.content)) as wav:
            self.assertEqual(
                (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                (1, 2, 24000),
            )
            self.assertEqual(wav.readframes(wav.getnframes()), PCM)
        self.assertEqual(self.engine.calls, [("Hello world.", "alloy", 1.5)])

    async def test_pcm_is_raw_little_endian_audio(self) -> None:
        response = await self.speech(response_format="pcm")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/pcm")
        self.assertEqual(response.content, PCM)

    async def test_health(self) -> None:
        self.assertEqual(
            (await self.client.get("/health")).json(),
            {"status": "ok", "model": "kokoro"},
        )

    async def test_invalid_requests_do_not_run_engine(self) -> None:
        cases = [
            {"input": ""},
            {"input": " \n "},
            {"input": "x" * 4097},
            {"input": 12},
            {"voice": ""},
            {"speed": 0.24},
            {"speed": 4.01},
            {"response_format": "json"},
            {"instructions": "Whisper"},
            {"stream_format": "sse"},
            {"unknown": True},
        ]
        for case in cases:
            with self.subTest(case=case):
                response = await self.speech(**case)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertFalse(self.engine.calls)

    async def test_required_openai_fields(self) -> None:
        for field in ("model", "input", "voice"):
            body = {"model": "kokoro", "input": "Hello", "voice": "alloy"}
            del body[field]
            response = await self.client.post("/v1/audio/speech", json=body)
            self.assertEqual(response.status_code, 400)

    async def test_unknown_model(self) -> None:
        self.assertEqual((await self.speech(model="other")).status_code, 404)
        self.assertFalse(self.engine.calls)

    async def test_missing_ffmpeg_returns_actionable_error(self) -> None:
        with patch("shutil.which", return_value=None):
            response = await self.speech(response_format="mp3")
        self.assertEqual(response.status_code, 400)
        self.assertIn("ffmpeg", response.json()["error"]["message"])
        self.assertFalse(self.engine.calls)

    async def test_default_output_is_mp3(self) -> None:
        with (
            patch("shutil.which", return_value="/bin/ffmpeg"),
            patch(
                "drove.workers.tts_runtime.worker.encode_audio",
                return_value=b"mp3 bytes",
            ) as encode,
        ):
            response = await self.client.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hello",
                    "voice": "alloy",
                },
            )
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        self.assertEqual(response.content, b"mp3 bytes")
        encode.assert_called_once_with(PCM, 24000, "mp3", 1.0)

    async def test_errors_do_not_expose_engine_internals(self) -> None:
        with (
            patch.object(
                self.engine,
                "synthesize",
                side_effect=RuntimeError("/private/model/path"),
            ),
            self.assertLogs("drove.workers.tts_runtime.worker", level="ERROR"),
        ):
            response = await self.speech()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("/private", response.text)

    async def test_engine_without_native_speed_uses_audio_conversion(self) -> None:
        self.engine.native_speed = False
        with (
            patch("shutil.which", return_value="/bin/ffmpeg"),
            patch(
                "drove.workers.tts_runtime.worker.encode_audio",
                return_value=b"audio",
            ) as encode,
        ):
            response = await self.speech(speed=0.25)
        self.assertEqual(response.status_code, 200)
        encode.assert_called_once_with(PCM, 24000, "wav", 0.25)


class CathyVoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CathyEngine.__new__(CathyEngine)
        self.engine.engines = engines
        self.engine.engine = MagicMock()
        self.engine.name = "kokoro"
        self.engine.default_voice = "female"
        self.engine.engine.pipeline.lang_code = "a"

    def test_kokoro_aliases_and_blends(self) -> None:
        self.assertEqual(self.engine._voice("alloy"), "af_heart")
        self.assertEqual(self.engine._voice("male"), "am_michael:2,am_fenrir:1")
        self.assertEqual(self.engine._voice("af_heart:2,af_bella:1"), "af_heart:2,af_bella:1")

    def test_uncurated_default_voice_is_validated_against_model(self) -> None:
        self.engine.default_voice = "af_sarah"
        self.assertEqual(self.engine._voice("alloy"), "af_sarah")
        self.engine.engine.validate_voice.assert_called_once_with("af_sarah")
        self.engine.engine.validate_voice.side_effect = ValueError("Voice absent")
        with self.assertRaisesRegex(ValueError, "Voice absent"):
            self.engine._voice("af_missing")

    def test_voice_paths_invalid_weights_and_wrong_language_rejected(self) -> None:
        for voice in (
            "/tmp/voice.pt",
            "../voice",
            "af_heart:-1",
            "af_heart:nan",
            "af_heart:inf",
            "bf_emma",
        ):
            with self.subTest(voice=voice), self.assertRaises(ValueError):
                self.engine._voice(voice)

    def test_qwen_speakers(self) -> None:
        self.engine.name = "qwen"
        self.assertEqual(self.engine._voice("alloy"), "Vivian")
        self.assertEqual(self.engine._voice("Ryan"), "Ryan")
        with self.assertRaises(ValueError):
            self.engine._voice("/tmp/voice.wav")

    def test_cloning_reference_only_from_configuration(self) -> None:
        self.engine.name = "fish"
        self.engine.default_voice = "/trusted/reference.wav"
        self.assertEqual(self.engine._voice("alloy"), "/trusted/reference.wav")
        with self.assertRaises(ValueError):
            self.engine._voice("/etc/passwd")

    def test_cathy_chunk_limit_is_preserved(self) -> None:
        for text in ("word " * 1000, "字" * 4096, "Hello. " * 600):
            chunks = engines.split_sentences(text, 400)
            self.assertTrue(chunks)
            self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))
            self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))


class AudioEncodingTests(unittest.TestCase):
    def test_wav_and_pcm_need_no_ffmpeg(self) -> None:
        with patch("shutil.which", return_value=None):
            self.assertTrue(encode_audio(PCM, 24000, "wav").startswith(b"RIFF"))
            self.assertEqual(encode_audio(PCM, 24000, "pcm"), PCM)
            with self.assertRaisesRegex(ValueError, "ffmpeg"):
                encode_audio(PCM, 44100, "pcm")

    def test_encoder_failure_is_sanitized(self) -> None:
        with (
            patch("shutil.which", return_value="/bin/ffmpeg"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, b"", b"/private/path"),
            ),
            self.assertLogs("drove.workers.tts_runtime.worker", level="ERROR"),
        ):
            with self.assertRaisesRegex(RuntimeError, "^Failed to encode speech audio.$"):
                encode_audio(PCM, 24000, "mp3")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is optional")
    def test_real_encodings_are_decodable(self) -> None:
        for fmt in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            with self.subTest(format=fmt):
                data = encode_audio(PCM, 24000, fmt)
                args = ["ffmpeg", "-v", "error"]
                if fmt == "pcm":
                    args += ["-f", "s16le", "-ar", "24000", "-ac", "1"]
                decoded = subprocess.run(
                    args + ["-i", "pipe:0", "-f", "s16le", "pipe:1"],
                    input=data,
                    capture_output=True,
                )
                self.assertEqual(decoded.returncode, 0, decoded.stderr)
                self.assertTrue(decoded.stdout)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is optional")
    def test_extreme_speeds_and_pcm_resampling(self) -> None:
        for speed in (0.25, 4.0):
            with self.subTest(speed=speed):
                self.assertTrue(encode_audio(PCM * 10, 24000, "wav", speed).startswith(b"RIFF"))
        self.assertEqual(len(encode_audio(PCM, 48000, "pcm")), len(PCM) // 2)


if __name__ == "__main__":
    unittest.main()
