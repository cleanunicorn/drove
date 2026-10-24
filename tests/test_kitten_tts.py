"""KittenTTS integration contracts without heavyweight runtime dependencies."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from drove.cli.models import models_app
from drove.config import Config
from drove.downloader import resolve_download
from drove.model_config import ModelConfig, load_model_config, save_model_config
from drove.model_store import ModelStore
from drove.server_manager import ServerManager, _estimate_model_memory
from drove.workers.tts_runtime import engines
from drove.workers.tts_runtime.worker import CathyEngine, create_tts_app


@pytest.fixture
def kitten(tmp_path: Path):
    model = MagicMock(sample_rate=24000, available_voices=["Bella", "Bruno", "German", "Kiki"])
    factory = MagicMock(return_value=model)
    with (
        patch.object(engines, "_quiet"),
        patch.object(engines, "import_module", return_value=SimpleNamespace(KittenTTS=factory)),
    ):
        engine = engines.build_engine("kitten", "female", 1, "cpu", model_dir=tmp_path)
    factory.assert_called_once_with(str(tmp_path), device="cpu")
    return engine, model


def wrapper(engine, default="female"):
    worker = CathyEngine.__new__(CathyEngine)
    worker.engines = engines
    worker.name = "kitten"
    worker.default_voice = default
    worker.engine = engine
    worker.native_speed = False
    return worker


def test_builtin_voices_and_multilingual_expression_are_preserved(kitten):
    engine, model = kitten
    worker = wrapper(engine)
    assert worker._voice("alloy") == "Bella"
    assert worker._voice("male") == "Bruno"
    assert worker._voice("German") == "German"
    for voice in ["/etc/passwd", "../secret.wav", "missing", "https://host/voice.wav"]:
        with pytest.raises(ValueError, match="Unknown KittenTTS"):
            worker._voice(voice)
    engine.voice = "German"
    engine.preset = "expressive"
    text = "[joyful] Guten Morgen <laugh> (((Welt)))!"
    list(engine.paragraph(text))
    model.generate.assert_called_once_with(
        text, voice="German", normalize=False, preset="expressive"
    )


def test_native_chunking_and_normalization_options(kitten):
    engine, model = kitten
    engine.chunk_limit = 200
    engine.normalize = True
    list(engine.paragraph("$12.50"))
    assert model.generate.call_args.kwargs["normalize"] is True
    assert model.generate.call_args.kwargs["advanced"] == {"chunk_chars": 200}


@pytest.mark.parametrize("transcript", [None, "", "Actual words in the clip."])
def test_cloning_uses_only_configured_reference(tmp_path, kitten, transcript):
    _, model = kitten
    clip = tmp_path / "reference.wav"
    clip.write_bytes(b"audio")
    if transcript is not None:
        clip.with_suffix(".txt").write_text(transcript)
    with (
        patch.object(engines, "_quiet"),
        patch.object(
            engines, "import_module", return_value=SimpleNamespace(KittenTTS=lambda *a, **k: model)
        ),
    ):
        if not transcript:
            with pytest.raises(ValueError, match="companion .txt"):
                engines.KittenEngine(str(clip), 1, "cpu")
            return
        engine = engines.KittenEngine(str(clip), 1, "cpu")
    worker = wrapper(engine, str(clip))
    assert worker._voice("alloy") == str(clip)
    for voice in ["/etc/passwd", str(tmp_path / "other.wav"), "Bruno"]:
        with pytest.raises(ValueError, match="configured reference"):
            worker._voice(voice)
    list(engine.paragraph("Hello."))
    model.generate.assert_called_once_with(
        "Hello.", reference=str(clip), reference_text=transcript, normalize=False, preset="stable"
    )


def test_download_keeps_references_and_licenses_and_avoids_unused_weights(tmp_path):
    wanted = {
        "config.json",
        "lm/config.json",
        "lm/tokenizer.json",
        "lm/merges.txt",
        "lm/model-ternary.safetensors",
        "speaker/model.safetensors",
        "speaker/LICENSE",
        "voices/voices.json",
        "voices/bruno.wav",
        "voices/bruno.npz",
        "LICENSE.md",
        "README.md",
    }
    skipped = {
        "lm/model.safetensors",
        "lm/model-tl2-emb4.safetensors",
        "cpp/default/decoder.pt",
        "cpp/model.gguf",
        "decoders/student_w4.pt",
    }
    api = MagicMock()
    api.model_info.return_value.siblings = [
        SimpleNamespace(rfilename=p, size=1) for p in wanted | skipped
    ]

    def download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    with (
        patch("drove.downloader.HfApi", return_value=api),
        patch("drove.downloader.hf_hub_download", side_effect=download),
    ):
        plan = resolve_download("KittenML/kitten-tts-2", name_override="kitten")
        assert plan.is_tts and not plan.is_asr
        assert plan.tts_engine == "kitten"
        assert set(plan._all_remote_files()) == wanted
        primary = plan.execute(tmp_path)
        result = CliRunner().invoke(
            models_app,
            ["download", "KittenML/kitten-tts-2", "--name", "kitten-cli", "--yes"],
            obj={"config": Config(models_dir=tmp_path)},
        )
        assert result.exit_code == 0, result.output
        assert "Stellon Labs Community License" in result.output
        assert load_model_config(ModelStore(tmp_path).resolve("kitten-cli")).tts_engine == "kitten"
    assert load_model_config(primary).tts_engine == "kitten"
    assert ModelStore(tmp_path).resolve("kitten") == primary


def test_registration_command_and_resource_reservation(tmp_path):
    result = CliRunner().invoke(
        models_app,
        ["add-tts", "kitten", "--engine", "kitten"],
        obj={"config": Config(models_dir=tmp_path)},
    )
    assert result.exit_code == 0, result.output
    primary = ModelStore(tmp_path).resolve("kitten")
    assert _estimate_model_memory(primary) == 6144 * 1024**2
    cfg = ModelConfig(
        backend="tts",
        tts_engine="kitten",
        tts_normalize=True,
        tts_preset="expressive",
        tts_device="cpu",
        tts_max_chunk_chars=250,
    )
    assert cfg.to_llama_args() == []
    save_model_config(primary, cfg)
    with patch("shutil.which", return_value="/bin/uv"):
        cmd = ServerManager(Config(models_dir=tmp_path))._build_tts_command(
            "kitten", primary, cfg, 9000
        )
    assert cmd[cmd.index("--engine") + 1] == "kitten"
    assert Path(cmd[cmd.index("--with-requirements") + 1]).is_file()
    assert "--normalize" in cmd
    assert cmd[cmd.index("--preset") + 1] == "expressive"
    assert cmd[cmd.index("--device") + 1] == "cpu"


async def test_speech_api_returns_kitten_audio_and_rejects_voice_paths(kitten):
    engine, model = kitten
    # Minimal numpy stand-in checks conversion without installing ML in Drove.
    array = MagicMock()
    array.size = 3
    array.reshape.return_value = array
    array.__mul__.return_value = array
    array.astype.return_value.tobytes.return_value = b"\x00\x00" * 3
    np = SimpleNamespace(
        asarray=lambda a: array,
        concatenate=lambda a: array,
        isfinite=lambda a: SimpleNamespace(all=lambda: True),
        clip=lambda *a: array,
    )
    with patch("drove.workers.tts_runtime.worker.import_module", return_value=np):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_tts_app(wrapper(engine), "kitten")),
            base_url="http://worker",
        ) as client:
            response = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kitten",
                    "input": "Hi!",
                    "voice": "Bruno",
                    "response_format": "wav",
                },
            )
            assert response.status_code == 200
            assert response.content.startswith(b"RIFF")
            assert model.generate.call_args.kwargs["voice"] == "Bruno"
            response = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "kitten",
                    "input": "Hi!",
                    "voice": "/etc/passwd",
                    "response_format": "wav",
                },
            )
            assert response.status_code == 400
            assert model.generate.call_count == 1
