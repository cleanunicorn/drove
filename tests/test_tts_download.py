"""TTS downloads use the regular planner/CLI and serve the downloaded files."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from drove.backend import infer_tts_engine
from drove.cli.models import _select_quant_variant, models_app
from drove.config import Config
from drove.downloader import FileStatus, resolve_download
from drove.model_config import load_download_info, load_model_config, set_model_config_key
from drove.model_store import ModelStore
from drove.server_manager import ServerManager
from drove.workers.tts_runtime import engines

FILES = {
    "kokoro-v1_0.pth": b"weights",
    "voices/af_heart.pt": b"voice",
    "config.json": b"{}",
    "speech_tokenizer/config.json": b'{"tokenizer":true}',
}


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    siblings = [SimpleNamespace(rfilename=path, size=len(data)) for path, data in FILES.items()]
    siblings.append(SimpleNamespace(rfilename="README.md", size=100))
    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(siblings=siblings)
    monkeypatch.setattr("drove.downloader.HfApi", lambda: api)
    downloaded: list[str] = []

    def download(*, repo_id: str, filename: str, local_dir: str, tqdm_class=None) -> str:
        downloaded.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FILES[filename])
        return str(path)

    monkeypatch.setattr("drove.downloader.hf_hub_download", download)
    return downloaded


@pytest.mark.parametrize(
    "repo,engine",
    [
        ("hexgrad/Kokoro-82M", "kokoro"),
        ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen"),
        ("ResembleAI/chatterbox-nano", "chatterbox"),
        ("fishaudio/s1-mini", "fish"),
    ],
)
def test_supported_repositories_detect_tts(repo: str, engine: str, hub: list[str]) -> None:
    plan = resolve_download(repo)
    assert plan.tts_engine == engine
    assert infer_tts_engine(repo.upper()) == engine
    assert plan.is_tts and not plan.is_asr
    assert plan.total_bytes == sum(map(len, FILES.values()))
    assert "README.md" not in plan._all_remote_files()
    assert not hub
    assert _select_quant_variant(plan, None) is None


def test_explicit_engine_supports_compatible_repos(hub: list[str]) -> None:
    plan = resolve_download("my-org/my-model", tts_engine="qwen")
    assert plan.tts_engine == "qwen"
    with pytest.raises(ValueError, match="Unknown TTS engine"):
        resolve_download("my-org/my-model", tts_engine="bad")
    with pytest.raises(ValueError, match="quantization"):
        resolve_download("hexgrad/Kokoro-82M:Q8_0")


def test_download_preserves_layout_and_skips_complete_files(tmp_path: Path, hub: list[str]) -> None:
    plan = resolve_download("hexgrad/Kokoro-82M")
    primary = plan.execute(tmp_path)
    assert primary.name == "model.tts"
    assert load_model_config(primary).tts_model_dir == "."
    assert load_model_config(primary).tts_engine == "kokoro"
    for name, content in FILES.items():
        assert (primary.parent / name).read_bytes() == content
    assert not (primary.parent / "af_heart.pt").exists()
    assert ModelStore(tmp_path).resolve(plan.local_name) == primary
    assert {status for status, _ in plan.check_local_files(tmp_path).values()} == {
        FileStatus.COMPLETE
    }
    plan.execute(tmp_path)
    assert sorted(hub) == sorted(FILES)
    # An incomplete voice alone is re-downloaded, in its original directory.
    (primary.parent / "voices/af_heart.pt").write_bytes(b"v")
    plan.execute(tmp_path)
    assert hub.count("voices/af_heart.pt") == 2
    assert hub.count("kokoro-v1_0.pth") == 1


def test_failed_download_does_not_publish_marker(tmp_path: Path, hub: list[str]) -> None:
    plan = resolve_download("hexgrad/Kokoro-82M")
    with patch("drove.downloader.hf_hub_download", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError, match="offline"):
            plan.execute(tmp_path)
    assert not (plan.destination(tmp_path) / "model.tts").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../b", "a\\b"])
def test_download_rejects_escaping_names(name: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_download("hexgrad/Kokoro-82M", name_override=name)


def test_cli_download_registers_and_recovers_complete_download(
    tmp_path: Path, hub: list[str]
) -> None:
    runner = CliRunner()
    obj = {"config": Config(models_dir=tmp_path)}
    args = ["download", "hexgrad/Kokoro-82M", "--name", "kokoro", "--yes"]
    result = runner.invoke(models_app, args, obj=obj)
    assert result.exit_code == 0, result.output
    primary = ModelStore(tmp_path).resolve("kokoro")
    config = load_model_config(primary)
    assert (config.backend, config.tts_engine, config.tts_model_dir) == ("tts", "kokoro", ".")
    assert ModelStore(tmp_path).resolve("hexgrad/Kokoro-82M") == primary
    metadata = load_download_info(primary)
    assert metadata and sorted(metadata.files) == sorted(FILES)
    set_model_config_key(primary, "tts_voice", "male")
    primary.unlink()  # Recover a completed download whose registration was interrupted.
    result = runner.invoke(models_app, args[:-1], obj=obj)
    assert result.exit_code == 0, result.output
    assert primary.exists()
    assert load_model_config(primary).tts_voice == "male"
    assert sorted(hub) == sorted(FILES)
    with patch("shutil.which", return_value="/usr/bin/uv"):
        cmd = ServerManager(obj["config"])._build_tts_command(
            "kokoro",
            primary,
            load_model_config(primary),
            9000,
        )
    assert cmd[cmd.index("--model-dir") + 1] == str(primary.parent)


def test_cli_explicit_tts_engine(tmp_path: Path, hub: list[str]) -> None:
    result = CliRunner().invoke(
        models_app,
        ["download", "my-org/model", "--tts-engine", "qwen", "--yes"],
        obj={"config": Config(models_dir=tmp_path)},
    )
    assert result.exit_code == 0, result.output
    primary = ModelStore(tmp_path).resolve("my-org/model")
    assert load_model_config(primary).tts_engine == "qwen"


def test_kokoro_loads_checkpoint_and_voice_from_model_directory(tmp_path: Path) -> None:
    (tmp_path / "kokoro-v1_0.pth").touch()
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/af_heart.pt").touch()
    pipeline = MagicMock()
    kokoro = SimpleNamespace(KPipeline=MagicMock(return_value=pipeline), KModel=MagicMock())
    with (
        patch.object(engines, "_quiet"),
        patch.object(engines, "import_module", return_value=kokoro),
    ):
        engine = engines.KokoroEngine("female", 1.0, "cpu", model_dir=tmp_path)
    assert kokoro.KModel.call_args.kwargs["model"] == str(tmp_path / "kokoro-v1_0.pth")
    assert kokoro.KModel.call_args.kwargs["config"] == str(tmp_path / "config.json")
    pipeline.load_single_voice.assert_called_once_with(str(tmp_path / "voices/af_heart.pt"))
    with pytest.raises(ValueError, match="absent"):
        engine._resolve("af_bella")


def test_kokoro_validates_downloaded_uncurated_voice(tmp_path: Path) -> None:
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/af_sarah.pt").touch()
    engine = engines.KokoroEngine.__new__(engines.KokoroEngine)
    engine.model_dir = tmp_path
    engine.pipeline = MagicMock()
    engine.validate_voice("af_sarah")
    engine.pipeline.load_single_voice.assert_called_once_with(str(tmp_path / "voices/af_sarah.pt"))
    with pytest.raises(ValueError, match="absent"):
        engine.validate_voice("af_missing")


def test_kokoro_validates_remote_voice_against_repository() -> None:
    from huggingface_hub.errors import EntryNotFoundError

    engine = engines.KokoroEngine.__new__(engines.KokoroEngine)
    engine.model_dir = None
    engine.pipeline = MagicMock()
    engine.validate_voice("af_sarah")
    engine.pipeline.load_single_voice.assert_called_once_with("af_sarah")
    engine.pipeline.load_single_voice.side_effect = EntryNotFoundError("missing")
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        engine.validate_voice("af_missing")
    engine.pipeline.load_single_voice.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        engine.validate_voice("af_sarah")


def test_qwen_loads_downloaded_directory(tmp_path: Path) -> None:
    qwen = SimpleNamespace(Qwen3TTSModel=MagicMock())
    torch = SimpleNamespace(bfloat16="bfloat16")
    with (
        patch.object(engines, "_quiet"),
        patch("shutil.which", return_value="/bin/sox"),
        patch.object(
            engines,
            "import_module",
            side_effect=lambda name: qwen if name == "qwen_tts" else torch,
        ),
        patch.object(engines.QwenEngine, "_generate", return_value=([[]], 24000)),
    ):
        engines.QwenEngine("female", 1.0, "cpu", model_dir=tmp_path)
    assert qwen.Qwen3TTSModel.from_pretrained.call_args.args == (str(tmp_path),)


def test_chatterbox_uses_local_loader(tmp_path: Path) -> None:
    cls = MagicMock()
    with (
        patch.object(engines, "_quiet"),
        patch.object(
            engines,
            "import_module",
            return_value=SimpleNamespace(ChatterboxTurboTTS=cls),
        ),
        patch.object(engines.ChatterboxEngine, "reference_clip", return_value=tmp_path / "ref.wav"),
    ):
        engines.ChatterboxEngine("female", 1.0, "cpu", model_dir=tmp_path)
    cls.from_local.assert_called_once_with(tmp_path, device="cpu", nano=True)
    cls.from_pretrained.assert_not_called()


@pytest.mark.parametrize("reference_kind", ["preset", "custom", "missing", "empty"])
def test_fish_uses_local_checkpoints_without_snapshot_download(
    tmp_path: Path, reference_kind: str
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    voice = "female" if reference_kind == "preset" else str(reference)
    if reference_kind == "custom":
        reference.with_suffix(".txt").write_text("Actual words in the recording.")
    elif reference_kind == "empty":
        reference.with_suffix(".txt").write_text("  ")
    launch = MagicMock()
    load_decoder = MagicMock()
    load_decoder.return_value.spec_transform.sample_rate = 24000
    snapshot = MagicMock(side_effect=AssertionError("must not download weights during serving"))
    modules = {
        "fish_speech": SimpleNamespace(__path__=[str(tmp_path / "fish_speech")]),
        "fish_speech.inference_engine": SimpleNamespace(TTSInferenceEngine=MagicMock()),
        "tqdm": SimpleNamespace(tqdm=MagicMock()),
        "fish_speech.models.text2semantic.inference": SimpleNamespace(
            launch_thread_safe_queue=launch
        ),
        "torch": SimpleNamespace(bfloat16="bfloat16"),
        "fish_speech.models.dac.inference": SimpleNamespace(load_model=load_decoder),
        "fish_speech.utils.schema": SimpleNamespace(ServeReferenceAudio=MagicMock()),
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
        "huggingface_hub.errors": SimpleNamespace(GatedRepoError=RuntimeError),
    }
    with (
        patch.object(engines, "_quiet"),
        patch.object(
            engines,
            "import_module",
            side_effect=modules.__getitem__,
        ),
        patch.object(engines.FishEngine, "reference_clip", return_value=reference),
    ):
        if reference_kind in {"missing", "empty"}:
            with pytest.raises(ValueError, match="companion .txt transcript"):
                engines.FishEngine(voice, 1.0, "cpu", model_dir=tmp_path)
            launch.assert_not_called()
            return
        engine = engines.FishEngine(voice, 1.0, "cpu", model_dir=tmp_path)
    assert modules["fish_speech.utils.schema"].ServeReferenceAudio.call_args.kwargs["text"] == (
        engines.REFERENCE_TEXT if reference_kind == "preset" else "Actual words in the recording."
    )
    snapshot.assert_not_called()
    assert launch.call_args.kwargs["checkpoint_path"] == tmp_path
    assert load_decoder.call_args.kwargs["checkpoint_path"] == tmp_path / "codec.pth"
    assert engine.sr == 24000
