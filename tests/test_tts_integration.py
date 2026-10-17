"""TTS registration, backend selection, and managed worker lifecycle."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from drove.backend import BACKEND_TTS, detect_backend
from drove.cli.models import _detect_capabilities, add_tts_model, models_app
from drove.config import Config
from drove.integrations import describe_model
from drove.model_config import ModelConfig, load_model_config, set_model_config_key
from drove.model_store import ModelStore
from drove.server_manager import ServerManager, _estimate_model_memory


def register(tmp_path: Path, name: str = "kokoro", **kwargs: str) -> Path:
    ctx = typer.Context(
        typer.main.get_command(models_app), obj={"config": Config(models_dir=tmp_path)},
    )
    add_tts_model(
        ctx, name, kwargs.get("engine", "kokoro"), kwargs.get("voice", "female"),
        kwargs.get("device", "cpu"),
    )
    return ModelStore(tmp_path).resolve(name)


def test_registration_uses_existing_model_management(tmp_path: Path) -> None:
    primary = register(tmp_path)
    store = ModelStore(tmp_path)
    assert primary == tmp_path / "kokoro" / "model.tts"
    assert detect_backend(primary, load_model_config(primary)) == BACKEND_TTS
    assert _detect_capabilities(primary) == ["tts"]
    assert store.complete("ko") == ["kokoro"]
    assert [entry.name for entry in store.list()] == ["kokoro"]
    assert store.find_root("kokoro") == primary.parent
    assert describe_model(Config(models_dir=tmp_path), "kokoro", primary) is None
    set_model_config_key(primary, "tts_voice", "am_michael")
    assert load_model_config(primary).tts_voice == "am_michael"


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "..", "a/b", "--option"])
def test_registration_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(typer.BadParameter):
        register(tmp_path, name)
    assert not list(tmp_path.iterdir())


def test_registration_does_not_overwrite_models(tmp_path: Path) -> None:
    primary = register(tmp_path)
    before = primary.with_suffix(".toml").read_bytes()
    with pytest.raises(typer.BadParameter, match="already exists"):
        register(tmp_path, engine="fish")
    assert primary.with_suffix(".toml").read_bytes() == before


@pytest.mark.parametrize("key,value", [
    ("tts_engine", "bad"), ("tts_device", "--bad"),
    ("tts_memory_mb", "0"), ("tts_max_chunk_chars", "100"),
])
def test_tts_settings_validated(tmp_path: Path, key: str, value: str) -> None:
    primary = register(tmp_path)
    with pytest.raises(ValueError):
        set_model_config_key(primary, key, value)


def test_tts_config_is_never_sent_to_llama() -> None:
    cfg = ModelConfig(
        backend="tts", tts_engine="kokoro", tts_voice="female", tts_device="cpu",
        tts_language="English", tts_max_chunk_chars=400, tts_memory_mb=2048, ctx_size=4096,
    )
    assert cfg.to_llama_args() == ["--ctx-size", "4096"]


def test_tts_memory_accounts_for_cached_weights(tmp_path: Path) -> None:
    primary = register(tmp_path)
    assert _estimate_model_memory(primary) == 3072 * 1024**2
    set_model_config_key(primary, "tts_memory_mb", "5120")
    assert _estimate_model_memory(primary) == 5120 * 1024**2


@pytest.mark.parametrize("engine", ["kokoro", "qwen", "chatterbox", "fish"])
def test_each_engine_uses_an_isolated_compatible_environment(tmp_path: Path, engine: str) -> None:
    primary = register(tmp_path, engine=engine)
    manager = ServerManager(Config(models_dir=tmp_path))
    with patch("shutil.which", return_value="/usr/bin/uv"):
        cmd = manager._build_tts_command("kokoro", primary, load_model_config(primary), 9000)
    assert cmd[:5] == ["/usr/bin/uv", "run", "--no-project", "--python", "3.12"]
    requirements = Path(cmd[cmd.index("--with-requirements") + 1])
    assert requirements.name == f"{engine}.txt"
    assert "torch>=2.6.0,<2.9" in requirements.read_text()
    assert cmd[cmd.index("--engine") + 1] == engine
    assert cmd[cmd.index("--port") + 1] == "9000"


def test_missing_uv_has_actionable_error(tmp_path: Path) -> None:
    primary = register(tmp_path)
    manager = ServerManager(Config(models_dir=tmp_path))
    with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="requires uv"):
        manager._build_tts_command("kokoro", primary, load_model_config(primary), 9000)


async def test_lazy_start_and_managed_shutdown_use_tts_worker(tmp_path: Path) -> None:
    register(tmp_path)
    manager = ServerManager(Config(models_dir=tmp_path, prompt_cache=False))
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.stdout = None
    process.stderr = None
    process.wait = AsyncMock(return_value=0)
    with patch("shutil.which", return_value="/usr/bin/uv"), patch(
        "asyncio.create_subprocess_exec", AsyncMock(return_value=process),
    ) as spawn, patch.object(manager, "_wait_for_health", AsyncMock()), patch(
        "drove.server_manager._find_free_port", return_value=9000,
    ):
        try:
            await manager.ensure_running("kokoro", claim=True)
            assert "kokoro" in manager.loaded_models
            assert "worker.py" in " ".join(spawn.call_args.args)
            assert manager._instances["kokoro"].backend == BACKEND_TTS
            manager.request_finished("kokoro")
        finally:
            await manager.stop()
    process.send_signal.assert_called_once()
    assert not manager.loaded_models
