"""Tests for global config loading."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import tomli_w

from drove.config import (
    ConfigError,
    load_config,
    parse_size,
    resolve_config_path,
    save_tui_completed_hints,
    set_config_value,
)


def test_defaults(tmp_path: Path) -> None:
    """Config has sensible defaults when no file or env vars exist."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.listen_port == 8080
    assert cfg.idle_timeout_seconds == 1800
    assert cfg.speech_timeout_seconds == 600
    assert cfg.llama_server_bin == "llama-server"


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(
        tomli_w.dumps(
            {"listen_port": 9090, "idle_timeout_seconds": 600, "speech_timeout_seconds": 900}
        ).encode()
    )
    cfg = load_config(cfg_file)
    assert cfg.listen_port == 9090
    assert cfg.idle_timeout_seconds == 600
    assert cfg.speech_timeout_seconds == 900


def test_env_var_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(
        tomli_w.dumps({"listen_port": 9090, "speech_timeout_seconds": 900}).encode()
    )
    monkeypatch.setenv("DROVE_LISTEN_PORT", "7777")
    monkeypatch.setenv("DROVE_SPEECH_TIMEOUT_SECONDS", "1200")
    cfg = load_config(cfg_file)
    assert cfg.listen_port == 7777
    assert cfg.speech_timeout_seconds == 1200


def test_models_dir_expands_home(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"models_dir": "~/mymodels"}).encode())
    cfg = load_config(cfg_file)
    assert not str(cfg.models_dir).startswith("~")
    assert cfg.models_dir == Path.home() / "mymodels"


def test_save_and_reload(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    set_config_value("listen_port", 1234, cfg_file)
    set_config_value("speech_timeout_seconds", 2400, cfg_file)

    reloaded = load_config(cfg_file)
    assert reloaded.listen_port == 1234
    assert reloaded.speech_timeout_seconds == 2400


@pytest.mark.parametrize("value", [0, -1])
def test_speech_timeout_must_be_positive(tmp_path: Path, value: int) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"speech_timeout_seconds": value}).encode())
    with pytest.raises(ConfigError, match="speech_timeout_seconds"):
        load_config(cfg_file)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("", 0),
        ("1024", 1024),
        ("500b", 500),
        ("24GB", 24 * 1000**3),
        ("16GiB", 16 * 1024**3),
        ("512 MB", 512 * 1000**2),
        ("1.5gib", int(1.5 * 1024**3)),
        (2048, 2048),
    ],
)
def test_parse_size(value: str | int, expected: int) -> None:
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["24 gigabytes", "GB", "-1GB", "1..5GB"])
def test_parse_size_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid size"):
        parse_size(value)


def test_max_memory_defaults_to_unlimited(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.max_memory == "0"
    assert cfg.max_memory_bytes == 0


def test_max_memory_from_toml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"max_memory": "24GB"}).encode())
    cfg = load_config(cfg_file)
    assert cfg.max_memory_bytes == 24 * 1000**3


def test_max_memory_invalid_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"max_memory": "lots"}).encode())
    with pytest.raises(ConfigError, match="Invalid size"):
        load_config(cfg_file)


def test_max_memory_survives_save_and_reload(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    set_config_value("max_memory", "16GiB", cfg_file)

    reloaded = load_config(cfg_file)
    assert reloaded.max_memory == "16GiB"
    assert reloaded.max_memory_bytes == 16 * 1024**3


def test_resolve_config_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default = tmp_path / "default.toml"
    monkeypatch.setattr("drove.config.DEFAULT_CONFIG_PATH", default)
    monkeypatch.delenv("DROVE_CONFIG", raising=False)
    assert resolve_config_path() == default

    monkeypatch.setenv("DROVE_CONFIG", str(tmp_path / "env.toml"))
    assert resolve_config_path() == tmp_path / "env.toml"
    assert resolve_config_path(tmp_path / "flag.toml") == tmp_path / "flag.toml"


def test_load_config_reads_drove_config_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "custom.toml"
    cfg.write_text(tomli_w.dumps({"listen_port": 9123}))
    monkeypatch.setenv("DROVE_CONFIG", str(cfg))
    assert load_config().listen_port == 9123


@pytest.mark.parametrize("model", [None, "", "   ", "org/model:Q4_K_M"])
def test_default_model_roundtrip(tmp_path: Path, model: str | None) -> None:
    from drove.config import Config

    cfg = Config(default_model=model)
    expected = model if model and model.strip() else None
    assert cfg.default_model == expected
    path = tmp_path / "config.toml"
    set_config_value("default_model", model, path)
    assert load_config(path).default_model == expected
    assert ("default_model" in path.read_text()) == (expected is not None)


def test_default_model_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    assert load_config(path).default_model is None
    path.write_text('default_model = "file-model"\n')
    assert load_config(path).default_model == "file-model"
    monkeypatch.setenv("DROVE_DEFAULT_MODEL", "env-model")
    assert load_config(path).default_model == "env-model"
    monkeypatch.setenv("DROVE_DEFAULT_MODEL", " ")
    assert load_config(path).default_model is None


def test_tutorial_progress_round_trips_without_losing_settings(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    set_config_value("listen_port", 1234, cfg_file)
    set_config_value("tui_theme", "textual-light", cfg_file)
    set_config_value("tui_completed_hints", ["send", "commands"], cfg_file)
    reloaded = load_config(cfg_file)
    assert reloaded.tui_completed_hints == ["send", "commands"]
    assert reloaded.listen_port == 1234
    assert reloaded.tui_theme == "textual-light"


def test_saving_tip_progress_keeps_env_overrides_out_of_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('listen_port = 8080\nallowed_tools = ["shell_execute"]\n')
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9999")
    save_tui_completed_hints(["send", "commands"], cfg_file)
    assert cfg_file.read_text() == (
        'listen_port = 8080\nallowed_tools = [\n    "shell_execute",\n]\n'
        'tui_completed_hints = [\n    "commands",\n    "send",\n]\n'
    )
    monkeypatch.delenv("DROVE_LISTEN_PORT")
    assert load_config(cfg_file).tui_completed_hints == ["commands", "send"]


def test_saving_tip_progress_creates_minimal_file(tmp_path: Path) -> None:
    cfg_file = tmp_path / "nested" / "config.toml"
    save_tui_completed_hints([], cfg_file)
    assert cfg_file.read_text() == "tui_completed_hints = []\n"


def test_set_config_value_patches_one_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drove.config import set_config_value

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('models_dir = "~/m"\nunknown = "kept"\n[llama_server]\nthreads = 8\n')
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    monkeypatch.setenv("DROVE_LLAMA_SERVER__N_GPU_LAYERS", "3")

    set_config_value("llama_server.n_gpu_layers", 10, cfg_file)
    set_config_value("prompt_cache_dir", "~/cache", cfg_file)

    assert tomllib.loads(cfg_file.read_text()) == {
        "models_dir": "~/m",
        "unknown": "kept",
        "llama_server": {"threads": 8, "n_gpu_layers": 10},
        "prompt_cache_dir": str(Path.home() / "cache"),
    }


def test_set_config_value_none_removes_key(tmp_path: Path) -> None:
    from drove.config import set_config_value

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('default_model = "m"\nlisten_port = 1\n')
    set_config_value("default_model", " ", cfg_file)
    assert cfg_file.read_text() == "listen_port = 1\n"


def test_set_config_value_rejects_unknown_key(tmp_path: Path) -> None:
    from drove.config import set_config_value

    cfg_file = tmp_path / "config.toml"
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("nope", 1, cfg_file)
    with pytest.raises(ValueError, match="Unknown config key"):
        set_config_value("llama_server.nope", 1, cfg_file)
    assert not cfg_file.exists()


def test_add_allowed_tool_appends_to_file_list_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drove.config import add_allowed_tool

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('allowed_tools = ["fetch_url"]\n')
    monkeypatch.setenv("DROVE_ALLOWED_TOOLS", '["write_file"]')
    add_allowed_tool("shell_execute", cfg_file)
    add_allowed_tool("shell_execute", cfg_file)
    assert tomllib.loads(cfg_file.read_text()) == {"allowed_tools": ["fetch_url", "shell_execute"]}


def test_write_default_config_ignores_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from drove.config import write_default_config

    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    monkeypatch.setenv("DROVE_LLAMA_N_GPU_LAYERS", "4")
    cfg_file = tmp_path / "new" / "config.toml"
    cfg = write_default_config(cfg_file)
    assert cfg.listen_port == 8080
    data = tomllib.loads(cfg_file.read_text())
    assert data["listen_port"] == 8080
    assert data["llama_server"] == {"n_gpu_layers": -1}
    assert "default_model" not in data
    monkeypatch.delenv("DROVE_LISTEN_PORT")
    monkeypatch.delenv("DROVE_LLAMA_N_GPU_LAYERS")
    assert load_config(cfg_file).model_dump() == cfg.model_dump()


def test_load_config_names_file_and_key(tmp_path: Path) -> None:
    from drove.config import ConfigError

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[llama_server]\nthreads = "x"\n')
    with pytest.raises(ConfigError) as info:
        load_config(cfg_file)
    assert str(cfg_file) in str(info.value) and "llama_server.threads" in str(info.value)


def test_invalid_nested_env_fails_in_load_not_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from drove.config import ConfigError

    monkeypatch.setenv("DROVE_LLAMA_N_GPU_LAYERS", "all")
    with pytest.raises(ConfigError, match="DROVE_LLAMA_N_GPU_LAYERS"):
        load_config(tmp_path / "missing.toml")
