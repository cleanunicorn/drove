"""Tests for global config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from drove.config import load_config, parse_size, resolve_config_path


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
    cfg = load_config(tmp_path / "nonexistent.toml")
    cfg = cfg.model_copy(update={"listen_port": 1234, "speech_timeout_seconds": 2400})
    cfg.save(cfg_file)

    reloaded = load_config(cfg_file)
    assert reloaded.listen_port == 1234
    assert reloaded.speech_timeout_seconds == 2400


@pytest.mark.parametrize("value", [0, -1])
def test_speech_timeout_must_be_positive(tmp_path: Path, value: int) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"speech_timeout_seconds": value}).encode())
    with pytest.raises(ValidationError, match="speech_timeout_seconds"):
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
    with pytest.raises(ValidationError, match="Invalid size"):
        load_config(cfg_file)


def test_max_memory_survives_save_and_reload(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg = load_config(tmp_path / "nonexistent.toml")
    cfg = cfg.model_copy(update={"max_memory": "16GiB"})
    cfg.save(cfg_file)

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
    cfg.save(path)
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
