"""Tests for global config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from drove.config import load_config, parse_size


def test_defaults(tmp_path: Path) -> None:
    """Config has sensible defaults when no file or env vars exist."""
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.listen_port == 8080
    assert cfg.idle_timeout_seconds == 1800
    assert cfg.llama_server_bin == "llama-server"


def test_toml_overrides_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"listen_port": 9090, "idle_timeout_seconds": 600}).encode())
    cfg = load_config(cfg_file)
    assert cfg.listen_port == 9090
    assert cfg.idle_timeout_seconds == 600


def test_env_var_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"listen_port": 9090}).encode())
    monkeypatch.setenv("DROVE_LISTEN_PORT", "7777")
    cfg = load_config(cfg_file)
    assert cfg.listen_port == 7777


def test_models_dir_expands_home(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(tomli_w.dumps({"models_dir": "~/mymodels"}).encode())
    cfg = load_config(cfg_file)
    assert not str(cfg.models_dir).startswith("~")
    assert cfg.models_dir == Path.home() / "mymodels"


def test_save_and_reload(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg = load_config(tmp_path / "nonexistent.toml")
    cfg = cfg.model_copy(update={"listen_port": 1234})
    cfg.save(cfg_file)

    reloaded = load_config(cfg_file)
    assert reloaded.listen_port == 1234


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
