"""Tests for global config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from vllama.config import load_config


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
    monkeypatch.setenv("VLLAMA_LISTEN_PORT", "7777")
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
