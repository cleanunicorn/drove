"""Tests for global config loading."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import tomli_w

from vllama.config import DEFAULT_CONFIG_PATH, Config, load_config


@pytest.fixture(autouse=True)
def _reset_config_path() -> Generator:
    """Reset Config.model_config toml_file to default before every test."""
    Config.model_config["toml_file"] = str(DEFAULT_CONFIG_PATH)
    yield
    Config.model_config["toml_file"] = str(DEFAULT_CONFIG_PATH)


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


def test_agents_permissions_default_empty(tmp_path: Path) -> None:
    """Default config has no per-tool permission overrides."""
    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.agents.permissions == {}


def test_agents_permissions_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps(
            {
                "agents": {
                    "permissions": {
                        "write_file": "auto",
                        "bash": "deny",
                    },
                },
            }
        ).encode()
    )
    cfg = load_config(path)
    assert cfg.agents.permissions == {"write_file": "auto", "bash": "deny"}


def test_agents_permissions_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var overrides TOML for agents.permissions."""
    path = tmp_path / "c.toml"
    path.write_bytes(tomli_w.dumps({"agents": {"permissions": {"bash": "prompt"}}}).encode())
    monkeypatch.setenv("VLLAMA_AGENTS__PERMISSIONS", '{"bash": "auto"}')
    cfg = load_config(path)
    assert cfg.agents.permissions == {"bash": "auto"}


def test_agents_permissions_invalid_value_rejected(tmp_path: Path) -> None:
    """Unknown decision value raises at load time."""
    path = tmp_path / "c.toml"
    path.write_bytes(tomli_w.dumps({"agents": {"permissions": {"bash": "ignore"}}}).encode())
    with pytest.raises(Exception):
        load_config(path)
