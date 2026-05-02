"""Global configuration loaded from TOML file with env var overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w
from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "drove" / "config.toml"
LEGACY_CONFIG_PATH = Path.home() / ".config" / "vllama" / "config.toml"
DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "drove" / "models"
DEFAULT_SESSIONS_DIR = Path.home() / ".local" / "share" / "drove" / "sessions"
DEFAULT_OBSERVE_DIR = Path.home() / ".local" / "share" / "drove" / "observe"

# Module-level mutable so load_config() can point to a custom path
_config_path: Path = DEFAULT_CONFIG_PATH


class LlamaServerDefaults(BaseSettings):
    """Default llama-server args applied to all models (overridable per-model)."""

    model_config = SettingsConfigDict(env_prefix="DROVE_LLAMA_")

    n_gpu_layers: int = -1
    threads: int | None = None


class Config(BaseSettings):
    """Global drove configuration.

    Source priority (highest → lowest):
      1. Environment variables (DROVE_*)
      2. TOML config file
      3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="DROVE_",
        env_nested_delimiter="__",
        extra="ignore",
        toml_file=str(DEFAULT_CONFIG_PATH),
    )

    models_dir: Path = DEFAULT_MODELS_DIR
    sessions_dir: Path = DEFAULT_SESSIONS_DIR
    observe: bool = False
    observe_dir: Path = DEFAULT_OBSERVE_DIR
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    listen_port_https: int = 8443
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None
    allowed_tools: list[str] = []
    llama_server_bin: str = "llama-server"
    startup_timeout_seconds: int = 300  # max wait for llama-server to become healthy
    idle_timeout_seconds: int = 1800  # 30 minutes
    max_loaded_models: int = 1
    llama_server_host: str = "127.0.0.1"
    tui_theme: str = "textual-dark"

    llama_server: LlamaServerDefaults = LlamaServerDefaults()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init kwargs > env vars > TOML file > defaults
        return (init_settings, env_settings, TomlConfigSettingsSource(settings_cls))

    @field_validator("models_dir", "sessions_dir", "observe_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        return Path(v).expanduser()

    @field_validator("ssl_certfile", "ssl_keyfile", mode="before")
    @classmethod
    def expand_optional_path(cls, v: Any) -> Path | None:
        if v is None or v == "":
            return None
        return Path(v).expanduser()

    def save(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "models_dir": str(self.models_dir),
            "sessions_dir": str(self.sessions_dir),
            "observe": self.observe,
            "observe_dir": str(self.observe_dir),
            "listen_host": self.listen_host,
            "listen_port": self.listen_port,
            "listen_port_https": self.listen_port_https,
            "ssl_certfile": str(self.ssl_certfile) if self.ssl_certfile else "",
            "ssl_keyfile": str(self.ssl_keyfile) if self.ssl_keyfile else "",
            "allowed_tools": self.allowed_tools,
            "llama_server_bin": self.llama_server_bin,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "max_loaded_models": self.max_loaded_models,
            "llama_server_host": self.llama_server_host,
            "tui_theme": self.tui_theme,
            "llama_server": {
                k: v for k, v in self.llama_server.model_dump().items() if v is not None
            },
        }
        path.write_bytes(tomli_w.dumps(data).encode())


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file (if it exists), then apply env var overrides."""
    config_path = path or DEFAULT_CONFIG_PATH
    _migrate_legacy_config(config_path)
    # Point the TOML source at the requested file by temporarily updating model_config
    Config.model_config["toml_file"] = str(config_path)  # type: ignore[index]
    return Config()


def _migrate_legacy_config(config_path: Path) -> None:
    """Move ~/.config/vllama/config.toml to ~/.config/drove/config.toml on first run."""
    if config_path != DEFAULT_CONFIG_PATH:
        return
    if config_path.exists() or not LEGACY_CONFIG_PATH.exists():
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(LEGACY_CONFIG_PATH.read_text())
    print(f"[drove] migrated config from {LEGACY_CONFIG_PATH} to {config_path}")
