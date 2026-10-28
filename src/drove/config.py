"""Global configuration loaded from TOML file with env var overrides."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import Field, ValidationError, field_validator
from pydantic_core import ErrorDetails
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
    TomlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "drove" / "config.toml"
LEGACY_CONFIG_PATH = Path.home() / ".config" / "vllama" / "config.toml"
DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "drove" / "models"
DEFAULT_SESSIONS_DIR = Path.home() / ".local" / "share" / "drove" / "sessions"
DEFAULT_OBSERVE_DIR = Path.home() / ".local" / "share" / "drove" / "observe"
DEFAULT_PROMPT_CACHE_DIR = Path.home() / ".local" / "share" / "drove" / "prompt-cache"

# Module-level mutable so load_config() can point to a custom path
_config_path: Path = DEFAULT_CONFIG_PATH

_SIZE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_size(value: str | int) -> int:
    """Parse a human-readable size ('24GB', '16GiB', '512 MB', plain bytes) to bytes.

    Decimal units (KB, MB, GB, TB) use powers of 1000; binary units
    (KiB, MiB, GiB, TiB) use powers of 1024. A bare number means bytes.
    """
    if isinstance(value, int):
        return value
    text = value.strip().lower().replace(" ", "")
    if not text:
        return 0
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]i?b|b)?", text)
    if match is None:
        raise ValueError(
            f"Invalid size '{value}'. Use a number of bytes or a unit suffix, e.g. '24GB', '16GiB'."
        )
    number, unit = match.groups()
    return int(float(number) * _SIZE_UNITS[unit or "b"])


class LlamaServerDefaults(BaseSettings):
    """Default llama-server args applied to all models (overridable per-model)."""

    model_config = SettingsConfigDict(env_prefix="DROVE_LLAMA_")

    n_gpu_layers: int = -1
    threads: int | None = None


ChatHint = Literal["send", "newline", "commands", "details", "stop"]


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
    default_model: str | None = None
    sessions_dir: Path = DEFAULT_SESSIONS_DIR
    observe: bool = False
    observe_dir: Path = DEFAULT_OBSERVE_DIR
    # Observe records older than this are deleted at startup and hourly; 0 = keep forever.
    observe_max_age_days: int = Field(default=30, ge=0)
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    allowed_tools: list[str] = []
    llama_server_bin: str = "llama-server"
    startup_timeout_seconds: int = 300  # max wait for llama-server to become healthy
    speech_timeout_seconds: int = Field(default=600, gt=0)  # read timeout for buffered speech
    idle_timeout_seconds: int = 1800  # 30 minutes
    max_loaded_models: int = 1
    max_memory: str = "0"  # combined memory budget for loaded models, e.g. "24GB"; "0" = unlimited
    # Persist each model's prompt (KV) cache to disk across idle shutdowns, so a
    # woken model does not have to re-process a prompt it already saw. On by
    # default; the TTL is what keeps the cache directory from growing forever.
    prompt_cache: bool = True
    prompt_cache_dir: Path = DEFAULT_PROMPT_CACHE_DIR
    prompt_cache_ttl_seconds: int = 3600  # discard cache files older than this; 0 = never expire
    prompt_cache_timeout_seconds: int = 60  # max wait for a single save/restore call
    llama_server_host: str = "127.0.0.1"
    tui_theme: str = "textual-dark"
    tui_completed_hints: list[ChatHint] = []

    # A factory, not an instance: DROVE_LLAMA_* is then read (and can fail) in
    # load_config() rather than at import time.
    llama_server: LlamaServerDefaults = Field(default_factory=LlamaServerDefaults)

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

    @field_validator("models_dir", "sessions_dir", "observe_dir", "prompt_cache_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        # Raise ValueError, not TypeError/RuntimeError, so a bad path is a validation error.
        try:
            return Path(v).expanduser()
        except TypeError as exc:
            raise ValueError(f"Invalid path {v!r}; use a string such as '~/models'.") from exc
        except RuntimeError as exc:
            raise ValueError(f"Cannot expand '~' in {v!r}: unknown user or home.") from exc

    @field_validator("default_model")
    @classmethod
    def normalize_default_model(cls, value: str | None) -> str | None:
        return value if value is not None and value.strip() else None

    @field_validator("max_memory", mode="before")
    @classmethod
    def validate_max_memory(cls, v: Any) -> str:
        text = str(v)
        parse_size(text)  # raises ValueError on invalid input
        return text

    @property
    def max_memory_bytes(self) -> int:
        return parse_size(self.max_memory)


class _FileLlamaServerDefaults(LlamaServerDefaults):
    """``LlamaServerDefaults`` validated from explicit values only (no ``DROVE_LLAMA_*``)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class _FileConfig(Config):
    """``Config`` validated from explicit values only: no env vars, no TOML source.

    Used to check what is (or would be) in the file, and for true defaults.
    """

    model_config = SettingsConfigDict(toml_file=None)

    llama_server: _FileLlamaServerDefaults = Field(default_factory=_FileLlamaServerDefaults)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class ConfigError(ValueError):
    """A config file or ``DROVE_*`` value that cannot be loaded or saved, said plainly."""

    def __init__(self, message: str, next_step: str | None = None) -> None:
        super().__init__(message)
        self.next_step = next_step


def resolve_config_path(path: Path | None = None) -> Path:
    """Return the config file to use: *path*, else ``$DROVE_CONFIG``, else the default."""
    if path is not None:
        return _expand_config_path(path, "--config")
    env_path = os.environ.get("DROVE_CONFIG")
    if env_path:
        return _expand_config_path(Path(env_path), "DROVE_CONFIG")
    return DEFAULT_CONFIG_PATH


def _expand_config_path(path: Path, source: str) -> Path:
    try:
        return path.expanduser()
    except RuntimeError as exc:
        raise ConfigError(
            f"Cannot expand '~' in config path {str(path)!r} from {source}: unknown user or home.",
            next_step=f"Pass an absolute path in {source}.",
        ) from exc


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file (if it exists), then apply env var overrides.

    Raises ConfigError naming the file or ``DROVE_*`` variable and the key at fault.
    """
    config_path = resolve_config_path(path)
    _migrate_legacy_config(config_path)
    data = read_config_file(config_path)  # surfaces TOML syntax errors with the path
    # Point the TOML source at the requested file by temporarily updating model_config
    Config.model_config["toml_file"] = str(config_path)
    try:
        return Config()
    except (ValidationError, SettingsError) as exc:
        _validate_file_data(data, config_path)  # blame the file first, if it is at fault
        raise _env_error(exc) from exc


def read_config_file(path: Path | None = None) -> dict[str, Any]:
    """Return the raw TOML mapping of the config file ({} when it does not exist)."""
    config_path = resolve_config_path(path)
    if not config_path.exists():
        return {}
    try:
        return tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Config file {config_path} is not valid TOML: {exc}.",
            next_step=f"Fix the syntax in {config_path}, or reset it: drove init --force",
        ) from exc


def set_config_value(key: str, value: Any, path: Path | None = None) -> None:
    """Write one key (``name`` or ``llama_server.name``) into the config file.

    Every other key keeps its file value; env overrides and defaults are never
    written. The merged file is validated first, so an invalid value, or another
    invalid key already in the file, raises ConfigError and leaves the file as is.
    A value that normalizes to None (an empty ``default_model``) removes the key.
    """
    section, _, name = key.rpartition(".")
    model: type[BaseSettings] = _FileLlamaServerDefaults if section else _FileConfig
    if section not in ("", "llama_server") or name not in model.model_fields:
        raise ValueError(f"Unknown config key '{key}'.")
    config_path = resolve_config_path(path)
    data = read_config_file(config_path)
    table = data
    if section:
        nested = data.setdefault(section, {})
        if not isinstance(nested, dict):
            raise ConfigError(f"Config file {config_path}: '{section}' must be a table.")
        table = nested
    table[name] = value
    validated: BaseSettings = _validate_file_data(data, config_path)
    if section:
        validated = getattr(validated, section)
    stored = getattr(validated, name)
    if stored is None:
        del table[name]
    else:
        table[name] = str(stored) if isinstance(stored, Path) else stored
    _write_config_file(data, config_path)


def add_allowed_tool(entry: str, path: Path | None = None) -> None:
    """Append a chat tool grant to the file's ``allowed_tools``.

    *entry* is a token from ``tool_policy.Grant.token()``; bare tool names are
    ignored by the approval policy. Env-provided entries are never copied in.
    """
    tools = read_config_file(path).get("allowed_tools", [])
    if not isinstance(tools, list):
        tools = []
    if entry not in tools:
        set_config_value("allowed_tools", [*tools, entry], path)


def save_tui_completed_hints(hints: list[ChatHint], path: Path | None = None) -> None:
    """Persist chat tip progress without rewriting any other key."""
    set_config_value("tui_completed_hints", sorted(hints), path)


def write_default_config(path: Path | None = None) -> Config:
    """Overwrite the config file with every setting at its built-in default.

    Env overrides are ignored: this is what ``drove init`` writes. Returns the defaults.
    """
    defaults = _FileConfig()
    data: dict[str, Any] = {
        "models_dir": str(defaults.models_dir),
        "sessions_dir": str(defaults.sessions_dir),
        "observe": defaults.observe,
        "observe_dir": str(defaults.observe_dir),
        "observe_max_age_days": defaults.observe_max_age_days,
        "listen_host": defaults.listen_host,
        "listen_port": defaults.listen_port,
        "allowed_tools": defaults.allowed_tools,
        "llama_server_bin": defaults.llama_server_bin,
        "startup_timeout_seconds": defaults.startup_timeout_seconds,
        "speech_timeout_seconds": defaults.speech_timeout_seconds,
        "idle_timeout_seconds": defaults.idle_timeout_seconds,
        "max_loaded_models": defaults.max_loaded_models,
        "max_memory": defaults.max_memory,
        "prompt_cache": defaults.prompt_cache,
        "prompt_cache_dir": str(defaults.prompt_cache_dir),
        "prompt_cache_ttl_seconds": defaults.prompt_cache_ttl_seconds,
        "prompt_cache_timeout_seconds": defaults.prompt_cache_timeout_seconds,
        "llama_server_host": defaults.llama_server_host,
        "tui_theme": defaults.tui_theme,
        "tui_completed_hints": defaults.tui_completed_hints,
        "llama_server": {
            k: v for k, v in defaults.llama_server.model_dump().items() if v is not None
        },
    }
    _write_config_file(data, resolve_config_path(path))
    return defaults


def _write_config_file(data: dict[str, Any], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(tomli_w.dumps(data).encode())


def _validate_file_data(data: dict[str, Any], config_path: Path) -> _FileConfig:
    """Validate file values alone (no env); raise ConfigError naming the bad keys."""
    try:
        return _FileConfig(**data)
    except ValidationError as exc:
        problems = "; ".join(f"{_error_key(err)}: {_error_message(err)}" for err in exc.errors())
        raise ConfigError(
            f"Invalid config file {config_path}: {problems}.",
            next_step=f"Set a valid value with drove config <key> <value>, or edit {config_path}.",
        ) from exc


def _env_error(exc: ValidationError | SettingsError) -> ConfigError:
    """Describe a load failure the file is not responsible for: a bad ``DROVE_*`` value."""
    if isinstance(exc, ValidationError):
        keys = [_error_key(err) for err in exc.errors()]
        details = "; ".join(_error_message(err) for err in exc.errors())
    else:
        match = re.search(r'field "(\w+)"', str(exc))
        keys = [match.group(1)] if match else []
        details = "not valid JSON" if match else str(exc)
    names = list(dict.fromkeys(name for key in keys for name in _env_names(key))) or ["DROVE_*"]
    return ConfigError(
        f"Invalid environment variable {', '.join(names)} ({', '.join(keys) or 'unknown key'}): "
        f"{details}.",
        next_step="Fix or unset the variable; DROVE_* values override config.toml.",
    )


def _error_key(err: ErrorDetails) -> str:
    """Dotted config key of a validation error (``llama_server.`` added for its fields)."""
    key = ".".join(str(part) for part in err["loc"])
    return f"llama_server.{key}" if key in LlamaServerDefaults.model_fields else key


def _error_message(err: ErrorDetails) -> str:
    return err["msg"].removeprefix("Value error, ").rstrip(".")


def _env_names(key: str) -> list[str]:
    """Return the set ``DROVE_*`` variables that can supply *key* (env names ignore case)."""
    root, _, rest = key.partition(".")
    candidates = [f"DROVE_{root}"]  # the whole field, e.g. a JSON list or object
    if root == "llama_server" and rest:
        leaf = rest.partition(".")[0]
        candidates = [f"DROVE_LLAMA_SERVER__{leaf}", f"DROVE_LLAMA_{leaf}", *candidates]
    present = {name.upper(): name for name in os.environ}
    return [present[c.upper()] for c in candidates if c.upper() in present]


def _migrate_legacy_config(config_path: Path) -> None:
    """Move ~/.config/vllama/config.toml to ~/.config/drove/config.toml on first run."""
    if config_path != DEFAULT_CONFIG_PATH:
        return
    if config_path.exists() or not LEGACY_CONFIG_PATH.exists():
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(LEGACY_CONFIG_PATH.read_text())
    print(f"[drove] migrated config from {LEGACY_CONFIG_PATH} to {config_path}")
