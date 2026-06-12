"""Per-model configuration stored as sidecar TOML files."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ConfigDict, field_validator


class DownloadInfo(BaseModel):
    """Metadata about how a model was downloaded."""

    repo_id: str
    files: list[str]
    quant: str | None = None


#: Config keys that drove consumes itself; excluded from llama-server args.
_DROVE_ONLY_FIELDS = frozenset({"backend", "asr_model", "asr_quantization"})


class ModelConfig(BaseModel):
    """Per-model parameters.

    Most keys map to llama-server CLI flags (snake_case → --kebab-case).
    See: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

    Keys in ``_DROVE_ONLY_FIELDS`` configure drove itself (backend selection
    and the built-in ASR worker) and are never forwarded to llama-server.
    """

    model_config = ConfigDict(extra="ignore")

    # Context and memory
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    main_gpu: int | None = None
    tensor_split: str | None = None

    # Batching
    batch_size: int | None = None
    ubatch_size: int | None = None
    n_parallel: int | None = None

    # Sampling defaults
    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # Performance
    threads: int | None = None
    threads_batch: int | None = None

    # Flash attention (on, off, auto)
    flash_attn: str | None = None

    @field_validator("flash_attn", mode="before")
    @classmethod
    def _coerce_flash_attn(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return "on" if v else "off"
        return str(v)

    # Rope scaling
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    # Quantization
    cache_type_k: str | None = None
    cache_type_v: str | None = None

    # Multimodal
    mmproj: str | None = None

    # Drove-specific settings (never passed to llama-server)
    backend: str | None = None  # "llama" (default) or "asr"
    asr_model: str | None = None  # onnx-asr model type, e.g. "nemo-parakeet-tdt-0.6b-v3"
    asr_quantization: str | None = None  # e.g. "int8"

    def to_llama_args(self) -> list[str]:
        """Convert config to llama-server CLI arguments."""
        args: list[str] = []
        for field, value in self.model_dump(exclude_none=True).items():
            if field in _DROVE_ONLY_FIELDS:
                continue
            flag = "--" + field.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])
        return args

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


# Safe-token shape for values handed to the ASR worker's command line.
# Not a whitelist: onnx-asr supports more model types than drove's known-repo
# table, and setting asr_model manually is the documented escape hatch.
_ASR_VALUE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_drove_key(key: str, value: str) -> None:
    """Validate drove-specific config values when they are set via the CLI.

    Fails fast with a helpful message instead of leaving a value that only
    blows up when the backend subprocess starts.
    """
    if key == "backend":
        from drove.backend import VALID_BACKENDS

        if value.strip().lower() not in VALID_BACKENDS:
            valid = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(f"Unknown backend '{value}'. Valid backends: {valid}")
    elif key in ("asr_model", "asr_quantization") and not _ASR_VALUE_RE.match(value):
        raise ValueError(
            f"Invalid value for '{key}': '{value}'. "
            "Use letters, digits, dots, dashes, and underscores only "
            "(e.g. nemo-parakeet-tdt-0.6b-v3, int8)."
        )


GLOBAL_CONFIG_FILENAME = "_global.toml"


def global_config_path(models_dir: Path) -> Path:
    """Return the path to the global model config file in the models directory."""
    return models_dir / GLOBAL_CONFIG_FILENAME


def load_global_model_config(models_dir: Path) -> ModelConfig:
    """Load global model config from _global.toml, returning defaults if absent."""
    cfg_path = global_config_path(models_dir)
    if not cfg_path.exists():
        return ModelConfig()
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    return ModelConfig(**data)


def save_global_model_config(models_dir: Path, config: ModelConfig) -> None:
    """Write global model config to _global.toml."""
    cfg_path = global_config_path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(tomli_w.dumps(config.to_dict()).encode())


def set_global_model_config_key(models_dir: Path, key: str, value: str) -> ModelConfig:
    """Set a single key in the global model config, coercing the string value."""
    config = load_global_model_config(models_dir)
    fields = ModelConfig.model_fields

    if key not in fields:
        valid = ", ".join(sorted(fields.keys()))
        raise ValueError(f"Unknown config key '{key}'. Valid keys: {valid}")

    _validate_drove_key(key, value)

    annotation = fields[key].annotation
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        raise ValueError(f"Cannot set NoneType field '{key}'")

    args = getattr(annotation, "__args__", None)
    inner = args[0] if args else annotation

    if inner is bool:
        coerced: Any = value.lower() in ("1", "true", "yes")
    elif inner is int:
        coerced = int(value)
    elif inner is float:
        coerced = float(value)
    else:
        coerced = value

    updated = config.model_copy(update={key: coerced})
    save_global_model_config(models_dir, updated)
    return updated


def config_path_for_model(model_path: Path) -> Path:
    """Return the sidecar config path for a model file."""
    return model_path.with_suffix(".toml")


def load_model_config(model_path: Path) -> ModelConfig:
    """Load per-model config from sidecar TOML, returning defaults if absent."""
    cfg_path = config_path_for_model(model_path)
    if not cfg_path.exists():
        return ModelConfig()
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    return ModelConfig(**data)


def save_model_config(model_path: Path, config: ModelConfig) -> None:
    """Write per-model config to sidecar TOML file, preserving [download] section."""
    cfg_path = config_path_for_model(model_path)

    # Preserve existing [download] section if present
    existing: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            existing = tomllib.load(f)

    data = config.to_dict()
    if "download" in existing:
        data["download"] = existing["download"]

    cfg_path.write_bytes(tomli_w.dumps(data).encode())


def set_model_config_key(model_path: Path, key: str, value: str) -> ModelConfig:
    """Set a single key in the model config, coercing the string value to the right type."""
    config = load_model_config(model_path)
    fields = ModelConfig.model_fields

    if key not in fields:
        valid = ", ".join(sorted(fields.keys()))
        raise ValueError(f"Unknown config key '{key}'. Valid keys: {valid}")

    _validate_drove_key(key, value)

    annotation = fields[key].annotation
    # Resolve Optional[X] → X
    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        raise ValueError(f"Cannot set NoneType field '{key}'")

    args = getattr(annotation, "__args__", None)
    inner = args[0] if args else annotation

    if inner is bool:
        coerced: Any = value.lower() in ("1", "true", "yes")
    elif inner is int:
        coerced = int(value)
    elif inner is float:
        coerced = float(value)
    else:
        coerced = value

    updated = config.model_copy(update={key: coerced})
    save_model_config(model_path, updated)
    return updated


def load_download_info(model_path: Path) -> DownloadInfo | None:
    """Load download metadata from sidecar TOML, or None if absent."""
    cfg_path = config_path_for_model(model_path)
    if not cfg_path.exists():
        return None
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    dl = data.get("download")
    if not dl:
        return None
    return DownloadInfo(**dl)


def resolve_model_alias(models_dir: Path, ref: str) -> str | None:
    """Resolve a HuggingFace reference (org/repo or org/repo:quant) to a local model name.

    Scans sidecar TOML files (recursively) for matching repo_id and optional
    quant tag.  Returns the local model name, or None if no match found.
    """
    # Parse optional quant tag
    quant: str | None
    if ":" in ref:
        repo_id, quant_tag = ref.rsplit(":", 1)
        quant = quant_tag.strip().lower() or None
    else:
        repo_id, quant = ref, None

    for p in sorted(models_dir.rglob("*.toml")):
        if p.name == GLOBAL_CONFIG_FILENAME:
            continue
        try:
            with p.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        dl = data.get("download")
        if not dl or dl.get("repo_id") != repo_id:
            continue
        stored_quant = dl.get("quant", "")
        if quant is None or (stored_quant and stored_quant.lower() == quant):
            # Model name is the parent directory relative to models_dir,
            # or the TOML stem for legacy flat files.
            if p.parent == models_dir:
                return p.stem
            return str(p.parent.relative_to(models_dir))
    return None


def save_download_info(model_path: Path, info: DownloadInfo) -> None:
    """Write download metadata to the [download] section of the sidecar TOML."""
    cfg_path = config_path_for_model(model_path)

    existing: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            existing = tomllib.load(f)

    existing["download"] = info.model_dump(exclude_none=True)
    cfg_path.write_bytes(tomli_w.dumps(existing).encode())
