"""Per-model configuration stored as sidecar TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """llama-server parameters for a specific model.

    Keys map to llama-server CLI flags (snake_case → --kebab-case).
    See: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
    """

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

    # Flash attention
    flash_attn: bool | None = None

    # Rope scaling
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    # Quantization
    cache_type_k: str | None = None
    cache_type_v: str | None = None

    def to_llama_args(self) -> list[str]:
        """Convert config to llama-server CLI arguments."""
        args: list[str] = []
        for field, value in self.model_dump(exclude_none=True).items():
            flag = "--" + field.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.extend([flag, str(value)])
        return args

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


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
    """Write per-model config to sidecar TOML file."""
    cfg_path = config_path_for_model(model_path)
    cfg_path.write_bytes(tomli_w.dumps(config.to_dict()).encode())


def set_model_config_key(model_path: Path, key: str, value: str) -> ModelConfig:
    """Set a single key in the model config, coercing the string value to the right type."""
    config = load_model_config(model_path)
    fields = ModelConfig.model_fields

    if key not in fields:
        valid = ", ".join(sorted(fields.keys()))
        raise ValueError(f"Unknown config key '{key}'. Valid keys: {valid}")

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
