"""Tests for per-model config."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.model_config import (
    ModelConfig,
    load_model_config,
    save_model_config,
    set_model_config_key,
)


def fake_model(tmp_path: Path, name: str = "mymodel") -> Path:
    p = tmp_path / f"{name}.gguf"
    p.write_bytes(b"")  # empty placeholder
    return p


def test_defaults_when_no_sidecar(tmp_path: Path) -> None:
    model = fake_model(tmp_path)
    cfg = load_model_config(model)
    assert cfg.ctx_size is None
    assert cfg.n_gpu_layers is None


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = fake_model(tmp_path)
    cfg = ModelConfig(ctx_size=4096, n_gpu_layers=32, flash_attn="on")
    save_model_config(model, cfg)

    loaded = load_model_config(model)
    assert loaded.ctx_size == 4096
    assert loaded.n_gpu_layers == 32
    assert loaded.flash_attn == "on"


def test_to_llama_args(tmp_path: Path) -> None:
    cfg = ModelConfig(ctx_size=8192, n_gpu_layers=-1, flash_attn="on", temp=0.7)
    args = cfg.to_llama_args()
    assert "--ctx-size" in args
    assert "8192" in args
    assert "--flash-attn" in args
    assert "on" in args
    assert "--temp" in args
    assert "0.7" in args


def test_set_model_config_key_int(tmp_path: Path) -> None:
    model = fake_model(tmp_path)
    updated = set_model_config_key(model, "ctx_size", "8192")
    assert updated.ctx_size == 8192
    assert load_model_config(model).ctx_size == 8192


def test_set_model_config_key_bool(tmp_path: Path) -> None:
    model = fake_model(tmp_path)
    updated = set_model_config_key(model, "flash_attn", "on")
    assert updated.flash_attn == "on"


def test_set_model_config_key_invalid(tmp_path: Path) -> None:
    model = fake_model(tmp_path)
    with pytest.raises(ValueError, match="Unknown config key"):
        set_model_config_key(model, "nonexistent_key", "value")
