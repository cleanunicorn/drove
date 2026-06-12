"""Tests for backend detection and ASR model-type inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.backend import BACKEND_ASR, BACKEND_LLAMA, detect_backend, infer_asr_model_type
from drove.model_config import ModelConfig


def test_detect_backend_gguf_defaults_to_llama() -> None:
    assert detect_backend(Path("/models/m/model.gguf"), ModelConfig()) == BACKEND_LLAMA


def test_detect_backend_onnx_defaults_to_asr() -> None:
    assert detect_backend(Path("/models/m/encoder-model.onnx"), ModelConfig()) == BACKEND_ASR


def test_detect_backend_explicit_config_wins() -> None:
    cfg = ModelConfig(backend="asr")
    assert detect_backend(Path("/models/m/model.gguf"), cfg) == BACKEND_ASR


def test_detect_backend_invalid_value_raises() -> None:
    cfg = ModelConfig(backend="bogus")
    with pytest.raises(ValueError, match="Unknown backend"):
        detect_backend(Path("/models/m/model.gguf"), cfg)


def test_infer_asr_model_type_from_repo_id() -> None:
    assert (
        infer_asr_model_type("istupakov/parakeet-tdt-0.6b-v3-onnx") == "nemo-parakeet-tdt-0.6b-v3"
    )
    assert (
        infer_asr_model_type("istupakov/parakeet-tdt-0.6b-v2-onnx") == "nemo-parakeet-tdt-0.6b-v2"
    )


def test_infer_asr_model_type_from_bare_name() -> None:
    assert infer_asr_model_type("parakeet-tdt-0.6b-v3") == "nemo-parakeet-tdt-0.6b-v3"


def test_infer_asr_model_type_unknown_returns_none() -> None:
    assert infer_asr_model_type("unsloth/Qwen3-8B-GGUF") is None
