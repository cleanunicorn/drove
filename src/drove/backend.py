"""Backend selection for models: llama-server vs the built-in ASR worker."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drove.model_config import ModelConfig

BACKEND_LLAMA = "llama"
BACKEND_ASR = "asr"

VALID_BACKENDS = frozenset({BACKEND_LLAMA, BACKEND_ASR})


def detect_backend(model_path: Path, model_cfg: ModelConfig) -> str:
    """Return the backend for a model.

    An explicit ``backend`` key in the model config wins; otherwise the
    primary file extension decides (``.onnx`` → ASR worker, anything else →
    llama-server).
    """
    if model_cfg.backend:
        backend = model_cfg.backend.strip().lower()
        if backend not in VALID_BACKENDS:
            valid = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(f"Unknown backend '{model_cfg.backend}'. Valid backends: {valid}")
        return backend
    if model_path.suffix.lower() == ".onnx":
        return BACKEND_ASR
    return BACKEND_LLAMA


# onnx-asr model types keyed by normalized repo/directory stem (lowercase,
# trailing "-onnx" stripped).  See https://github.com/istupakov/onnx-asr for
# the full list of supported architectures.
_ASR_MODEL_TYPES = {
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-ctc-0.6b": "nemo-parakeet-ctc-0.6b",
    "parakeet-rnnt-0.6b": "nemo-parakeet-rnnt-0.6b",
    "gigaam-v2-ctc": "gigaam-v2-ctc",
    "gigaam-v2-rnnt": "gigaam-v2-rnnt",
    "whisper-base": "whisper-base",
}


def infer_asr_model_type(ref: str) -> str | None:
    """Infer the onnx-asr model type from a HuggingFace repo id or model name.

    >>> infer_asr_model_type("istupakov/parakeet-tdt-0.6b-v3-onnx")
    'nemo-parakeet-tdt-0.6b-v3'
    """
    stem = ref.rsplit("/", 1)[-1].strip().lower().removesuffix("-onnx")
    return _ASR_MODEL_TYPES.get(stem)
