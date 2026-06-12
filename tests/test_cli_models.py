"""Tests for the `drove models` CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.cli.models import _detect_capabilities, _select_quant_variant
from drove.downloader import DownloadPlan


def _make_model(models_dir: Path, name: str, *extra_files: str) -> Path:
    """Create a model directory with a primary .gguf file and any extra files."""
    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    primary = model_dir / f"{name}.gguf"
    primary.write_bytes(b"")
    for fname in extra_files:
        (model_dir / fname).write_bytes(b"")
    return primary


def test_detect_capabilities_empty_for_text_only_model(tmp_path: Path) -> None:
    primary = _make_model(tmp_path, "text-model")
    assert _detect_capabilities(primary) == []


def test_detect_capabilities_vision_from_mmproj_file(tmp_path: Path) -> None:
    """A sibling mmproj-*.gguf file should be enough to signal vision support."""
    primary = _make_model(tmp_path, "vision-model", "mmproj-BF16.gguf")
    assert _detect_capabilities(primary) == ["vision"]


def test_detect_capabilities_vision_from_sidecar_config(tmp_path: Path) -> None:
    """An ``mmproj`` entry in the sidecar TOML also signals vision support."""
    primary = _make_model(tmp_path, "vision-cfg")
    sidecar = primary.with_suffix(".toml")
    sidecar.write_text('mmproj = "mmproj-BF16.gguf"\n')
    assert _detect_capabilities(primary) == ["vision"]


def test_detect_capabilities_ignores_unrelated_gguf(tmp_path: Path) -> None:
    """A regular .gguf file (shards, etc.) should not be mistaken for mmproj."""
    primary = _make_model(tmp_path, "sharded", "sharded-00002-of-00003.gguf")
    assert _detect_capabilities(primary) == []


def test_detect_capabilities_mmproj_filename_case_insensitive(tmp_path: Path) -> None:
    primary = _make_model(tmp_path, "upper", "MMPROJ-f16.gguf")
    assert _detect_capabilities(primary) == ["vision"]


# ── quant variant selection ──────────────────────────────────────────────────

_ONNX_ALL = {
    "encoder-model.onnx": 2_400,
    "encoder-model.int8.onnx": 600,
    "decoder_joint-model.onnx": 80,
    "decoder_joint-model.int8.onnx": 20,
}
_ONNX_DEFAULT = {"encoder-model.onnx": 2_400, "decoder_joint-model.onnx": 80}
_ONNX_INT8 = {"encoder-model.int8.onnx": 600, "decoder_joint-model.int8.onnx": 20}


def _onnx_plan() -> DownloadPlan:
    return DownloadPlan(
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        files=dict(_ONNX_DEFAULT),
        local_name="istupakov/parakeet-tdt-0.6b-v3-onnx",
        sharded=False,
        onnx_files=dict(_ONNX_ALL),
    )


def _choose(monkeypatch: pytest.MonkeyPatch, answer: str | None) -> None:
    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", lambda quants: answer)


def test_select_quant_variant_onnx_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant == "int8"
    assert plan.files == _ONNX_INT8
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx:int8"


def test_select_quant_variant_onnx_default_keeps_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "default")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant is None
    assert plan.files == _ONNX_DEFAULT
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx"


def test_select_quant_variant_onnx_all_takes_every_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _choose(monkeypatch, None)
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant is None
    assert plan.files == _ONNX_ALL


def test_select_quant_variant_onnx_keeps_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    plan.local_name = "my-parakeet"
    quant = _select_quant_variant(plan, "my-parakeet")
    assert quant == "int8"
    assert plan.local_name == "my-parakeet"


def test_select_quant_variant_onnx_single_variant_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(quants: dict[str, int]) -> str | None:
        raise AssertionError("prompt should not be shown")

    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", boom)
    plan = DownloadPlan(
        repo_id="istupakov/whisper-base-onnx",
        files=dict(_ONNX_DEFAULT),
        local_name="istupakov/whisper-base-onnx",
        sharded=False,
        onnx_files=dict(_ONNX_DEFAULT),
    )
    assert _select_quant_variant(plan, None) is None
    assert plan.files == _ONNX_DEFAULT


def test_select_quant_variant_gguf_choice_filters_and_renames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _choose(monkeypatch, "Q4_K_M")
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files={"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200},
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    quant = _select_quant_variant(plan, None)
    assert quant == "Q4_K_M"
    assert plan.files == {"Qwen3-8B-Q4_K_M.gguf": 100}
    assert plan.local_name == "unsloth/Qwen3-8B-GGUF:Q4_K_M"


def test_select_quant_variant_gguf_all_keeps_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, None)
    files = {"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200}
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files=dict(files),
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    assert _select_quant_variant(plan, None) is None
    assert plan.files == files
