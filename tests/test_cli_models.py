"""Tests for the `drove models` CLI helpers."""

from __future__ import annotations

from pathlib import Path

from drove.cli.models import _detect_capabilities


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
