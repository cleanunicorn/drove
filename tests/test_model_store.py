"""Tests for ModelStore — the single model-resolution authority."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.model_store import ModelBackend, ModelStore

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_gguf(models_dir: Path, *parts: str) -> Path:
    """Create an empty .gguf file at models_dir / *parts."""
    p = models_dir.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


# ── resolve ───────────────────────────────────────────────────────────────────


def test_resolve_flat_legacy_file(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mymodel.gguf")
    assert ModelStore(tmp_path).resolve("mymodel") == tmp_path / "mymodel.gguf"


def test_resolve_subdirectory_model(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "mymodel", "mymodel.gguf")
    assert ModelStore(tmp_path).resolve("mymodel") == primary


def test_resolve_namespaced_model(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "org", "mymodel", "mymodel.gguf")
    assert ModelStore(tmp_path).resolve("org/mymodel") == primary


def test_resolve_nested_gguf_via_rglob(tmp_path: Path) -> None:
    """A GGUF nested one level deep inside the model dir must be found."""
    primary = _make_gguf(tmp_path, "mymodel", "weights", "shard.gguf")
    assert ModelStore(tmp_path).resolve("mymodel") == primary


def test_resolve_alphabetically_first_shard(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mymodel", "shard-002.gguf")
    first = _make_gguf(tmp_path, "mymodel", "shard-001.gguf")
    assert ModelStore(tmp_path).resolve("mymodel") == first


def test_resolve_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="mymodel"):
        ModelStore(tmp_path).resolve("mymodel")


def test_resolve_exact_local_path_takes_priority_over_alias(tmp_path: Path) -> None:
    """org/repo directory must win over any HF alias mapping to a different model."""
    # Explicit local directory: org/repo/model.gguf
    primary = _make_gguf(tmp_path, "org", "repo", "model.gguf")
    # A sidecar TOML that maps the same "org/repo" repo_id to a different local name
    _make_gguf(tmp_path, "other-model", "other.gguf")
    (tmp_path / "other-model" / "other.toml").write_bytes(b'[download]\nrepo_id = "org/repo"\n')
    # resolve() must return the explicit directory, not redirect via alias
    assert ModelStore(tmp_path).resolve("org/repo") == primary


def test_resolve_safetensors_returns_primary(tmp_path: Path) -> None:
    model_dir = tmp_path / "unconverted"
    model_dir.mkdir()
    primary = model_dir / "model.safetensors"
    primary.write_bytes(b"")
    assert ModelStore(tmp_path).resolve("unconverted") == primary


def test_resolve_backend_detects_mlx_for_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx-model"
    model_dir.mkdir()
    (model_dir / "weights.safetensors").write_bytes(b"")
    assert ModelStore(tmp_path).resolve_backend("mlx-model") is ModelBackend.MLX


# ── find_root ────────────────────────────────────────────────────────────────


def test_find_root_returns_directory(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mymodel", "mymodel.gguf")
    assert ModelStore(tmp_path).find_root("mymodel") == tmp_path / "mymodel"


def test_find_root_returns_flat_file(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mymodel.gguf")
    assert ModelStore(tmp_path).find_root("mymodel") == tmp_path / "mymodel.gguf"


def test_find_root_returns_none_for_missing(tmp_path: Path) -> None:
    assert ModelStore(tmp_path).find_root("nope") is None


# ── list ─────────────────────────────────────────────────────────────────────


def test_list_empty_dir(tmp_path: Path) -> None:
    assert ModelStore(tmp_path).list() == []


def test_list_missing_dir(tmp_path: Path) -> None:
    assert ModelStore(tmp_path / "nonexistent").list() == []


def test_list_flat_legacy_model(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mymodel.gguf")
    entries = ModelStore(tmp_path).list()
    assert len(entries) == 1
    assert entries[0].name == "mymodel"
    assert entries[0].primary == tmp_path / "mymodel.gguf"


def test_list_subdirectory_model(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "mymodel", "mymodel.gguf")
    entries = ModelStore(tmp_path).list()
    assert len(entries) == 1
    assert entries[0].name == "mymodel"
    assert entries[0].primary == primary


def test_list_namespaced_model(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "org", "mymodel", "mymodel.gguf")
    entries = ModelStore(tmp_path).list()
    assert len(entries) == 1
    assert entries[0].name == "org/mymodel"
    assert entries[0].primary == primary


def test_list_multiple_models_sorted(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "b-model", "b.gguf")
    _make_gguf(tmp_path, "a-model", "a.gguf")
    names = [e.name for e in ModelStore(tmp_path).list()]
    assert names == ["a-model", "b-model"]


def test_list_total_bytes(tmp_path: Path) -> None:
    p = _make_gguf(tmp_path, "mymodel", "mymodel.gguf")
    p.write_bytes(b"x" * 1024)
    entries = ModelStore(tmp_path).list()
    assert entries[0].total_bytes == 1024


# ── complete ─────────────────────────────────────────────────────────────────


def test_complete_prefix_match(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "qwen3", "q.gguf")
    _make_gguf(tmp_path, "llama3", "l.gguf")
    assert ModelStore(tmp_path).complete("qw") == ["qwen3"]


def test_complete_case_insensitive(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "Qwen3", "q.gguf")
    assert ModelStore(tmp_path).complete("qw") == ["Qwen3"]


def test_complete_empty_prefix_returns_all(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "alpha", "a.gguf")
    _make_gguf(tmp_path, "beta", "b.gguf")
    assert set(ModelStore(tmp_path).complete("")) == {"alpha", "beta"}


def test_complete_no_match(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "alpha", "a.gguf")
    assert ModelStore(tmp_path).complete("xyz") == []
