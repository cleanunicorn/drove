"""Tests for ModelStore — the single model-resolution authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drove.model_store import ModelStore, validate_model_name

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


def test_resolve_non_gguf_formats_not_returned(tmp_path: Path) -> None:
    """resolve() must return None (→ FileNotFoundError) for non-GGUF-only dirs."""
    model_dir = tmp_path / "unconverted"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        ModelStore(tmp_path).resolve("unconverted")


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


# ── ONNX (ASR) models ────────────────────────────────────────────────────────


def _make_onnx_model(models_dir: Path, *parts: str) -> Path:
    """Create an ONNX ASR model dir with encoder/decoder/vocab files."""
    d = models_dir.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "encoder-model.onnx").write_bytes(b"")
    (d / "decoder_joint-model.onnx").write_bytes(b"")
    (d / "vocab.txt").write_text("a\nb\n")
    return d


def test_resolve_onnx_model_dir(tmp_path: Path) -> None:
    _make_onnx_model(tmp_path, "parakeet")
    primary = ModelStore(tmp_path).resolve("parakeet")
    assert primary.suffix == ".onnx"
    assert primary.name == "decoder_joint-model.onnx"  # sorted-first onnx


def test_resolve_prefers_gguf_over_onnx(tmp_path: Path) -> None:
    d = _make_onnx_model(tmp_path, "mixed")
    (d / "model.gguf").write_bytes(b"")
    primary = ModelStore(tmp_path).resolve("mixed")
    assert primary.suffix == ".gguf"


def test_resolve_namespaced_onnx_model(tmp_path: Path) -> None:
    _make_onnx_model(tmp_path, "istupakov", "parakeet-tdt-0.6b-v3-onnx")
    primary = ModelStore(tmp_path).resolve("istupakov/parakeet-tdt-0.6b-v3-onnx")
    assert primary.suffix == ".onnx"


def test_list_includes_onnx_model_with_onnx_primary(tmp_path: Path) -> None:
    _make_onnx_model(tmp_path, "parakeet")
    entries = ModelStore(tmp_path).list()
    assert len(entries) == 1
    assert entries[0].name == "parakeet"
    assert entries[0].primary.suffix == ".onnx"


def test_complete_includes_onnx_models(tmp_path: Path) -> None:
    _make_onnx_model(tmp_path, "parakeet")
    assert ModelStore(tmp_path).complete("para") == ["parakeet"]


# ── name validation and containment ──────────────────────────────────────────

_VALID_NAMES = [
    "mymodel",
    "org/mymodel",
    "unsloth/Qwen3-8B-GGUF:Q4_K_M",
    "model..v2",
    "a.b_c-d",
]

_INVALID_NAMES = [
    "",
    ".",
    "..",
    "/",
    "/abs",
    "/abs/path/model",
    "../x",
    "../../some/dir",
    "a/../../b",
    "sub/..",
    "./qwen",
    "qwen/",
    "a//b",
    "a/./b",
    "a/b/c",
    "a\\b",
    "C:\\x",
    "my model",
    " qwen",
    "qwen\n",
    "nul\x00",
    "%2e%2e",
    "-flag",
    ".hidden",
]


@pytest.mark.parametrize("name", _VALID_NAMES)
def test_validate_model_name_accepts_safe_names(name: str) -> None:
    assert validate_model_name(name) == name


@pytest.mark.parametrize("name", _INVALID_NAMES)
def test_validate_model_name_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        validate_model_name(name)


@pytest.mark.parametrize("value", [None, 1, True, ["m"], {"m": 1}, b"m"])
def test_validate_model_name_rejects_non_strings(value: object) -> None:
    with pytest.raises(ValueError, match="string"):
        validate_model_name(value)


@pytest.mark.parametrize("name", _INVALID_NAMES)
def test_store_rejects_unsafe_names_without_touching_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("filesystem accessed for an invalid name")

    for attr in ("rglob", "glob", "is_dir", "exists", "iterdir"):
        monkeypatch.setattr(Path, attr, forbidden)
    store = ModelStore(tmp_path)
    with pytest.raises(ValueError):
        store.resolve(name)
    with pytest.raises(ValueError):
        store.find_root(name)


def _outside(tmp_path: Path, label: str) -> Path:
    outside = tmp_path.parent / f"{tmp_path.name}-{label}"
    outside.mkdir()
    return outside


def test_store_ignores_outward_model_directory_symlink(tmp_path: Path) -> None:
    outside = _outside(tmp_path, "dir")
    _make_gguf(outside, "model.gguf")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve("linked")
    assert store.find_root("linked") is None
    assert "linked" not in [e.name for e in store.list()]
    assert store.complete("") == []


def test_store_ignores_outward_namespace_symlink(tmp_path: Path) -> None:
    outside = _outside(tmp_path, "ns")
    _make_gguf(outside, "model", "model.gguf")
    (tmp_path / "org").symlink_to(outside, target_is_directory=True)
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve("org/model")
    assert store.find_root("org/model") is None
    assert store.list() == []
    assert store.complete("org") == []


def test_store_ignores_outward_flat_and_primary_file_symlinks(tmp_path: Path) -> None:
    outside = _outside(tmp_path, "file")
    target = _make_gguf(outside, "weights.gguf")
    (tmp_path / "flat.gguf").symlink_to(target)
    (tmp_path / "boxed").mkdir()
    (tmp_path / "boxed" / "model.gguf").symlink_to(target)
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve("flat")
    assert store.find_root("flat") is None
    with pytest.raises(FileNotFoundError):
        store.resolve("boxed")
    assert [e.name for e in store.list()] == []


def test_store_keeps_symlinks_that_stay_inside(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "real", "model.gguf")
    (tmp_path / "alias-dir").symlink_to(tmp_path / "real", target_is_directory=True)
    assert ModelStore(tmp_path).resolve("alias-dir") == tmp_path / "alias-dir" / primary.name


def test_store_supports_symlinked_models_dir(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    primary = _make_gguf(storage, "org", "model", "model.gguf")
    link = tmp_path / "models"
    link.symlink_to(storage, target_is_directory=True)
    store = ModelStore(link)
    assert store.resolve("org/model") == link / "org" / "model" / primary.name
    assert [e.name for e in store.list()] == ["org/model"]


def test_store_ignores_alias_that_is_not_a_valid_name(tmp_path: Path) -> None:
    from drove.model_config import DownloadInfo, save_download_info

    primary = _make_gguf(tmp_path, "a", "b", "c", "model.gguf")
    save_download_info(primary, DownloadInfo(repo_id="org/repo", files=[primary.name]))
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve("org/repo")
    assert store.find_root("org/repo") is None


@pytest.mark.parametrize("dangling", [False, True])
def test_store_ignores_model_whose_sidecar_escapes(tmp_path: Path, dangling: bool) -> None:
    """A sidecar symlinked outside models_dir would let config reads/writes escape."""
    models = tmp_path / "models"
    _make_gguf(models, "real", "model.gguf")
    _make_gguf(models, "flat.gguf")
    external = tmp_path / "external.toml"
    if not dangling:
        external.write_text("ctx_size = 2048\n")
    (models / "real" / "model.toml").symlink_to(external)
    (models / "flat.toml").symlink_to(external)
    store = ModelStore(models)
    for name in ("real", "flat"):
        with pytest.raises(FileNotFoundError):
            store.resolve(name)
    assert store.list() == []
    if not dangling:
        assert external.read_text() == "ctx_size = 2048\n"


def test_store_keeps_model_with_inside_sidecar_symlink(tmp_path: Path) -> None:
    primary = _make_gguf(tmp_path, "real", "model.gguf")
    shared = tmp_path / "shared.toml"
    shared.write_text("ctx_size = 2048\n")
    (tmp_path / "real" / "model.toml").symlink_to(shared)
    assert ModelStore(tmp_path).resolve("real") == primary


def test_alias_scan_skips_sidecars_outside_models_dir(tmp_path: Path) -> None:
    from drove.model_config import resolve_model_alias

    models = tmp_path / "models"
    _make_gguf(models, "real", "model.gguf")
    external = tmp_path / "external.toml"
    external.write_text('[download]\nrepo_id = "org/repo"\nfiles = ["model.gguf"]\n')
    (models / "real" / "meta.toml").symlink_to(external)
    assert resolve_model_alias(models, "org/repo") is None
    with pytest.raises(FileNotFoundError):
        ModelStore(models).resolve("org/repo")


def test_inventory_only_offers_names_the_validator_accepts(tmp_path: Path) -> None:
    """Every listed or completed name must be usable with resolve() and the CLI."""
    _make_gguf(tmp_path, "good", "model.gguf")
    _make_gguf(tmp_path, "org", "good", "model.gguf")
    _make_gguf(tmp_path, "my model.gguf")
    _make_gguf(tmp_path, "bad name", "model.gguf")
    _make_gguf(tmp_path, "org", "bad name", "model.gguf")
    _make_gguf(tmp_path, "bad ns", "inner", "model.gguf")
    _make_gguf(tmp_path, "_scratch", "model.gguf")
    store = ModelStore(tmp_path)
    listed = [e.name for e in store.list()]
    assert listed == ["good", "org/good"]
    assert store.complete("") == ["good", "org/good"]
    for name in listed:
        assert store.resolve(name)


def test_inventory_and_resolve_pick_the_same_servable_candidate(tmp_path: Path) -> None:
    """list(), complete() and resolve() agree when the first candidate is unservable."""
    models = tmp_path / "models"
    external = tmp_path / "external.toml"
    external.write_text("ctx_size = 2048\n")
    _make_gguf(models, "multi", "a.gguf")
    safe = _make_gguf(models, "multi", "b.gguf")
    (models / "multi" / "a.toml").symlink_to(external)
    _make_gguf(models, "org", "multi", "a.gguf")
    nested_safe = _make_gguf(models, "org", "multi", "b.gguf")
    (models / "org" / "multi" / "a.toml").symlink_to(external)
    _make_gguf(models, "blocked", "a.gguf")
    (models / "blocked" / "a.toml").symlink_to(external)
    store = ModelStore(models)
    entries = {e.name: e.primary for e in store.list()}
    assert entries == {"multi": safe, "org/multi": nested_safe}
    assert store.complete("") == ["multi", "org/multi"]
    for name, path in entries.items():
        assert store.resolve(name) == path
    with pytest.raises(FileNotFoundError):
        store.resolve("blocked")


# ── servable and complete downloads only (#117) ───────────────────────────────


def _write_manifest(root: Path, state: str, files: dict[str, int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".drove-download.json").write_text(
        json.dumps({"version": 1, "state": state, "files": files})
    )


@pytest.mark.parametrize("name", ["interrupted", "org/interrupted"])
def test_cache_only_directory_is_not_a_model(tmp_path: Path, name: str) -> None:
    """An interrupted first transfer leaves only Hugging Face cache metadata."""
    cache = tmp_path / name / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model.gguf.lock").write_bytes(b"")
    (cache / "abc123.incomplete").write_bytes(b"partial")
    (cache / "model.gguf.metadata").write_text("etag\n")
    _make_gguf(tmp_path, name, ".cache", "huggingface", "stray.gguf")
    _make_gguf(tmp_path, "ready.gguf")

    store = ModelStore(tmp_path)
    assert [e.name for e in store.list()] == ["ready"]
    assert store.complete("inter") == []
    assert store.complete("org/") == []
    with pytest.raises(FileNotFoundError):
        store.resolve(name)


@pytest.mark.parametrize("extension", ["safetensors", "bin", "pt"])
@pytest.mark.parametrize("name", ["raw", "org/raw"])
def test_raw_weight_directory_is_not_a_model(tmp_path: Path, extension: str, name: str) -> None:
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / f"model.{extension}").write_bytes(b"weights")
    (folder / "config.json").write_text("{}")

    store = ModelStore(tmp_path)
    assert store.list() == []
    assert store.complete("") == []
    with pytest.raises(FileNotFoundError):
        store.resolve(name)


def test_hidden_descendants_never_become_primary(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "m", ".partial", "a.gguf")
    primary = _make_gguf(tmp_path, "m", "b.gguf")
    store = ModelStore(tmp_path)
    assert store.resolve("m") == primary
    assert [e.primary for e in store.list()] == [primary]


@pytest.mark.parametrize("layout", ["dir", "flat"])
def test_first_shard_without_the_rest_is_incomplete(tmp_path: Path, layout: str) -> None:
    parts = ("big", "big-00001-of-00003.gguf") if layout == "dir" else ("big-00001-of-00003.gguf",)
    _make_gguf(tmp_path, *parts)
    name = "big" if layout == "dir" else "big-00001-of-00003"

    store = ModelStore(tmp_path)
    assert store.list() == []
    assert store.complete("big") == []
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.resolve(name)


def test_missing_middle_shard_is_incomplete(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "big", "big-00001-of-00003.gguf")
    _make_gguf(tmp_path, "big", "big-00003-of-00003.gguf")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ModelStore(tmp_path).resolve("big")
    assert ModelStore(tmp_path).list() == []


def test_later_shard_alone_is_not_listed(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "big", "big-00002-of-00003.gguf")
    _make_gguf(tmp_path, "big-00002-of-00003.gguf")
    assert ModelStore(tmp_path).list() == []


def test_complete_shard_set_resolves_to_first_shard(tmp_path: Path) -> None:
    first = _make_gguf(tmp_path, "big", "big-00001-of-00003.gguf")
    _make_gguf(tmp_path, "big", "big-00002-of-00003.gguf")
    _make_gguf(tmp_path, "big", "big-00003-of-00003.gguf")
    store = ModelStore(tmp_path)
    assert store.resolve("big") == first
    assert [e.name for e in store.list()] == ["big"]


def test_incomplete_manifest_hides_model(tmp_path: Path) -> None:
    root = tmp_path / "org" / "asr"
    (root).mkdir(parents=True)
    (root / "model.onnx").write_bytes(b"onnx")
    _write_manifest(root, "incomplete", {"model.onnx": 4, "model.onnx.data": 10})

    store = ModelStore(tmp_path)
    assert store.list() == []
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.resolve("org/asr")
    # Still found for management (delete) purposes.
    assert store.find_root("org/asr") == root


def test_complete_manifest_resolves(tmp_path: Path) -> None:
    root = tmp_path / "asr"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"onnx")
    (root / "vocab.txt").write_bytes(b"")
    _write_manifest(root, "complete", {"model.onnx": 4, "vocab.txt": 0})
    assert ModelStore(tmp_path).resolve("asr") == root / "model.onnx"
    assert [e.name for e in ModelStore(tmp_path).list()] == ["asr"]


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_complete_manifest_with_damaged_file_is_incomplete(tmp_path: Path, damage: str) -> None:
    root = tmp_path / "m"
    primary = _make_gguf(tmp_path, "m", "m.gguf")
    primary.write_bytes(b"gguf")
    (root / "mmproj.gguf").write_bytes(b"proj")
    _write_manifest(root, "complete", {"m.gguf": 4, "mmproj.gguf": 4})
    if damage == "truncate":
        (root / "mmproj.gguf").write_bytes(b"p")
    else:
        (root / "mmproj.gguf").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ModelStore(tmp_path).resolve("m")
    assert ModelStore(tmp_path).list() == []


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"version": 99, "state": "complete", "files": {}}',
        '{"version": 1, "state": "complete", "files": {"../escape.gguf": 0}}',
        '{"version": 1, "state": "complete", "files": ["m.gguf"]}',
    ],
)
def test_unreadable_or_unsafe_manifest_fails_closed(tmp_path: Path, content: str) -> None:
    _make_gguf(tmp_path, "m", "m.gguf")
    _make_gguf(tmp_path, "escape.gguf")
    (tmp_path / "m" / ".drove-download.json").write_text(content)
    assert [e.name for e in ModelStore(tmp_path).list()] == ["escape"]
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ModelStore(tmp_path).resolve("m")


def test_incomplete_exact_root_does_not_fall_back_to_alias(tmp_path: Path) -> None:
    root = tmp_path / "org" / "repo"
    _make_gguf(tmp_path, "org", "repo", "model.gguf")
    _write_manifest(root, "incomplete", {"model.gguf": 0})
    _make_gguf(tmp_path, "other", "other.gguf")
    (tmp_path / "other" / "other.toml").write_text('[download]\nrepo_id = "org/repo"\n')
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ModelStore(tmp_path).resolve("org/repo")


@pytest.mark.parametrize("layout", ["flat", "dir", "nested", "org", "onnx"])
def test_legacy_complete_downloads_without_manifest_still_resolve(
    tmp_path: Path, layout: str
) -> None:
    """Downloads made before completion tracking keep working."""
    if layout == "flat":
        expected, name = _make_gguf(tmp_path, "old.gguf"), "old"
    elif layout == "dir":
        expected, name = _make_gguf(tmp_path, "old", "old.gguf"), "old"
    elif layout == "nested":
        expected, name = _make_gguf(tmp_path, "old", "weights", "old.gguf"), "old"
    elif layout == "org":
        expected, name = _make_gguf(tmp_path, "org", "old", "old-Q4_K_M.gguf"), "org/old"
    else:
        expected = tmp_path / "old" / "model.onnx"
        expected.parent.mkdir()
        expected.write_bytes(b"onnx")
        name = "old"
    # Completed Hub downloads keep their cache metadata around.
    cache = expected.parent / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "old.gguf.lock").write_bytes(b"")
    store = ModelStore(tmp_path)
    assert store.resolve(name) == expected
    assert [e.primary for e in store.list()] == [expected]


@pytest.mark.parametrize("request_name", ["org", "org/asr"])
def test_incomplete_record_holds_for_every_spelling_of_the_model(
    tmp_path: Path, request_name: str
) -> None:
    """A parent namespace name must not reach a child download that is unfinished."""
    root = tmp_path / "org" / "asr"
    root.mkdir(parents=True)
    (root / "model.onnx").write_bytes(b"onnx")
    _write_manifest(root, "incomplete", {"model.onnx": 4, "vocab.txt": 1})
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ModelStore(tmp_path).resolve(request_name)


def test_record_above_the_listed_directory_still_applies(tmp_path: Path) -> None:
    root = tmp_path / "old"
    _make_gguf(tmp_path, "old", "weights", "old.gguf")
    _write_manifest(root, "incomplete", {"weights/old.gguf": 0})
    store = ModelStore(tmp_path)
    assert store.list() == []
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.resolve("old/weights")


@pytest.mark.parametrize("recorded", [{"z.gguf": 8}, {}])
def test_complete_record_must_cover_the_selected_primary(
    tmp_path: Path, recorded: dict[str, int]
) -> None:
    """A stale or empty record cannot vouch for a file it never downloaded."""
    root = tmp_path / "m"
    _make_gguf(tmp_path, "m", "a.gguf").write_bytes(b"part")
    _make_gguf(tmp_path, "m", "z.gguf").write_bytes(b"x" * 8)
    _write_manifest(root, "complete", recorded)
    store = ModelStore(tmp_path)
    assert store.list() == []
    with pytest.raises(FileNotFoundError, match="a.gguf is not part of its download record"):
        store.resolve("m")


def test_tts_marker_needs_no_entry_in_its_record(tmp_path: Path) -> None:
    root = tmp_path / "voice"
    root.mkdir()
    (root / "model.tts").write_bytes(b"")
    (root / "lm").mkdir()
    (root / "lm" / "model.safetensors").write_bytes(b"w")
    _write_manifest(root, "complete", {"lm/model.safetensors": 1})
    assert ModelStore(tmp_path).resolve("voice") == root / "model.tts"
