"""Tests for ModelStore — the single model-resolution authority."""

from __future__ import annotations

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


# ── mmproj projectors never become the primary ───────────────────────────────


def _write_download(path: Path, files: list[str], extra: str = "") -> None:
    listed = ", ".join(f'"{f}"' for f in files)
    path.write_text(f'{extra}[download]\nrepo_id = "org/repo"\nfiles = [{listed}]\n')


def _agree(store: ModelStore, name: str, expected: Path) -> None:
    assert store.resolve(name) == expected
    assert {e.name: e.primary for e in store.list()}[name] == expected
    assert name in store.complete("")


@pytest.mark.parametrize(
    ("projector", "weights"),
    [
        ("mmproj-F16.gguf", "zeta-q4_k_m.gguf"),
        ("MMPROJ-F16.GGUF", "zeta-q4_k_m.gguf"),
        ("MmProj-f16.GgUf", "Zeta-Q4.GGUF"),
        ("mmproj-F16.gguf", "ZETA.GGUF"),
        ("zz-mmproj.gguf", "aa-weights.gguf"),
        ("aa-mmproj.gguf", "zz-weights.gguf"),
    ],
)
@pytest.mark.parametrize(
    ("name", "layout", "weights_sub"),
    [("vis", ("vis",), ()), ("org/vis", ("org", "vis"), ()), ("vis", ("vis",), ("weights",))],
)
def test_primary_is_weights_not_projector(
    tmp_path: Path,
    projector: str,
    weights: str,
    name: str,
    layout: tuple[str, ...],
    weights_sub: tuple[str, ...],
) -> None:
    _make_gguf(tmp_path, *layout, projector)
    expected = _make_gguf(tmp_path, *layout, *weights_sub, weights)
    _agree(ModelStore(tmp_path), name, expected)


def test_sharded_vision_model_resolves_to_shard_one(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "vis", "mmproj-F16.gguf")
    first = _make_gguf(tmp_path, "vis", "shard-00001-of-00002.gguf")
    _make_gguf(tmp_path, "vis", "shard-00002-of-00002.gguf")
    _agree(ModelStore(tmp_path), "vis", first)


def test_sharded_vision_model_with_sidecar_resolves_to_shard_one(tmp_path: Path) -> None:
    first = _make_gguf(tmp_path, "vis", "shard-00001-of-00002.gguf")
    _make_gguf(tmp_path, "vis", "shard-00002-of-00002.gguf")
    _make_gguf(tmp_path, "vis", "mmproj-F16.gguf")
    _write_download(
        first.with_suffix(".toml"),
        ["shard-00001-of-00002.gguf", "shard-00002-of-00002.gguf", "mmproj-F16.gguf"],
    )
    _agree(ModelStore(tmp_path), "vis", first)


def test_sidecar_files_list_selects_weights_over_sorted_first_decoy(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "vis", "a-decoy.gguf")
    weights = _make_gguf(tmp_path, "vis", "zeta.gguf")
    _make_gguf(tmp_path, "vis", "mmproj-F16.gguf")
    _write_download(weights.with_suffix(".toml"), ["mmproj-F16.gguf", "remote/path/zeta.gguf"])
    _agree(ModelStore(tmp_path), "vis", weights)


@pytest.mark.parametrize(
    "files",
    [[], ["mmproj-F16.gguf"], ["gone.gguf"], ["other-model.gguf"]],
)
def test_stale_download_metadata_falls_back_to_sorted_weights(
    tmp_path: Path, files: list[str]
) -> None:
    first = _make_gguf(tmp_path, "vis", "a.gguf")
    weights = _make_gguf(tmp_path, "vis", "zeta.gguf")
    _make_gguf(tmp_path, "vis", "mmproj-F16.gguf")
    _write_download(weights.with_suffix(".toml"), files)
    _agree(ModelStore(tmp_path), "vis", first)


def test_malformed_download_metadata_falls_back(tmp_path: Path) -> None:
    first = _make_gguf(tmp_path, "vis", "a.gguf")
    weights = _make_gguf(tmp_path, "vis", "zeta.gguf")
    weights.with_suffix(".toml").write_text("[download\nbroken")
    first.with_suffix(".toml").write_text("[download]\nrepo_id = 1\n")
    _agree(ModelStore(tmp_path), "vis", first)


def test_escaping_sidecar_is_never_read_for_preference(tmp_path: Path) -> None:
    models = tmp_path / "models"
    external = tmp_path / "external.toml"
    _write_download(external, ["zeta.gguf"])
    first = _make_gguf(models, "vis", "a.gguf")
    _make_gguf(models, "vis", "zeta.gguf")
    (models / "vis" / "zeta.toml").symlink_to(external)
    _agree(ModelStore(models), "vis", first)


@pytest.mark.parametrize("layout", [("vis",), ("org", "vis")])
def test_projector_only_directory_is_not_a_model(tmp_path: Path, layout: tuple[str, ...]) -> None:
    proj = _make_gguf(tmp_path, *layout, "MMPROJ-F16.GGUF")
    _make_gguf(tmp_path, *layout, "mmproj-BF16.gguf")
    proj.with_suffix(".toml").write_text("ctx_size = 1\n")
    (proj.parent / "README.md").write_text("notes")
    name = "/".join(layout)
    store = ModelStore(tmp_path)
    assert store.list() == []
    assert store.complete("") == []
    with pytest.raises(FileNotFoundError):
        store.resolve(name)
    assert store.find_root(name) == proj.parent


def test_legacy_flat_projector_is_not_a_model(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "mmproj-F16.gguf")
    store = ModelStore(tmp_path)
    assert store.list() == []
    with pytest.raises(FileNotFoundError):
        store.resolve("mmproj-F16")


def test_projector_alongside_onnx_serves_onnx(tmp_path: Path) -> None:
    _make_gguf(tmp_path, "asr", "mmproj-F16.gguf")
    onnx = _make_gguf(tmp_path, "asr", "model.onnx")
    _agree(ModelStore(tmp_path), "asr", onnx)


def test_undecodable_sidecar_does_not_break_inventory(tmp_path: Path) -> None:
    bad = _make_gguf(tmp_path, "bad", "main.gguf")
    bad.with_suffix(".toml").write_bytes(b"\xff\xfe[download]")
    good = _make_gguf(tmp_path, "good", "main.gguf")
    store = ModelStore(tmp_path)
    assert {e.name: e.primary for e in store.list()} == {"bad": bad, "good": good}
    assert store.resolve("bad") == bad


@pytest.mark.parametrize("filename", ["mmproj-speech.onnx", "MMPROJ-voice.tts"])
def test_mmproj_in_non_gguf_model_name_is_still_a_model(tmp_path: Path, filename: str) -> None:
    primary = _make_gguf(tmp_path, "speech", filename)
    _agree(ModelStore(tmp_path), "speech", primary)
