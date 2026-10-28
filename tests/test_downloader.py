"""Tests for downloader helpers."""

from __future__ import annotations

from drove.downloader import (
    DownloadPlan,
    available_onnx_quants,
    available_quants,
    filter_by_quant,
    filter_onnx_quant,
    is_onnx_files,
    quant_tag,
)


def test_quant_tag_extracts_common_tags() -> None:
    assert quant_tag("model-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_tag("model-IQ3_XXS.gguf") == "IQ3_XXS"
    assert quant_tag("model-BF16.gguf") == "BF16"
    assert quant_tag("model-F16.gguf") == "F16"
    assert quant_tag("model-Q8_0.gguf") == "Q8_0"


def test_quant_tag_returns_none_when_absent() -> None:
    assert quant_tag("plain-model.gguf") is None


def test_available_quants_groups_and_sums_sizes() -> None:
    files = {
        "m-Q4_K_M.gguf": 100,
        "m-Q4_K_M-00001-of-00002.gguf": 50,
        "m-Q8_0.gguf": 200,
        "m-BF16.gguf": 400,
    }
    result = available_quants(files)
    assert result == {"Q4_K_M": 150, "Q8_0": 200, "BF16": 400}


def test_available_quants_skips_files_without_tag() -> None:
    files = {"plain.gguf": 10, "m-Q5_K_M.gguf": 20}
    assert available_quants(files) == {"Q5_K_M": 20}


def test_filter_by_quant_is_case_insensitive() -> None:
    files = {"m-Q4_K_M.gguf": 1, "m-Q8_0.gguf": 2}
    assert filter_by_quant(files, "q4_k_m") == {"m-Q4_K_M.gguf": 1}


def test_filter_by_quant_exact_match_no_overlap() -> None:
    """Selecting F16 must not also match BF16 files (substring collision)."""
    files = {"m-F16.gguf": 100, "m-BF16.gguf": 200}
    assert filter_by_quant(files, "F16") == {"m-F16.gguf": 100}
    assert filter_by_quant(files, "BF16") == {"m-BF16.gguf": 200}


# ── ONNX (ASR) repos ─────────────────────────────────────────────────────────

_ONNX_FILES = {
    "encoder-model.onnx": 2_400,
    "encoder-model.int8.onnx": 600,
    "decoder_joint-model.onnx": 80,
    "decoder_joint-model.int8.onnx": 20,
}


def test_is_onnx_files() -> None:
    assert is_onnx_files(_ONNX_FILES)
    assert not is_onnx_files({"model-Q4_K_M.gguf": 1})


def test_filter_onnx_quant_default_excludes_quant_variants() -> None:
    assert filter_onnx_quant(_ONNX_FILES, None) == {
        "encoder-model.onnx": 2_400,
        "decoder_joint-model.onnx": 80,
    }


def test_filter_onnx_quant_int8_selects_only_int8() -> None:
    assert filter_onnx_quant(_ONNX_FILES, "int8") == {
        "encoder-model.int8.onnx": 600,
        "decoder_joint-model.int8.onnx": 20,
    }


def test_filter_onnx_quant_unknown_tag_returns_empty() -> None:
    assert filter_onnx_quant(_ONNX_FILES, "int4") == {}


def test_available_onnx_quants_groups_default_and_variants() -> None:
    assert available_onnx_quants(_ONNX_FILES) == {"default": 2_480, "int8": 620}


def test_available_onnx_quants_single_variant() -> None:
    files = {"encoder-model.onnx": 100, "decoder_joint-model.onnx": 50}
    assert available_onnx_quants(files) == {"default": 150}


def test_download_plan_keeps_full_onnx_file_set() -> None:
    """The plan retains all ONNX variants so the CLI can offer a quant choice."""
    plan = DownloadPlan(
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        files=filter_onnx_quant(_ONNX_FILES, None),
        local_name="istupakov/parakeet-tdt-0.6b-v3-onnx",
        sharded=False,
        onnx_files=_ONNX_FILES,
    )
    assert available_onnx_quants(plan.onnx_files) == {"default": 2_480, "int8": 620}
    assert filter_onnx_quant(plan.onnx_files, "int8") == {
        "encoder-model.int8.onnx": 600,
        "decoder_joint-model.int8.onnx": 20,
    }


def test_download_plan_includes_extra_files() -> None:
    plan = DownloadPlan(
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        files={"encoder-model.onnx": 2_400, "decoder_joint-model.onnx": 80},
        local_name="istupakov/parakeet-tdt-0.6b-v3-onnx",
        sharded=False,
        extra_files={"vocab.txt": 10, "config.json": 1},
    )
    assert plan.is_asr
    assert plan.total_bytes == 2_491
    assert set(plan._all_remote_files()) == {
        "encoder-model.onnx",
        "decoder_joint-model.onnx",
        "vocab.txt",
        "config.json",
    }


def test_download_plan_gguf_is_not_asr() -> None:
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files={"Qwen3-8B-Q8_0.gguf": 100},
        local_name="unsloth/Qwen3-8B-GGUF:Q8_0",
        sharded=False,
    )
    assert not plan.is_asr


def test_download_events_track_resume_skip_and_cache_hits(tmp_path, monkeypatch):
    from pathlib import Path

    from drove.downloader import DownloadEvent

    plan = DownloadPlan(
        "org/repo",
        {"a.gguf": 10, "b.gguf": 20},
        "model",
        False,
        mmproj_files={"mmproj.gguf": 5},
        extra_files={"config.json": 2},
    )
    dest = plan.destination(tmp_path)
    dest.mkdir()
    (dest / "a.gguf").write_bytes(b"a" * 10)
    (dest / "mmproj.gguf").write_bytes(b"x")  # Partial final file is not resume evidence.
    events: list[DownloadEvent] = []
    legacy = []

    def download(*, repo_id, filename, local_dir, tqdm_class):
        if filename == "b.gguf":
            # Hidden Hub partial exists even though the final target is missing.
            with tqdm_class(initial=8, total=20, disable=True, name="http", unit="B") as bar:
                bar.update(4)
            # A retry reconstructs the callback at the saved offset.
            with tqdm_class(initial=12, total=20, disable=False, name="http") as bar:
                bar.update(8)
        elif filename == "mmproj.gguf":
            with tqdm_class(initial=0, total=5, name="xet") as bar:
                bar.update(5.0)
        # config.json is a cache hit: Hub need not construct a progress bar.
        path = Path(local_dir) / filename
        path.write_bytes(b"x" * plan._all_remote_files()[filename])
        return str(path)

    monkeypatch.setattr("drove.downloader.hf_hub_download", download)
    plan.execute(tmp_path, lambda *args: legacy.append(args), event_cb=events.append)
    assert legacy[0] == (1, 4, "a.gguf (skipped)")
    assert [e.completed for e in events if e.state == "resumed"] == [18, 22]
    assert sum(e.transferred for e in events) == 17
    assert events[-1].completed == events[-1].total == 37
    assert max(e.completed for e in events) <= 37
    assert [(e.filename, e.state) for e in events if e.state != "progress"] == [
        ("a.gguf", "skipped"),
        ("b.gguf", "new"),
        ("b.gguf", "resumed"),
        ("b.gguf", "resumed"),
        ("b.gguf", "complete"),
        ("mmproj.gguf", "retrying"),
        ("mmproj.gguf", "complete"),
        ("config.json", "new"),
        ("config.json", "complete"),
    ]


def test_failed_transfer_never_emits_complete(tmp_path, monkeypatch):
    import pytest

    events = []
    plan = DownloadPlan("org/repo", {"model.gguf": 10}, "model", False)

    def download(**kwargs):
        with kwargs["tqdm_class"](initial=0, total=10) as bar:
            bar.update(100)  # Even a bad estimate must not overflow overall completion.
        raise RuntimeError("offline")

    monkeypatch.setattr("drove.downloader.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="offline"):
        plan.execute(tmp_path, event_cb=events.append)
    assert max(e.completed for e in events) == 10
    assert not any(e.state == "complete" for e in events)


def test_interrupt_keeps_tts_unpublished(tmp_path, monkeypatch):
    import pytest

    plan = DownloadPlan("org/repo", {"weights.bin": 10}, "voice", False, tts_engine="kokoro")
    events = []

    def download(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("drove.downloader.hf_hub_download", download)
    with pytest.raises(KeyboardInterrupt):
        plan.execute(tmp_path, event_cb=events.append)
    assert not (plan.destination(tmp_path) / "model.tts").exists()
    assert not any(e.state in ("complete", "configuring") for e in events)


# ── servable downloads only, completion record (#117) ─────────────────────────


def _hub(monkeypatch, files: dict[str, int]):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    api = MagicMock()
    api.model_info.return_value = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files.items()]
    )
    monkeypatch.setattr("drove.downloader.HfApi", lambda: api)
    transfer = MagicMock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr("drove.downloader.hf_hub_download", transfer)
    return transfer


def _fake_transfer(sizes: dict[str, int], fail_on: str | None = None):
    from pathlib import Path

    calls: list[str] = []

    def download(*, repo_id, filename, local_dir, **kwargs):
        calls.append(filename)
        if filename == fail_on:
            raise KeyboardInterrupt
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * sizes[filename])
        return str(path)

    download.calls = calls  # type: ignore[attr-defined]
    return download


import pytest  # noqa: E402


@pytest.mark.parametrize("extension", ["safetensors", "bin", "pt"])
def test_raw_weight_repository_is_refused_without_downloading(monkeypatch, extension):
    from drove.downloader import resolve_download

    transfer = _hub(
        monkeypatch,
        {f"model-00001-of-00002.{extension}": 10, "config.json": 1, "tokenizer.json": 1},
    )
    with pytest.raises(ValueError, match="GGUF") as excinfo:
        resolve_download("org/raw-model")
    assert "convert" in str(excinfo.value)
    transfer.assert_not_called()


def test_mixed_repository_downloads_only_gguf(monkeypatch):
    from drove.downloader import resolve_download

    _hub(monkeypatch, {"model.Q4_K_M.gguf": 10, "model.safetensors": 99, "pytorch_model.bin": 9})
    plan = resolve_download("org/mixed")
    assert plan.files == {"model.Q4_K_M.gguf": 10}


def test_projector_only_repository_is_refused(monkeypatch):
    from drove.downloader import resolve_download

    _hub(monkeypatch, {"mmproj-model-f16.gguf": 10})
    with pytest.raises(ValueError, match="GGUF"):
        resolve_download("org/proj-only")


def test_interrupted_shard_download_is_recorded_and_never_served(tmp_path, monkeypatch):
    from drove.download_state import read_state
    from drove.model_store import IncompleteModelError, ModelStore

    sizes = {
        "big-00001-of-00003.gguf": 3,
        "big-00002-of-00003.gguf": 4,
        "big-00003-of-00003.gguf": 5,
    }
    plan = DownloadPlan("org/big", sizes, "big", True, mmproj_files={"mmproj.gguf": 2})
    store = ModelStore(tmp_path)
    monkeypatch.setattr(
        "drove.downloader.hf_hub_download",
        _fake_transfer({**sizes, "mmproj.gguf": 2}, "big-00002-of-00003.gguf"),
    )
    with pytest.raises(KeyboardInterrupt):
        plan.execute(tmp_path)

    dest = plan.destination(tmp_path)
    state = read_state(dest)
    assert state is not None and not state.complete
    assert state.files == {**sizes, "mmproj.gguf": 2}
    assert (dest / "big-00001-of-00003.gguf").exists()  # kept for resume
    assert store.list() == []
    with pytest.raises(IncompleteModelError, match="did not finish"):
        store.resolve("big")

    retry = _fake_transfer({**sizes, "mmproj.gguf": 2})
    monkeypatch.setattr("drove.downloader.hf_hub_download", retry)
    primary = plan.execute(tmp_path)
    assert retry.calls == ["big-00002-of-00003.gguf", "big-00003-of-00003.gguf", "mmproj.gguf"]
    assert read_state(dest).complete
    assert store.resolve("big") == primary == dest / "big-00001-of-00003.gguf"


def test_record_is_incomplete_while_transferring(tmp_path, monkeypatch):
    """A running proxy never sees a model mid-download, even an ONNX one."""
    from drove.model_store import ModelStore

    sizes = {"model.onnx": 4, "model.onnx.data": 8, "vocab.txt": 1}
    plan = DownloadPlan(
        "org/asr",
        {"model.onnx": 4, "model.onnx.data": 8},
        "asr",
        False,
        extra_files={"vocab.txt": 1},
    )
    seen: list[list[str]] = []
    transfer = _fake_transfer(sizes)

    def observing(**kwargs):
        seen.append([e.name for e in ModelStore(tmp_path).list()])
        return transfer(**kwargs)

    monkeypatch.setattr("drove.downloader.hf_hub_download", observing)
    plan.execute(tmp_path)
    assert seen == [[], [], []]
    assert [e.name for e in ModelStore(tmp_path).list()] == ["asr"]


def test_tts_download_is_recorded_complete_after_marker(tmp_path, monkeypatch):
    from drove.download_state import read_state

    plan = DownloadPlan(
        "org/voice", {"weights.safetensors": 3}, "voice", False, tts_engine="kokoro"
    )
    monkeypatch.setattr(
        "drove.downloader.hf_hub_download", _fake_transfer({"weights.safetensors": 3})
    )
    primary = plan.execute(tmp_path)
    assert primary.name == "model.tts"
    state = read_state(primary.parent)
    assert state is not None and state.complete
    assert state.files == {"weights.safetensors": 3}


def test_unknown_remote_size_needs_only_the_file(tmp_path, monkeypatch):
    from drove.model_store import ModelStore

    plan = DownloadPlan("org/m", {"m.gguf": 0}, "m", False)
    monkeypatch.setattr("drove.downloader.hf_hub_download", _fake_transfer({"m.gguf": 6}))
    primary = plan.execute(tmp_path)
    assert ModelStore(tmp_path).resolve("m") == primary


def test_files_flattening_to_one_name_are_refused(tmp_path, monkeypatch):
    plan = DownloadPlan("org/m", {"a/model.gguf": 1, "b/model.gguf": 1}, "m", False)
    monkeypatch.setattr("drove.downloader.hf_hub_download", _fake_transfer({}))
    with pytest.raises(ValueError, match="model.gguf"):
        plan.execute(tmp_path)
    assert not plan.destination(tmp_path).exists()
