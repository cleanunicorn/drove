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
