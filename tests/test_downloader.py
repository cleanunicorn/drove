"""Tests for downloader helpers."""

from __future__ import annotations

from drove.downloader import (
    DownloadPlan,
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
