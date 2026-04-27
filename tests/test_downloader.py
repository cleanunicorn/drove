"""Tests for downloader helpers."""

from __future__ import annotations

from drove.downloader import available_quants, filter_by_quant, quant_tag


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
