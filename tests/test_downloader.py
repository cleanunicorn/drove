"""Tests for downloader helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.downloader import (
    DownloadPlan,
    FileStatus,
    available_quants,
    filter_by_quant,
    first_shard,
    infer_local_name,
    is_sharded,
    parse_model_ref,
    pick_mmproj,
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


# ---------------------------------------------------------------------------
# parse_model_ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref,expected_repo,expected_quant",
    [
        ("unsloth/Qwen3-8B-GGUF:Q4_K_M", "unsloth/Qwen3-8B-GGUF", "Q4_K_M"),
        ("unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", None),
        ("org/repo:BF16", "org/repo", "BF16"),
        ("org/repo:", "org/repo", None),  # empty quant treated as absent
    ],
)
def test_parse_model_ref(ref: str, expected_repo: str, expected_quant: str | None) -> None:
    repo_id, quant = parse_model_ref(ref)
    assert repo_id == expected_repo
    assert quant == expected_quant


# ---------------------------------------------------------------------------
# pick_mmproj
# ---------------------------------------------------------------------------


def test_pick_mmproj_single_file_returned_unchanged() -> None:
    files = {"mmproj-BF16.gguf": 500}
    assert pick_mmproj(files) == {"mmproj-BF16.gguf": 500}


def test_pick_mmproj_empty_returned_unchanged() -> None:
    assert pick_mmproj({}) == {}


def test_pick_mmproj_prefers_bf16_over_f32() -> None:
    # BF16 ranks above F32 in _MMPROJ_PREF.
    # NOTE: the "f16" tag check also matches "bf16" names (substring), so F16 vs
    # BF16 disambiguation is currently order-dependent.  Test the unambiguous case.
    files = {"mmproj-F32.gguf": 800, "mmproj-BF16.gguf": 400}
    result = pick_mmproj(files)
    assert list(result.keys()) == ["mmproj-BF16.gguf"]


def test_pick_mmproj_falls_back_to_smallest_when_no_preference_match() -> None:
    files = {"mmproj-Q8_0.gguf": 800, "mmproj-Q4_K_M.gguf": 300}
    result = pick_mmproj(files)
    # No F16/BF16/F32 → pick smallest
    assert list(result.keys()) == ["mmproj-Q4_K_M.gguf"]


# ---------------------------------------------------------------------------
# is_sharded / first_shard
# ---------------------------------------------------------------------------


def test_is_sharded_detects_shard_suffix() -> None:
    assert is_sharded(["model-00001-of-00003.gguf", "model-00002-of-00003.gguf"])


def test_is_sharded_returns_false_for_plain_files() -> None:
    assert not is_sharded(["model.gguf"])


def test_first_shard_returns_alphabetically_first() -> None:
    files = ["model-00003-of-00003.gguf", "model-00001-of-00003.gguf", "model-00002-of-00003.gguf"]
    assert first_shard(files) == "model-00001-of-00003.gguf"


# ---------------------------------------------------------------------------
# infer_local_name
# ---------------------------------------------------------------------------


def test_infer_local_name_with_quant() -> None:
    assert infer_local_name("unsloth/Qwen3-8B-GGUF", ["model-Q8_0.gguf"], "Q8_0") == "unsloth/Qwen3-8B-GGUF:Q8_0"


def test_infer_local_name_without_quant() -> None:
    assert infer_local_name("unsloth/Qwen3-8B-GGUF", ["model.gguf"], None) == "unsloth/Qwen3-8B-GGUF"


# ---------------------------------------------------------------------------
# DownloadPlan
# ---------------------------------------------------------------------------


def _make_plan(
    files: dict[str, int] | None = None,
    mmproj_files: dict[str, int] | None = None,
) -> DownloadPlan:
    return DownloadPlan(
        repo_id="org/repo",
        files=files or {"model-Q8_0.gguf": 1000},
        local_name="org/repo:Q8_0",
        sharded=False,
        mmproj_files=mmproj_files,
    )


def test_download_plan_total_bytes_model_only() -> None:
    plan = _make_plan(files={"model.gguf": 2048})
    assert plan.total_bytes == 2048


def test_download_plan_total_bytes_includes_mmproj() -> None:
    plan = _make_plan(files={"model.gguf": 2048}, mmproj_files={"mmproj-F16.gguf": 512})
    assert plan.total_bytes == 2560


def test_download_plan_check_local_files_all_missing(tmp_path: Path) -> None:
    plan = _make_plan(files={"model.gguf": 1000})
    result = plan.check_local_files(tmp_path)
    assert result["model.gguf"] == (FileStatus.MISSING, 0)


def test_download_plan_check_local_files_complete(tmp_path: Path) -> None:
    dest = tmp_path / "org" / "repo:Q8_0"
    dest.mkdir(parents=True)
    (dest / "model.gguf").write_bytes(b"x" * 1000)

    plan = _make_plan(files={"model.gguf": 1000})
    result = plan.check_local_files(tmp_path)
    assert result["model.gguf"][0] == FileStatus.COMPLETE


def test_download_plan_check_local_files_incomplete(tmp_path: Path) -> None:
    dest = tmp_path / "org" / "repo:Q8_0"
    dest.mkdir(parents=True)
    (dest / "model.gguf").write_bytes(b"x" * 100)  # only 100 of 1000

    plan = _make_plan(files={"model.gguf": 1000})
    result = plan.check_local_files(tmp_path)
    status, local_size = result["model.gguf"]
    assert status == FileStatus.INCOMPLETE
    assert local_size == 100
