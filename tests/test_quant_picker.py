"""Quantization previews, recommendations, and real keyboard interaction."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import OptionList
from typer.testing import CliRunner

from drove.cli.main import app
from drove.cli.quant_picker import (
    ALL_VARIANTS,
    QuantPicker,
    available_memory,
    build_candidates,
    recommended_tag,
)
from drove.downloader import DownloadPlan


def _plan() -> DownloadPlan:
    return DownloadPlan(
        "org/model",
        {"model-Q4_K_M.gguf": 400, "model-Q8_0.gguf": 800},
        "org/model",
        False,
        mmproj_files={"mmproj.gguf": 100},
        extra_files={"tokenizer.json": 20},
    )


def test_candidates_have_exact_plans_and_union_sizes_without_writes(tmp_path: Path) -> None:
    plan = _plan()
    plan.files["untagged.gguf"] = 30
    choices = build_candidates(plan, None, tmp_path)
    assert [c.tag for c in choices] == ["Q4_K_M", "Q8_0", ALL_VARIANTS]
    assert choices[0].size_label == "520 bytes"
    assert choices[-1].size_label == "1,350 bytes"
    assert choices[0].plan.local_name == "org/model:Q4_K_M"
    assert choices[-1].plan.files == plan.files
    assert plan.local_name == "org/model"
    assert len(plan.files) == 3
    assert list(tmp_path.iterdir()) == []


def test_sizes_deduplicate_shared_files_and_count_shards(tmp_path: Path) -> None:
    plan = _plan()
    plan.files = {
        "Q4_K_M-00001-of-00002.gguf": 200,
        "Q4_K_M-00002-of-00002.gguf": 200,
        "Q8_0.gguf": 800,
    }
    plan.extra_files["mmproj.gguf"] = 100
    choices = build_candidates(plan, None, tmp_path)
    assert choices[0].size_label == "520 bytes"
    assert choices[0].plan.sharded
    assert choices[-1].size_label == "1,320 bytes"


@pytest.mark.parametrize("unknown", ["model-Q4_K_M.gguf", "tokenizer.json"])
def test_unknown_sizes_are_not_fabricated(tmp_path: Path, unknown: str) -> None:
    plan = _plan()
    (plan.files if unknown.endswith("gguf") else plan.extra_files)[unknown] = 0
    candidate = build_candidates(plan, None, tmp_path)[0]
    assert "total unknown" in candidate.size_label
    if unknown.endswith("gguf"):
        assert candidate.weight_bytes is None
        assert candidate.memory_note(100_000) == "memory estimate unavailable"


@pytest.mark.parametrize(
    ("available", "expected"),
    [(625, "Q4_K_M"), (624, "Q4_K_M"), (2000, "Q4_K_M"), (None, "Q4_K_M")],
)
def test_recommendation_and_threshold(tmp_path: Path, available: int | None, expected: str) -> None:
    choices = build_candidates(_plan(), None, tmp_path)
    assert recommended_tag(choices, available) == expected
    assert "likely too large" in choices[0].memory_note(624)
    assert "within estimated" in choices[0].memory_note(625)
    assert "unknown" in choices[0].memory_note(None)


def test_smallest_fitting_fallback_and_unknown_never_recommended_as_fitting(tmp_path: Path) -> None:
    plan = _plan()
    plan.files = {"Q8_0.gguf": 800, "Q5_K_M.gguf": 500, "Q4_K_M.gguf": 0}
    choices = build_candidates(plan, None, tmp_path)
    for memory in [2000, 1, None]:
        assert recommended_tag(choices, memory) == "Q5_K_M"
    plan.files = {"Q8_0.gguf": 0, "Q4_K_M.gguf": 0}
    assert recommended_tag(build_candidates(plan, None, tmp_path), 1000) == "Q8_0"


def test_memory_probe_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("drove.cli.quant_picker.psutil.virtual_memory", Mock(side_effect=OSError))
    assert available_memory() is None


def _write_files(destination: Path, files: dict[str, int]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, size in files.items():
        (destination / Path(filename).name).write_bytes(b"x" * size)


def test_local_complete_partial_and_base_location(tmp_path: Path) -> None:
    plan = _plan()
    dest = tmp_path / "org/model:Q4_K_M"
    _write_files(dest, {"model-Q4_K_M.gguf": 400, "mmproj.gguf": 100})
    assert build_candidates(plan, None, tmp_path)[0].local_status == "partly downloaded"
    (dest / "tokenizer.json").write_bytes(b"x" * 20)
    assert build_candidates(plan, None, tmp_path)[0].local_status == "fully downloaded"
    _write_files(
        tmp_path / "org/model", {"model-Q8_0.gguf": 800, "mmproj.gguf": 100, "tokenizer.json": 20}
    )
    status = build_candidates(plan, None, tmp_path)[1].local_status
    assert "fully downloaded at" in status and "separate location" in status


def test_unknown_size_existing_file_is_not_claimed_complete(tmp_path: Path) -> None:
    plan = _plan()
    plan.files["model-Q4_K_M.gguf"] = 0
    _write_files(
        tmp_path / "org/model:Q4_K_M",
        {"model-Q4_K_M.gguf": 400, "mmproj.gguf": 100, "tokenizer.json": 20},
    )
    assert build_candidates(plan, None, tmp_path)[0].local_status == "partly downloaded"


def test_custom_destination_and_short_file(tmp_path: Path) -> None:
    plan = _plan()
    plan.local_name = "custom"
    _write_files(
        tmp_path / "custom", {"model-Q4_K_M.gguf": 100, "mmproj.gguf": 100, "tokenizer.json": 20}
    )
    candidate = build_candidates(plan, "custom", tmp_path)[0]
    assert candidate.plan.local_name == "custom"
    assert candidate.local_status == "partly downloaded"


def test_attributable_hub_partial_and_unrelated_cache(tmp_path: Path) -> None:
    # Use the installed Hub implementation as an independent fixture oracle.
    from huggingface_hub._local_folder import get_local_download_paths

    plan = _plan()
    paths = get_local_download_paths(tmp_path / "org/model:Q4_K_M", "model-Q4_K_M.gguf")
    partial = paths.incomplete_path("a" * 64)
    partial.write_bytes(b"x" * 50)
    assert build_candidates(plan, None, tmp_path)[0].local_status == "partly downloaded"
    partial.unlink()
    (partial.parent / "unrelated.incomplete").write_bytes(b"bytes")
    assert build_candidates(plan, None, tmp_path)[0].local_status == ""


def test_nested_repo_partial_cache(tmp_path: Path) -> None:
    from huggingface_hub._local_folder import get_local_download_paths

    plan = _plan()
    plan.files = {"weights/model-Q4_K_M.gguf": 400, "weights/model-Q8_0.gguf": 800}
    paths = get_local_download_paths(tmp_path / "org/model:Q4_K_M", "weights/model-Q4_K_M.gguf")
    paths.incomplete_path("etag").write_bytes(b"part")
    assert build_candidates(plan, None, tmp_path)[0].local_status == "partly downloaded"


def test_onnx_candidates_default_all_and_preferred_int8(tmp_path: Path) -> None:
    plan = DownloadPlan(
        "org/speech",
        {"model.onnx": 800},
        "org/speech",
        False,
        extra_files={"tokens.txt": 10},
        onnx_files={"model.onnx": 800, "model.int8.onnx": 200},
    )
    choices = build_candidates(plan, None, tmp_path)
    assert choices[0].plan.local_name == "org/speech"
    assert choices[0].quant is None
    assert choices[1].plan.local_name == "org/speech:int8"
    assert choices[-1].plan.files == plan.onnx_files
    assert choices[-1].size_label == "1,010 bytes"
    assert recommended_tag(choices, 2000) == "int8"


async def test_keyboard_navigation_enter_and_separated_all(tmp_path: Path) -> None:
    choices = build_candidates(_plan(), None, tmp_path)
    picker = QuantPicker(choices, 2000)
    async with picker.run_test() as pilot:
        options = picker.query_one(OptionList)
        assert options.highlighted == 0
        await pilot.press("down", "enter")
    assert picker.return_value == "Q8_0"
    picker = QuantPicker(choices, 2000)
    async with picker.run_test() as pilot:
        await pilot.press("down", "down", "enter")
    assert picker.return_value == ALL_VARIANTS
    assert "Download every variant" in choices[-1].label(2000).plain


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_keyboard_cancel(tmp_path: Path, key: str) -> None:
    picker = QuantPicker(build_candidates(_plan(), None, tmp_path), 2000)
    async with picker.run_test() as pilot:
        await pilot.press(key)
    assert picker.return_value is None


async def test_narrow_terminal_long_list_scroll_and_literal_labels(tmp_path: Path) -> None:
    plan = _plan()
    plan.files = {f"model-Q{i}_K.gguf": i * 100 for i in range(2, 20)}
    choices = build_candidates(plan, None, tmp_path)
    picker = QuantPicker(choices, 100_000)
    async with picker.run_test(size=(45, 15)) as pilot:
        options = picker.query_one(OptionList)
        await pilot.press(*(["down"] * 17))
        assert options.highlighted == 17
        assert options.scroll_y > 0
        await pilot.press("enter")
    assert picker.return_value == "Q19_K"


def test_remote_label_is_literal(tmp_path: Path) -> None:
    plan = _plan()
    plan.local_name = "custom"
    # Model names cannot hold markup (#110), but the models directory can.
    models_dir = tmp_path / "[red]custom[/red]"
    _write_files(models_dir / plan.local_name, {"model-Q4_K_M.gguf": 1})
    label = build_candidates(plan, None, models_dir)[0].label(2000)
    assert "[red]custom[/red]" in label.plain
    assert label.spans == []


def _stub_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan: DownloadPlan
) -> tuple[Mock, Mock]:
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("drove.downloader.resolve_download", Mock(return_value=plan))
    execute = Mock(return_value=tmp_path / "model.gguf")
    monkeypatch.setattr(DownloadPlan, "execute", execute)
    metadata = Mock()
    monkeypatch.setattr("drove.cli.models.save_download_info", metadata)
    return execute, metadata


@pytest.mark.parametrize("answer", [None, "Q8_0", ALL_VARIANTS])
def test_cli_cancel_select_and_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str | None
) -> None:
    plan = _plan()
    plan.mmproj_files = {}
    execute, metadata = _stub_download(monkeypatch, tmp_path, plan)
    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", Mock(return_value=answer))
    result = CliRunner().invoke(app, ["models", "download", "org/model"], input="y\n")
    assert result.exit_code == 0, result.output
    if answer is None:
        assert "cancelled" in result.output
        execute.assert_not_called()
        metadata.assert_not_called()
        assert len(plan.files) == 2
    else:
        execute.assert_called_once()
        metadata.assert_called_once()
        assert len(plan.files) == (2 if answer == ALL_VARIANTS else 1)
        assert metadata.call_args.args[1].quant == (None if answer == ALL_VARIANTS else answer)


@pytest.mark.parametrize("args", [["org/model:Q4_K_M"], ["org/model", "--yes"]])
def test_cli_script_bypass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, args: list[str]
) -> None:
    plan = _plan()
    plan.mmproj_files = {}
    execute, _ = _stub_download(monkeypatch, tmp_path, plan)
    prompt = Mock(side_effect=AssertionError("picker should be bypassed"))
    memory = Mock(side_effect=AssertionError("memory should not be probed"))
    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", prompt)
    monkeypatch.setattr("drove.cli.quant_picker.available_memory", memory)
    result = CliRunner().invoke(app, ["models", "download", *args], input="y\n")
    if "--yes" in args:
        # Unattended with several quantizations: ask for :QUANT, download nothing.
        assert result.exit_code == 1
        assert "org/model:QUANT" in result.output
        execute.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        execute.assert_called_once()
    prompt.assert_not_called()
    memory.assert_not_called()


def test_noninteractive_ambiguous_download_guides_without_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execute, metadata = _stub_download(monkeypatch, tmp_path, _plan())
    result = CliRunner().invoke(app, ["models", "download", "org/model"])
    assert result.exit_code == 1
    # --yes no longer settles a multi-quant choice, so it is not suggested.
    assert ":QUANT" in result.output and "--yes" not in result.output
    assert "\x1b" not in result.output
    execute.assert_not_called()
    metadata.assert_not_called()


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_prompt_interrupts_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: type[BaseException]
) -> None:
    import sys

    from drove.cli.models import _prompt_quant_choice

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(QuantPicker, "run", Mock(side_effect=error))
    assert _prompt_quant_choice(build_candidates(_plan(), None, tmp_path), "textual-dark") is None


def test_single_variant_and_tts_skip_picker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from drove.cli.models import _select_quant_variant

    prompt = Mock(side_effect=AssertionError("picker must not run"))
    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", prompt)
    plan = _plan()
    plan.files = {"model-Q4_K_M.gguf": 400}
    assert _select_quant_variant(plan, None, tmp_path) is None
    plan = _plan()
    plan.tts_engine = "kokoro"
    assert _select_quant_variant(plan, None, tmp_path) is None
    prompt.assert_not_called()


def test_onnx_yes_keeps_default_and_skips_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = DownloadPlan(
        "org/speech",
        {"model.onnx": 800},
        "org/speech",
        False,
        onnx_files={"model.onnx": 800, "model.int8.onnx": 200},
    )
    execute, _ = _stub_download(monkeypatch, tmp_path, plan)
    monkeypatch.setattr("drove.cli.models.load_model_config", Mock())
    monkeypatch.setattr("drove.cli.models.save_model_config", Mock())
    memory = Mock(side_effect=AssertionError("memory must not run"))
    monkeypatch.setattr("drove.cli.quant_picker.available_memory", memory)
    result = CliRunner().invoke(app, ["models", "download", "org/speech", "--yes"])
    assert result.exit_code == 0, result.output
    assert plan.files == {"model.onnx": 800}
    execute.assert_called_once()
    memory.assert_not_called()


def test_prompt_reads_memory_once_and_uses_configured_theme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from drove.cli.models import _prompt_quant_choice

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    memory = Mock(return_value=1024)
    picker = Mock()
    picker.return_value.run.return_value = "Q4_K_M"
    monkeypatch.setattr("drove.cli.quant_picker.available_memory", memory)
    monkeypatch.setattr("drove.cli.quant_picker.QuantPicker", picker)
    candidates = build_candidates(_plan(), None, tmp_path)
    assert _prompt_quant_choice(candidates, "textual-light") == "Q4_K_M"
    memory.assert_called_once()
    picker.assert_called_once_with(candidates, 1024, "textual-light")
