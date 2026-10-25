"""Tests for the `drove models` CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.cli.models import _detect_capabilities, _select_quant_variant
from drove.downloader import DownloadPlan


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


# ── quant variant selection ──────────────────────────────────────────────────

_ONNX_ALL = {
    "encoder-model.onnx": 2_400,
    "encoder-model.int8.onnx": 600,
    "decoder_joint-model.onnx": 80,
    "decoder_joint-model.int8.onnx": 20,
}
_ONNX_DEFAULT = {"encoder-model.onnx": 2_400, "decoder_joint-model.onnx": 80}
_ONNX_INT8 = {"encoder-model.int8.onnx": 600, "decoder_joint-model.int8.onnx": 20}


def _onnx_plan() -> DownloadPlan:
    return DownloadPlan(
        repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
        files=dict(_ONNX_DEFAULT),
        local_name="istupakov/parakeet-tdt-0.6b-v3-onnx",
        sharded=False,
        onnx_files=dict(_ONNX_ALL),
    )


def _choose(monkeypatch: pytest.MonkeyPatch, answer: str | None) -> None:
    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", lambda quants: answer)


def test_select_quant_variant_onnx_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant == "int8"
    assert plan.files == _ONNX_INT8
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx:int8"


def test_select_quant_variant_onnx_default_keeps_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "default")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant is None
    assert plan.files == _ONNX_DEFAULT
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx"


def test_select_quant_variant_onnx_all_takes_every_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _choose(monkeypatch, None)
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None)
    assert quant is None
    assert plan.files == _ONNX_ALL


def test_select_quant_variant_onnx_keeps_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    plan.local_name = "my-parakeet"
    quant = _select_quant_variant(plan, "my-parakeet")
    assert quant == "int8"
    assert plan.local_name == "my-parakeet"


def test_select_quant_variant_onnx_single_variant_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(quants: dict[str, int]) -> str | None:
        raise AssertionError("prompt should not be shown")

    monkeypatch.setattr("drove.cli.models._prompt_quant_choice", boom)
    plan = DownloadPlan(
        repo_id="istupakov/whisper-base-onnx",
        files=dict(_ONNX_DEFAULT),
        local_name="istupakov/whisper-base-onnx",
        sharded=False,
        onnx_files=dict(_ONNX_DEFAULT),
    )
    assert _select_quant_variant(plan, None) is None
    assert plan.files == _ONNX_DEFAULT


def test_select_quant_variant_gguf_choice_filters_and_renames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _choose(monkeypatch, "Q4_K_M")
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files={"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200},
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    quant = _select_quant_variant(plan, None)
    assert quant == "Q4_K_M"
    assert plan.files == {"Qwen3-8B-Q4_K_M.gguf": 100}
    assert plan.local_name == "unsloth/Qwen3-8B-GGUF:Q4_K_M"


def test_select_quant_variant_gguf_all_keeps_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _choose(monkeypatch, None)
    files = {"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200}
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files=dict(files),
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    assert _select_quant_variant(plan, None) is None
    assert plan.files == files


@pytest.fixture
def download_cli(tmp_path, monkeypatch):
    """Use the real plan/adapter/CLI pipeline, replacing only Hub network I/O."""
    from typer.testing import CliRunner

    from drove.cli.models import models_app
    from drove.config import Config

    def run(plan, *, failure=None, extra_args=()):
        def resolve(ref, name_override=None, tts_engine=None):
            return plan

        def fetch(*, repo_id, filename, local_dir, tqdm_class):
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            size = plan._all_remote_files()[filename]
            with tqdm_class(total=size, initial=0, disable=True, name="test") as bar:
                if failure and filename == sorted(plan.files)[-1]:
                    # Keep a partial artifact as Hub would.
                    (path.parent / "partial.incomplete").write_bytes(b"partial")
                    bar.update(1)
                    raise failure
                path.write_bytes(b"x" * size)
                bar.update(size)
            return str(path)

        monkeypatch.setattr("drove.downloader.resolve_download", resolve)
        monkeypatch.setattr("drove.downloader.hf_hub_download", fetch)
        return CliRunner().invoke(
            models_app,
            ["download", plan.repo_id, "--yes", *extra_args],
            obj={"config": Config(models_dir=tmp_path)},
        )

    return run


def test_download_summary_and_existing_path_preserve_config(tmp_path, download_cli):
    from drove.model_config import load_model_config, set_model_config_key

    plan = DownloadPlan(
        "org/repo", {"m.gguf": 10}, "my model", False, mmproj_files={"mmproj.gguf": 4}
    )
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "Download complete" in result.output
    assert "Size on disk" in result.output and "mmproj.gguf" in result.output
    assert "Next: drove chat 'my model'" in result.output
    primary = plan.destination(tmp_path) / "m.gguf"
    assert load_model_config(primary).mmproj == "mmproj.gguf"
    size = sum(f.stat().st_size for f in primary.parent.rglob("*") if f.is_file())
    assert f"{size:,} bytes" in result.output
    set_model_config_key(primary, "mmproj", "custom.gguf")
    before = primary.with_suffix(".toml").read_bytes()
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "Already fully downloaded" in result.output and "custom.gguf" in result.output
    assert primary.with_suffix(".toml").read_bytes() == before
    assert "\x1b" not in result.output and "\r" not in result.output


@pytest.mark.parametrize("failure,exit_code", [(KeyboardInterrupt(), 130), (OSError("offline"), 1)])
def test_download_failures_keep_data_and_print_exact_retry(
    tmp_path, download_cli, failure, exit_code
):
    import shlex

    plan = DownloadPlan("org/repo", {"a.gguf": 10, "z.gguf": 10}, "my model", False)
    result = download_cli(plan, failure=failure, extra_args=("--name", "my model"))
    assert result.exit_code == exit_code, result.output
    assert "Download complete" not in result.output
    assert "Traceback" not in result.output
    assert (plan.destination(tmp_path) / "a.gguf").read_bytes() == b"x" * 10
    assert (plan.destination(tmp_path) / "partial.incomplete").read_bytes() == b"partial"
    assert "where supported" in result.output
    retry_line = next(
        line for line in result.output.splitlines() if line.startswith(("Resume:", "Retry:"))
    )
    assert shlex.split(retry_line.split(": ", 1)[1]) == [
        "drove",
        "models",
        "download",
        "org/repo",
        "--name",
        "my model",
        "--yes",
    ]
    if exit_code == 1:
        assert "Download failed for z.gguf" in result.output
    # A second invocation reuses the first completed file and succeeds.
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "a.gguf (skipped)" in result.output


def test_configuration_failure_is_not_reported_as_transfer_failure(download_cli, monkeypatch):
    plan = DownloadPlan("org/repo", {"m.gguf": 10}, "model", False)

    def fail(*args, **kwargs):
        raise OSError("read-only sidecar")

    monkeypatch.setattr("drove.cli.models.save_download_info", fail)
    result = download_cli(plan)
    assert result.exit_code == 1
    assert "Files downloaded, but model configuration failed" in result.output
    assert "Download complete" not in result.output and "Traceback" not in result.output


@pytest.mark.parametrize("known", [True, False])
def test_asr_summary_has_configuration_or_actionable_warning(download_cli, known):
    repo = "istupakov/parakeet-tdt-0.6b-v3-onnx" if known else "org/unknown"
    plan = DownloadPlan(repo, {"model.onnx": 10}, "asr", False)
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "asr_model" in result.output
    assert "Next: drove models info asr" in result.output
    assert "drove chat" not in result.output
    if known:
        assert "nemo-parakeet-tdt-0.6b-v3" in result.output
    else:
        assert "drove models config asr asr_model <type>" in result.output


def test_tts_summary_and_registration_failure(tmp_path, download_cli, monkeypatch):
    plan = DownloadPlan("org/tts", {"weights.bin": 10}, "voice", False, tts_engine="kokoro")
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "tts_engine" in result.output and "kokoro" in result.output
    assert "Next: drove models info voice" in result.output
    primary = plan.destination(tmp_path) / "model.tts"
    primary.unlink()

    def fail(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("drove.model_config.save_model_config", fail)
    result = download_cli(plan)
    assert result.exit_code == 1
    assert "configuration failed" in result.output
    assert not primary.exists()


def test_retry_command_quotes_selected_quant_name_and_explicit_engine():
    import shlex

    from drove.cli.models import _download_retry_command

    plan = DownloadPlan("org/repo", {"m.gguf": 10}, "my model", False)
    assert shlex.split(_download_retry_command(plan, "Q4_K_M", "it's mine", "kokoro")) == [
        "drove",
        "models",
        "download",
        "org/repo:Q4_K_M",
        "--name",
        "it's mine",
        "--tts-engine",
        "kokoro",
        "--yes",
    ]


def test_retry_for_all_onnx_variants_preserves_picker(monkeypatch):
    from drove.cli.models import _download_retry_command

    plan = _onnx_plan()
    _choose(monkeypatch, None)
    _select_quant_variant(plan, None)
    command = _download_retry_command(plan, None, None, None)
    assert "--yes" not in command  # --yes would silently select only the default variant.
    assert command == "drove models download " + plan.repo_id


@pytest.mark.parametrize("kind", ["projector", "asr", "metadata"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_printed_configuration_retry_repairs_missing_settings(
    tmp_path, download_cli, monkeypatch, kind, interrupt
):
    import shlex

    from typer.testing import CliRunner

    from drove.cli import models
    from drove.config import Config
    from drove.model_config import load_download_info, load_model_config, set_model_config_key

    if kind == "asr":
        plan = DownloadPlan(
            "istupakov/parakeet-tdt-0.6b-v3-onnx", {"model.onnx": 10}, "model", False
        )
    else:
        plan = DownloadPlan(
            "org/repo",
            {"model.gguf": 10},
            "model",
            False,
            mmproj_files={"mmproj.gguf": 3},
        )
    writer_name = "save_download_info" if kind == "metadata" else "save_model_config"
    writer = getattr(models, writer_name)
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            if interrupt:
                raise KeyboardInterrupt
            raise OSError("configuration storage temporarily unavailable")
        return writer(*args, **kwargs)

    monkeypatch.setattr(models, writer_name, fail_once)
    first = download_cli(plan)
    assert first.exit_code == (130 if interrupt else 1), first.output
    assert ("interrupted" if interrupt else "configuration failed") in first.output
    command = next(
        line.split(": ", 1)[1]
        for line in first.output.splitlines()
        if line.startswith(("Retry: ", "Resume: "))
    )
    primary = plan.destination(tmp_path) / next(iter(plan.files))
    set_model_config_key(primary, "ctx_size", "1234")
    # Preserve independent user choices made before retry, even if inference disagrees.
    if kind == "metadata":
        set_model_config_key(primary, "mmproj", "custom-projector.gguf")

    def no_network(**kwargs):
        pytest.fail("repair must reuse every complete payload")

    monkeypatch.setattr("drove.downloader.hf_hub_download", no_network)
    second = CliRunner().invoke(
        models.models_app, shlex.split(command)[2:], obj={"config": Config(models_dir=tmp_path)}
    )
    assert second.exit_code == 0, second.output
    assert len(calls) == 2
    cfg = load_model_config(primary)
    assert cfg.ctx_size == 1234
    assert load_download_info(primary).repo_id == plan.repo_id
    if kind == "asr":
        assert cfg.asr_model == "nemo-parakeet-tdt-0.6b-v3"
    else:
        assert cfg.mmproj == ("custom-projector.gguf" if kind == "metadata" else "mmproj.gguf")
    assert "Download complete" in second.output
