"""Tests for the `drove models` CLI helpers."""

from __future__ import annotations

from pathlib import Path

import click
import httpx
import pytest
import typer
from typer.testing import CliRunner

from drove.cli.main import app
from drove.cli.models import (
    _deletion_preview,
    _detect_capabilities,
    _loaded_warning,
    _select_quant_variant,
)
from drove.cli.quant_picker import ALL_VARIANTS
from drove.config import Config
from drove.downloader import DownloadPlan, FileStatus
from drove.model_config import DownloadInfo, save_download_info


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
    monkeypatch.setattr(
        "drove.cli.models._prompt_quant_choice",
        lambda candidates, theme: ALL_VARIANTS if answer is None else answer,
    )


def test_select_quant_variant_onnx_int8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None, tmp_path)
    assert quant == "int8"
    assert plan.files == _ONNX_INT8
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx:int8"


def test_select_quant_variant_onnx_default_keeps_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _choose(monkeypatch, "default")
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None, tmp_path)
    assert quant is None
    assert plan.files == _ONNX_DEFAULT
    assert plan.local_name == "istupakov/parakeet-tdt-0.6b-v3-onnx"


def test_select_quant_variant_onnx_all_takes_every_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _choose(monkeypatch, None)
    plan = _onnx_plan()
    quant = _select_quant_variant(plan, None, tmp_path)
    assert quant is None
    assert plan.files == _ONNX_ALL


def test_select_quant_variant_onnx_keeps_name_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _choose(monkeypatch, "int8")
    plan = _onnx_plan()
    plan.local_name = "my-parakeet"
    quant = _select_quant_variant(plan, "my-parakeet", tmp_path)
    assert quant == "int8"
    assert plan.local_name == "my-parakeet"


def test_select_quant_variant_onnx_single_variant_skips_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    assert _select_quant_variant(plan, None, tmp_path) is None
    assert plan.files == _ONNX_DEFAULT


def test_select_quant_variant_gguf_choice_filters_and_renames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _choose(monkeypatch, "Q4_K_M")
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files={"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200},
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    quant = _select_quant_variant(plan, None, tmp_path)
    assert quant == "Q4_K_M"
    assert plan.files == {"Qwen3-8B-Q4_K_M.gguf": 100}
    assert plan.local_name == "unsloth/Qwen3-8B-GGUF:Q4_K_M"


def test_select_quant_variant_gguf_all_keeps_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _choose(monkeypatch, None)
    files = {"Qwen3-8B-Q4_K_M.gguf": 100, "Qwen3-8B-Q8_0.gguf": 200}
    plan = DownloadPlan(
        repo_id="unsloth/Qwen3-8B-GGUF",
        files=dict(files),
        local_name="unsloth/Qwen3-8B-GGUF",
        sharded=False,
    )
    assert _select_quant_variant(plan, None, tmp_path) is None
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
        "org/repo", {"m.gguf": 10}, "my-model", False, mmproj_files={"mmproj.gguf": 4}
    )
    result = download_cli(plan)
    assert result.exit_code == 0, result.output
    assert "Download complete" in result.output
    assert "Size on disk" in result.output and "mmproj.gguf" in result.output
    assert "Next: drove chat my-model" in result.output
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

    plan = DownloadPlan("org/repo", {"a.gguf": 10, "z.gguf": 10}, "my-model", False)
    result = download_cli(plan, failure=failure, extra_args=("--name", "my-model"))
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
        "my-model",
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


def test_retry_for_all_onnx_variants_preserves_picker(monkeypatch, tmp_path):
    from drove.cli.models import _download_retry_command

    plan = _onnx_plan()
    _choose(monkeypatch, None)
    _select_quant_variant(plan, None, tmp_path)
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


# Contextual confirmations use real input; no test touches actual model storage.
@pytest.mark.parametrize("answer", ["\n", "n\n", ""])
def test_delete_refusal_preserves_inventory(tmp_path, monkeypatch, answer):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "model")
    primary.write_bytes(b"model")
    primary.with_suffix(".toml").write_text("ctx_size = 8\n")
    external = tmp_path / "model.toml"
    external.write_text("# external")
    monkeypatch.setattr(
        "drove.cli.models._loaded_warning", lambda *args: "Warning: loaded status unknown"
    )
    result = CliRunner().invoke(app, ["models", "delete", "model"], input=answer)
    assert result.exit_code != 0
    assert primary.exists() and external.exists()
    for text in [
        "model.gguf",
        "model.toml",
        "Estimated space",
        "Sidecar config",
        "unknown",
        "[y/n]",
        "(n)",
    ]:
        assert text in result.output


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_delete_yes_removes_exact_inventory_without_network(tmp_path, monkeypatch, flag):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "model")
    primary.write_bytes(b"12345")
    primary.with_suffix(".toml").write_bytes(b"123")
    external = tmp_path / "model.toml"
    external.write_bytes(b"12")
    kept = tmp_path / "other.gguf"
    kept.write_bytes(b"other")

    def unexpected(*args, **kwargs):
        raise AssertionError("no network or confirmation with --yes")

    monkeypatch.setattr("httpx.get", unexpected)
    monkeypatch.setattr("drove.cli.models.confirm_action", unexpected)
    result = CliRunner().invoke(app, ["models", "delete", "model", flag])
    assert result.exit_code == 0, result.output
    assert "10 bytes" in result.output
    assert not primary.parent.exists() and not external.exists() and kept.exists()


def test_deletion_inventory_alias_legacy_and_symlinks(tmp_path, monkeypatch):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "local")
    save_download_info(primary, DownloadInfo(repo_id="org/repo", files=[primary.name]))
    external = tmp_path / "local.toml"
    external.write_text("# external")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    protected = outside / "keep.txt"
    protected.write_bytes(b"preserved")
    (primary.parent / "link").symlink_to(outside, target_is_directory=True)
    preview = _deletion_preview(tmp_path, primary.parent)
    assert not any(path.name == "keep.txt" for path, _, _ in preview.files)
    result = CliRunner().invoke(app, ["models", "delete", "org/repo", "--yes"])
    assert result.exit_code == 0, result.output
    assert "symlink only" in result.output
    assert not primary.parent.exists() and not external.exists() and protected.exists()
    legacy = tmp_path / "legacy.gguf"
    legacy.write_bytes(b"GGUF")
    legacy.with_suffix(".toml").write_bytes(b"# cfg")
    result = CliRunner().invoke(app, ["models", "delete", "legacy", "--yes"])
    assert result.exit_code == 0
    assert not legacy.exists() and not legacy.with_suffix(".toml").exists()
    # A model root that resolves outside the store is not a model (#110):
    # deletion refuses it and leaves both the link and its target alone.
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(app, ["models", "delete", "linked", "--yes"])
    assert result.exit_code == 1, result.output
    assert link.is_symlink() and protected.exists()


@pytest.mark.parametrize("name", [".", "sub/..", "../outside"])
def test_delete_rejects_escape_or_storage_root(tmp_path, monkeypatch, name):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    (tmp_path / "sub").mkdir()
    (tmp_path.parent / "outside").mkdir(exist_ok=True)
    result = CliRunner().invoke(app, ["models", "delete", name, "--yes"])
    assert result.exit_code != 0
    assert tmp_path.exists() and (tmp_path.parent / "outside").exists()


def test_delete_inventory_error_preserves_files(tmp_path, monkeypatch):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "model")

    def inaccessible(self):
        raise PermissionError("cannot inspect files")

    monkeypatch.setattr(Path, "iterdir", inaccessible)
    result = CliRunner().invoke(app, ["models", "delete", "model", "--yes"])
    assert result.exit_code != 0 and "Cannot prepare deletion" in result.output
    assert primary.exists()


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"models": []}, "Not currently loaded"),
        ({"models": [{"name": "model", "active_requests": 2}]}, "2 active requests"),
        ({"models": [{"name": "model", "active_requests": 0}]}, "0 active requests"),
        ({"models": [{"name": "model"}]}, "active requests unknown"),
        ({"models": [{}]}, "status unknown"),
        ({"models": "wrong"}, "status unknown"),
        ([], "status unknown"),
    ],
)
def test_delete_loaded_status(tmp_path, monkeypatch, payload, expected):

    primary = _make_model(tmp_path, "model")

    def get(url, **kwargs):
        assert url == "http://[::1]:8080/status" and kwargs["timeout"] == 2.0
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    ctx = typer.Context(click.Command("test"))
    ctx.obj = {"config": Config(models_dir=tmp_path, listen_host="::", listen_port=8080)}
    assert expected in _loaded_warning(ctx, tmp_path, primary.parent)


@pytest.mark.parametrize("mode", ["timeout", "http", "json"])
def test_delete_unavailable_status_is_unknown(tmp_path, monkeypatch, mode):

    primary = _make_model(tmp_path, "model")

    def get(url, **kwargs):
        if mode == "timeout":
            raise httpx.ReadTimeout("offline")
        return httpx.Response(
            503 if mode == "http" else 200, content=b"bad", request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)
    ctx = typer.Context(click.Command("test"))
    ctx.obj = {"config": Config(models_dir=tmp_path)}
    assert "status unknown" in _loaded_warning(ctx, tmp_path, primary.parent)


@pytest.mark.parametrize(
    "resumed,verbose", [(False, False), (True, False), (False, True), (True, True)]
)
def test_download_confirmation_adjacent_and_refusal(tmp_path, monkeypatch, resumed, verbose):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    plan = DownloadPlan(
        repo_id="org/repo", local_name="model", sharded=False, files={"model.gguf": 2 * 1048576}
    )
    monkeypatch.setattr("drove.downloader.resolve_download", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        DownloadPlan,
        "check_local_files",
        lambda *args: {"model.gguf": (FileStatus.INCOMPLETE, 1048576)} if resumed else {},
    )

    def never(*args, **kwargs):
        raise AssertionError("declined download must not execute")

    monkeypatch.setattr(DownloadPlan, "execute", never)
    args = ["models", "download", "org/repo:Q4"] + (["--verbose"] if verbose else [])
    result = CliRunner().invoke(app, args, input="n\n")
    assert result.exit_code != 0
    lines = result.output.splitlines()
    summary = next(i for i, line in enumerate(lines) if "To download" in line)
    assert "approximately" in lines[summary + 1]
    assert ("1.0 MB" if resumed else "2.0 MB") in lines[summary + 1]
    assert ("Resume download" if resumed else "Download approximately") in lines[summary + 1]


@pytest.mark.parametrize("flag", ["--yes", "-y"])
@pytest.mark.parametrize("resumed", [False, True])
def test_download_yes_bypasses_all_prompts(tmp_path, monkeypatch, flag, resumed):

    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    plan = DownloadPlan(
        repo_id="org/repo", local_name="model", sharded=False, files={"model.gguf": 4}
    )
    monkeypatch.setattr("drove.downloader.resolve_download", lambda *args, **kwargs: plan)
    if resumed:
        target = tmp_path / "model" / "model.gguf"
        target.parent.mkdir()
        target.write_bytes(b"GG")
    called = []

    def execute(self, models_dir, **kwargs):
        called.append(1)
        target = models_dir / "model" / "model.gguf"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"GGUF")
        return target

    def never(*args, **kwargs):
        raise AssertionError("prompt not allowed")

    monkeypatch.setattr(DownloadPlan, "execute", execute)
    monkeypatch.setattr("drove.cli.models.confirm_action", never)
    monkeypatch.setattr("drove.cli.models._select_quant_variant", never)
    result = CliRunner().invoke(app, ["models", "download", "org/repo", flag])
    assert result.exit_code == 0, result.output
    assert called == [1]


def test_delete_loaded_alias_warning_then_explicit_consent(tmp_path, monkeypatch):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "local")
    save_download_info(primary, DownloadInfo(repo_id="org/repo", files=[primary.name]))

    def get(url, **kwargs):
        return httpx.Response(
            200,
            json={"models": [{"name": "org/repo", "active_requests": 3}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    result = CliRunner().invoke(app, ["models", "delete", "local"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "currently loaded" in result.output and "3 active requests" in result.output
    assert "does not stop it" in " ".join(result.output.split())
    assert not primary.parent.exists()


def test_delete_symlink_namespace_does_not_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    outside = tmp_path.parent / (tmp_path.name + "-namespace")
    outside.mkdir()
    primary = _make_model(outside, "model")
    (tmp_path / "org").symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(app, ["models", "delete", "org/model", "--yes"])
    assert result.exit_code != 0 and primary.exists()


def test_kitten_notice_precedes_final_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    plan = DownloadPlan(
        repo_id="org/repo",
        local_name="kitten",
        files={"model.onnx": 1048576},
        sharded=False,
        tts_engine="kitten",
    )
    monkeypatch.setattr("drove.downloader.resolve_download", lambda *a, **kw: plan)
    result = CliRunner().invoke(
        app, ["models", "download", "org/repo:Q4", "--verbose"], input="n\n"
    )
    assert result.exit_code != 0
    assert result.output.index("Community License") < result.output.index("To download")
    assert result.output.index("To download") < result.output.index("approximately 1.0 MB")


@pytest.mark.parametrize("tts", [False, True])
def test_complete_download_preserves_no_prompt_path(tmp_path, monkeypatch, tts):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    primary = _make_model(tmp_path, "model")
    primary.write_bytes(b"GGUF")
    plan = DownloadPlan(
        repo_id="org/repo",
        local_name="model",
        files={"model.gguf": 4},
        sharded=False,
        tts_engine="kokoro" if tts else None,
    )
    monkeypatch.setattr("drove.downloader.resolve_download", lambda *a, **kw: plan)
    monkeypatch.setattr(
        DownloadPlan, "check_local_files", lambda *a: {"model.gguf": (FileStatus.COMPLETE, 4)}
    )
    called = []

    def execute(*a, **kw):
        called.append(1)
        return primary

    def never(*a, **kw):
        raise AssertionError("complete download does not ask")

    monkeypatch.setattr(DownloadPlan, "execute", execute)
    monkeypatch.setattr("drove.cli.models.confirm_action", never)
    result = CliRunner().invoke(app, ["models", "download", "org/repo:Q4"])
    assert result.exit_code == 0, result.output
    assert called == ([1] if tts else [])


@pytest.mark.parametrize(
    "answer,flag,deleted", [("n\n", None, False), ("y\n", None, True), (None, "--yes", True)]
)
@pytest.mark.parametrize("linked_base", [False, True])
def test_delete_relative_models_directory(
    tmp_path, monkeypatch, answer, flag, deleted, linked_base
):
    monkeypatch.chdir(tmp_path)
    storage = tmp_path / "storage" if linked_base else tmp_path / "models"
    storage.mkdir()
    if linked_base:
        (tmp_path / "models").symlink_to(storage, target_is_directory=True)
    primary = _make_model(storage, "local")
    primary.with_suffix(".toml").write_text("ctx_size = 8\n")
    monkeypatch.setenv("DROVE_MODELS_DIR", "models")
    monkeypatch.setattr("drove.cli.models._loaded_warning", lambda *args: "loaded status unknown")
    args = ["models", "delete", "local"] + ([flag] if flag else [])
    result = CliRunner().invoke(app, args, input=answer)
    assert result.exit_code == (0 if deleted else 1), result.output
    assert "local/local.gguf" in result.output and "local/local.toml" in result.output
    assert primary.exists() is not deleted


@pytest.mark.parametrize("alias", [False, True])
def test_namespace_delete_warns_for_loaded_descendant(tmp_path, monkeypatch, alias):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    model = tmp_path / "org" / "child"
    model.mkdir(parents=True)
    primary = model / "child.gguf"
    primary.write_bytes(b"GGUF")
    save_download_info(primary, DownloadInfo(repo_id="upstream/repo", files=[primary.name]))
    loaded_name = "upstream/repo" if alias else "org/child"

    def get(url, **kwargs):
        return httpx.Response(
            200,
            json={"models": [{"name": loaded_name, "active_requests": 2}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    result = CliRunner().invoke(app, ["models", "delete", "org"], input="n\n")
    assert result.exit_code == 1, result.output
    assert "currently loaded:" in result.output
    assert loaded_name in result.output and "2 active requests" in result.output
    assert "Not currently loaded" not in result.output
    assert primary.exists()


def test_symlink_namespace_outside_store_is_not_previewed_or_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    outside = tmp_path.parent / (tmp_path.name + "-contents")
    primary = _make_model(outside, "child")
    (tmp_path / "org").symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(app, ["models", "delete", "org"], input="y\n")
    assert result.exit_code == 1
    assert "Delete model" not in result.output
    assert (tmp_path / "org").is_symlink()
    assert primary.exists()


@pytest.mark.parametrize("stage", ["transfer", "configuration"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_recovery_commands_preserve_custom_config(tmp_path, monkeypatch, stage, interrupt):
    import shlex

    import drove.cli.models as cli_models

    models_dir = tmp_path / "custom models"
    config_path = tmp_path / "custom config's.toml"
    config_path.write_text(f'models_dir = "{models_dir}"\n')
    plan = DownloadPlan("org/repo", {"model.gguf": 4}, "local-model", False)
    monkeypatch.setattr("drove.downloader.resolve_download", lambda *a, **k: plan)
    destinations = []
    failed = False
    finish = cli_models._finish_download

    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            if interrupt:
                raise KeyboardInterrupt
            raise OSError("simulated failure")

    def execute(self, destination, **kwargs):
        destinations.append(destination)
        if stage == "transfer":
            fail_once()
        primary = self.destination(destination) / "model.gguf"
        primary.parent.mkdir(parents=True, exist_ok=True)
        primary.write_bytes(b"gguf")
        return primary

    def configure(*args, **kwargs):
        if stage == "configuration":
            fail_once()
        return finish(*args, **kwargs)

    monkeypatch.setattr(DownloadPlan, "execute", execute)
    monkeypatch.setattr(cli_models, "_finish_download", configure)
    runner = CliRunner()
    first = runner.invoke(
        app, ["--config", str(config_path), "models", "download", plan.repo_id, "--yes"]
    )
    assert first.exit_code == (130 if interrupt else 1), first.output
    label = "Resume: " if interrupt else "Retry: "
    command = next(
        line.removeprefix(label) for line in first.output.splitlines() if line.startswith(label)
    )
    args = shlex.split(command)
    assert args[:3] == ["drove", "--config", str(config_path)]
    if stage == "configuration":
        assert "--repair-config" in args
        if not interrupt:
            inspect = next(
                line.removeprefix("Inspect configuration: ")
                for line in first.output.splitlines()
                if line.startswith("Inspect configuration: ")
            )
            assert shlex.split(inspect) == [
                "drove",
                "--config",
                str(config_path),
                "models",
                "config",
                plan.local_name,
            ]
    retried = runner.invoke(app, args[1:])
    assert retried.exit_code == 0, retried.output
    assert destinations == [models_dir, models_dir]
    assert (plan.destination(models_dir) / "model.gguf").is_file()


# ── model name validation (#110) ─────────────────────────────────────────────

_UNSAFE_CLI_NAMES = ["", ".", "..", "/", "/tmp/x", "../x", "sub/..", "./m", "m/", "a//b", "a\\b"]


def _sentinel_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    models = tmp_path / "models"
    primary = _make_model(models, "keep")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    sibling = tmp_path / "sessions" / "s.json"
    sibling.parent.mkdir()
    sibling.write_text("{}")
    return models, primary, outside


@pytest.mark.parametrize("name", _UNSAFE_CLI_NAMES)
@pytest.mark.parametrize(
    "command",
    [
        ["models", "delete", "{name}", "--yes"],
        ["models", "info", "{name}"],
        ["models", "config", "{name}"],
        ["models", "config", "{name}", "ctx_size", "1024"],
        ["models", "config", "{name}", "--unset", "ctx_size"],
    ],
)
def test_model_commands_reject_unsafe_names(tmp_path, monkeypatch, name, command):
    models, primary, outside = _sentinel_tree(tmp_path)
    monkeypatch.setenv("DROVE_MODELS_DIR", str(models))
    args = [part.replace("{name}", name) for part in command]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert "model name" in result.output.lower()
    assert primary.exists() and outside.read_text() == "keep"
    assert (tmp_path / "sessions" / "s.json").exists()
    assert not primary.with_suffix(".toml").exists()


@pytest.mark.parametrize("name", [*_UNSAFE_CLI_NAMES, "my model"])
def test_download_rejects_unsafe_name_before_any_work(tmp_path, monkeypatch, name):
    from drove.cli.models import models_app

    models = tmp_path / "models"

    def forbidden(*args, **kwargs):
        raise AssertionError("download work started for an invalid --name")

    monkeypatch.setattr("drove.downloader.resolve_download", forbidden)
    monkeypatch.setattr("drove.downloader.hf_hub_download", forbidden)
    result = CliRunner().invoke(
        models_app,
        ["download", "org/repo", "--name", name, "--yes"],
        obj={"config": Config(models_dir=models)},
    )
    assert result.exit_code == 1, result.output
    assert "model name" in result.output.lower()
    assert not models.exists()
    assert not (tmp_path / "x").exists()


def test_download_plan_destination_rejects_mutated_or_escaping_name(tmp_path):
    plan = DownloadPlan("org/repo", {"m.gguf": 10}, "ok", False)
    assert plan.destination(tmp_path) == tmp_path / "ok"
    plan.local_name = "../x"
    with pytest.raises(ValueError, match="relative"):
        plan.destination(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-dest")
    outside.mkdir()
    (tmp_path / "org").symlink_to(outside, target_is_directory=True)
    plan.local_name = "org/repo"
    with pytest.raises(ValueError, match="outside"):
        plan.destination(tmp_path)


@pytest.mark.parametrize("name", ["org/kokoro", "kokoro:v1", "../x", ""])
def test_add_tts_rejects_names_outside_the_flat_rule(tmp_path, monkeypatch, name):
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["models", "add-tts", name])
    assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []
