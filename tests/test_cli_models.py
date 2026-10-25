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
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    result = CliRunner().invoke(app, ["models", "delete", "linked", "--yes"])
    assert result.exit_code == 0, result.output
    assert not link.exists() and protected.exists()


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
