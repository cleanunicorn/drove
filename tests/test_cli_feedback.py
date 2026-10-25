"""Human feedback contracts and machine-output compatibility."""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from drove.cli import completions
from drove.cli.main import app
from drove.cli.output import error, info, notice, success, unknown, warning


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "coder.gguf").write_bytes(b"GGUF")
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'models_dir = "{models}"\nobserve_dir = "{tmp_path / "logs"}"\n')
    monkeypatch.setenv("DROVE_MODELS_DIR", str(models))
    monkeypatch.delenv("DROVE_OBSERVE", raising=False)
    monkeypatch.delenv("DROVE_OBSERVE_DIR", raising=False)
    return CliRunner(), cfg, models


def test_feedback_labels_streams_and_literal_text(capsys) -> None:
    success("[red]literal[/red]", next_step="Review values.")
    info("Working.")
    warning("Missing.", next_step="Retry.")
    error("Failed.")
    captured = capsys.readouterr()
    assert "Success: [red]literal[/red]" in captured.out
    assert "Info: Working." in captured.out
    assert "Next: Review values." in captured.out
    assert "Warning: Missing." in captured.err
    assert "Error: Failed." in captured.err
    assert "Next: Retry." in captured.err
    assert "\x1b" not in captured.out + captured.err


@pytest.mark.parametrize("no_color", [False, True])
def test_feedback_at_narrow_terminal_width(no_color: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    out = Console(file=stream, width=40, force_terminal=True, no_color=no_color)
    notice(
        "Warning", "A [blue]literal[/blue] name " * 2, next_step="Review configuration.", out=out
    )
    rendered = stream.getvalue()
    assert "Warning:" in rendered and "Next:" in rendered
    assert "[blue]literal[/blue]" in rendered
    if no_color:
        # Rich may retain bold styling under NO_COLOR; color itself is disabled.
        assert out.no_color
        assert "\x1b[33m" not in rendered and "\x1b[1;33m" not in rendered


def test_unknown_uses_plausible_deterministic_candidates() -> None:
    assert "ctx_size" in unknown("key", "ctx_siz", ["threads", "ctx_size"])
    assert "Did you mean" not in unknown("key", "zzzzzz", ["threads", "ctx_size"])
    assert unknown("model", "anything", []) == "Unknown model 'anything'."


@pytest.mark.parametrize(
    "args,expected",
    [
        (["config", "listen_por"], "listen_port"),
        (["config", "llama_server.n_gpu_layer", "1"], "llama_server.n_gpu_layers"),
        (["models", "config", "--global", "ctx_siz"], "ctx_size"),
        (["models", "config", "--global", "ctx_siz", "12"], "ctx_size"),
        (["models", "config", "--global", "--unset", "ctx_siz"], "ctx_size"),
        (["models", "config", "coder", "ctx_siz"], "ctx_size"),
        (["models", "config", "coder", "ctx_siz", "12"], "ctx_size"),
        (["models", "config", "coder", "--unset", "ctx_siz"], "ctx_size"),
        (["models", "info", "codr"], "coder"),
        (["models", "delete", "codr", "--yes"], "coder"),
        (["models", "config", "codr"], "coder"),
    ],
)
def test_unknown_keys_and_models_are_actionable_without_writes(cli, args, expected) -> None:
    runner, cfg, models = cli
    original = cfg.read_text()
    result = runner.invoke(app, ["-c", str(cfg), *args])
    assert result.exit_code == 1, result.output
    assert "Error:" in result.stderr and "Did you mean" in result.stderr
    assert expected in result.stderr and "Next:" in result.stderr
    assert "List" in result.stderr
    assert cfg.read_text() == original
    assert sorted(p.name for p in models.iterdir()) == ["coder.gguf"]


@pytest.mark.parametrize(
    "args,expected",
    [
        (["config", "observe"], "False\n"),
        (["config", "llama_server.n_gpu_layers"], "-1\n"),
        (["models", "config", "coder", "ctx_size"], "(not set)\n"),
        (["models", "config", "--global", "ctx_size"], "(not set)\n"),
    ],
)
def test_valid_getters_keep_exact_plain_output(cli, args, expected) -> None:
    runner, cfg, _ = cli
    result = runner.invoke(app, ["-c", str(cfg), *args], color=True)
    assert result.exit_code == 0, result.output
    assert result.stdout == expected
    assert result.stderr == ""


@pytest.mark.parametrize("global_mode", [False, True])
def test_model_config_mutations_confirm_and_getters_remain_raw(cli, global_mode) -> None:
    runner, cfg, models = cli
    target = ["--global"] if global_mode else ["coder"]
    prefix = ["-c", str(cfg), "models", "config", *target]
    result = runner.invoke(app, [*prefix, "ctx_size", "8192"])
    assert result.exit_code == 0 and "Success:" in result.stdout and "Next:" in result.stdout
    path = models / ("_global.toml" if global_mode else "coder.toml")
    assert "ctx_size = 8192" in path.read_text()
    assert runner.invoke(app, [*prefix, "ctx_size"]).stdout == "8192\n"
    result = runner.invoke(app, [*prefix, "--unset", "ctx_size"])
    assert result.exit_code == 0 and "Success:" in result.stdout and "Unset" in result.stdout
    assert runner.invoke(app, [*prefix, "ctx_size"]).stdout == "(not set)\n"


def test_init_summary_and_root_mutation(cli) -> None:
    runner, cfg, _ = cli
    result = runner.invoke(app, ["-c", str(cfg), "init", "--force"])
    assert result.exit_code == 0 and "Success:" in result.stdout
    assert "Models" in result.stdout and "Listen" in result.stdout
    assert "[llama_server]" not in result.stdout
    assert "[llama_server]" in cfg.read_text()
    result = runner.invoke(app, ["-c", str(cfg), "config", "listen_port", "9000"])
    assert result.exit_code == 0 and "Success:" in result.stdout and "Next:" in result.stdout
    assert "listen_port = 9000" in cfg.read_text()
    refused = runner.invoke(app, ["-c", str(cfg), "init"])
    assert refused.exit_code == 1 and "Warning:" in refused.stderr
    assert "--force" in refused.stderr and "Success:" not in refused.output


def test_delete_confirms_only_after_all_removals(cli, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, cfg, models = cli
    sidecar = models / "coder.toml"
    sidecar.write_text("ctx_size = 42\n")
    result = runner.invoke(app, ["-c", str(cfg), "models", "delete", "coder"], input="n\n")
    assert result.exit_code != 0 and "Success:" not in result.output
    assert (models / "coder.gguf").exists() and sidecar.exists()
    result = runner.invoke(app, ["-c", str(cfg), "models", "delete", "coder", "--yes"])
    assert result.exit_code == 0 and "Success:" in result.stdout and "Next:" in result.stdout
    assert not sidecar.exists() and not (models / "coder.gguf").exists()


def test_add_tts_confirms_registration(cli) -> None:
    runner, cfg, models = cli
    result = runner.invoke(
        app, ["-c", str(cfg), "models", "add-tts", "voice", "--engine", "kokoro"]
    )
    assert result.exit_code == 0, result.output
    assert "Success:" in result.stdout and "POST /v1/audio/speech" in result.stdout
    assert (models / "voice" / "model.tts").exists()


@pytest.mark.parametrize("web", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_observe_guidance_distinguishes_disabled_and_empty(cli, web, enabled) -> None:
    runner, cfg, _ = cli
    with cfg.open("a") as f:
        f.write(f"observe = {str(enabled).lower()}\n")
    result = runner.invoke(app, ["-c", str(cfg), "observe", *(["web"] if web else [])])
    assert result.exit_code == 1 and "Warning:" in result.stderr
    assert ("config observe true" in result.stderr) is not enabled
    assert "requests" in result.stderr


def test_server_connection_error_is_actionable(cli, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, cfg, _ = cli
    monkeypatch.setattr(httpx, "get", Mock(side_effect=httpx.ConnectError("offline")))
    result = runner.invoke(app, ["-c", str(cfg), "server", "status"])
    assert result.exit_code == 1 and "Warning:" in result.stderr
    assert "drove server" in result.stderr


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish", "powershell"])
def test_completions_confirm_real_writes_with_shell_specific_next_step(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shell: str
) -> None:
    runner, cfg, _ = cli
    dest = tmp_path / "completion" / shell
    rc = tmp_path / "rc" / shell
    monkeypatch.setattr(completions, "_generate_script", lambda _: "# exact script\n")
    monkeypatch.setattr(completions, "_completion_path", lambda _: dest)
    monkeypatch.setattr(completions, "_SHELL_RC", {} if shell == "fish" else {shell: rc})
    result = runner.invoke(app, ["-c", str(cfg), "completions", "install", shell])
    assert result.exit_code == 0, result.output
    assert dest.read_text() == "# exact script\n"
    assert "Success:" in result.stdout and "Next:" in result.stdout
    if shell == "fish":
        assert "automatically" in result.stdout and ".zshrc" not in result.stdout
        assert not rc.exists()
    else:
        assert rc.exists()
        assert (
            "source" in result.stdout
            if shell != "powershell"
            else "Restart PowerShell" in result.stdout
        )
    again = runner.invoke(app, ["-c", str(cfg), "completions", "install", shell])
    assert again.exit_code == 0 and "Success:" in again.stdout
    if shell != "fish":
        assert rc.read_text().count("# drove shell completions") == 1


def test_completion_script_and_dry_run_keep_exact_plain_output(cli, tmp_path, monkeypatch) -> None:
    runner, cfg, _ = cli
    dest, rc = tmp_path / "completion", tmp_path / "shellrc"
    script = "# [red]literal[/red]\ncomplete something\n"
    monkeypatch.setattr(completions, "_generate_script", lambda _: script)
    monkeypatch.setattr(completions, "_completion_path", lambda _: dest)
    monkeypatch.setattr(completions, "_SHELL_RC", {"bash": rc})
    result = runner.invoke(app, ["-c", str(cfg), "completions", "generate", "bash"], color=True)
    assert result.stdout == script and result.stderr == ""
    dry = runner.invoke(
        app, ["-c", str(cfg), "completions", "install", "bash", "--dry-run"], color=True
    )
    assert dry.stdout == (
        f"Writing completion script → {dest}\n\n"
        f"Add the following to {rc} to activate completions:\n\n"
        "    source ~/.bash_completions/drove\n\n\n"
        f"Restart your shell or run:  source {rc}\n"
    )
    assert not dest.exists() and not rc.exists()


def test_failed_completion_write_does_not_report_success(cli, tmp_path, monkeypatch) -> None:
    runner, cfg, _ = cli
    dest = tmp_path / "directory"
    dest.mkdir()
    monkeypatch.setattr(completions, "_generate_script", lambda _: "script")
    monkeypatch.setattr(completions, "_completion_path", lambda _: dest)
    result = runner.invoke(app, ["-c", str(cfg), "completions", "install", "fish"])
    assert result.exit_code == 1 and "Error:" in result.stderr
    assert "Success:" not in result.output


def test_failed_config_write_reports_error_without_success(cli, monkeypatch) -> None:
    from drove.config import Config

    runner, cfg, _ = cli
    original = cfg.read_text()
    monkeypatch.setattr(Config, "save", Mock(side_effect=OSError("read-only")))
    for args in (["init", "--force"], ["config", "listen_port", "9000"]):
        result = runner.invoke(app, ["-c", str(cfg), *args])
        assert result.exit_code == 1 and "Error:" in result.stderr
        assert "Success:" not in result.output
        assert cfg.read_text() == original


def test_partial_delete_failure_never_claims_success(cli, monkeypatch) -> None:
    runner, cfg, models = cli
    sidecar = models / "coder.toml"
    sidecar.write_text("ctx_size = 42\n")
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == sidecar:
            raise PermissionError("sidecar read-only")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    result = runner.invoke(app, ["-c", str(cfg), "models", "delete", "coder", "--yes"])
    assert result.exit_code == 1 and "Error:" in result.stderr
    assert "Could not finish deleting" in result.stderr and "Success:" not in result.output
    assert sidecar.exists() and not (models / "coder.gguf").exists()
