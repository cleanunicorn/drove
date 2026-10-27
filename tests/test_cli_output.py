"""Regression coverage for CLI reports and terminal-safe rendering."""

from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

from drove.cli.main import app
from drove.cli.models import _print_download_plan, _print_resolved_config
from drove.cli.output import fields, heading, table
from drove.cli.server import _exposure_note, _print_status
from drove.downloader import DownloadPlan, FileStatus


def test_reports_preserve_literal_values_at_narrow_width() -> None:
    stream = StringIO()
    out = Console(file=stream, width=40, color_system=None)
    heading(out, "Models", "[red]literal[/red]")
    fields(out, [("Path", "/models/[blue]literal[/blue]")])
    listing = table("NAME", "SIZE")
    listing.add_row(Text("org/" + "long-model-" * 5), "4.2 GB")
    out.print(listing)
    rendered = stream.getvalue()
    assert "[red]literal[/red]" in rendered
    assert "[blue]literal[/blue]" in rendered
    assert "org/" + "long-model-" * 5 in "".join(rendered.replace("4.2 GB", "").split())
    assert all(len(line) <= 40 for line in rendered.splitlines())
    assert "\x1b" not in rendered


def _unwrapped(text: str) -> str:
    """Drop whitespace so assertions survive Rich wrapping long values."""
    return "".join(text.split())


def test_model_list_empty_and_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Model names can't hold markup (#110); the storage path still can and
    # must print literally.
    models_dir = tmp_path / "[red]store"
    models_dir.mkdir()
    monkeypatch.setenv("DROVE_MODELS_DIR", str(models_dir))
    runner = CliRunner()
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "drove models download" in result.stdout
    primary = models_dir / "plain-model.gguf"
    primary.write_bytes(b"GGUF")
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "plain-model" in result.stdout
    assert "CONFIG" not in result.stdout
    assert str(models_dir) not in result.stdout
    detailed = runner.invoke(app, ["models", "list", "--verbose"])
    assert detailed.exit_code == 0
    assert "defaults" in detailed.stdout
    assert _unwrapped(str(models_dir)) in _unwrapped(detailed.stdout)
    assert "1 model(s)" in result.stdout
    assert "\x1b" not in result.stdout


def test_model_list_verbose_wraps_long_storage_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / ("long-directory-" * 6)
    models_dir.mkdir()
    (models_dir / "model.gguf").write_bytes(b"GGUF")
    monkeypatch.setenv("DROVE_MODELS_DIR", str(models_dir))
    monkeypatch.setenv("COLUMNS", "40")
    detailed = CliRunner().invoke(app, ["models", "list", "--verbose"])
    assert detailed.exit_code == 0
    assert str(models_dir) not in detailed.stdout  # wrapped across lines at 40 columns
    assert _unwrapped(str(models_dir)) in _unwrapped(detailed.stdout)
    assert str(models_dir) not in CliRunner().invoke(app, ["models", "list"]).stdout


def test_config_single_value_remains_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9001")
    result = CliRunner().invoke(app, ["config", "listen_port"])
    assert result.exit_code == 0
    assert result.stdout == "9001\n"


@pytest.mark.parametrize("legacy_process", [True, False])
def test_status_reports_all_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], legacy_process: bool
) -> None:
    proc = {"memory_rss_bytes": 2**30, "cpu_percent": 12.5}
    data = {
        "server": {"uptime_seconds": 65, "listen": "0.0.0.0:8080"},
        "models": [
            {
                "name": "[red]model",
                "loaded_seconds": 60,
                "idle_seconds": 2,
                "idle_timeout_seconds": 1800,
                "active_requests": 1,
            }
        ],
        "process": proc if legacy_process else {"[blue]model": proc, "unavailable": None},
        "requests": {"total": 10, "active": 1, "errors": 2},
        "tokens": {
            "prompt": 100,
            "completion": 50,
            "total": 150,
            "speed": {"last_tok_per_sec": 22, "avg_tok_per_sec": 20},
            "ttft": {"last_seconds": 0.123, "avg_seconds": 0.234},
        },
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    _print_status("http://localhost:8080", watch=True, verbose=True)
    output = capsys.readouterr().out
    for value in (
        "running",
        "1m 5s",
        "[red]model",
        "1.0 GB",
        "12.5%",
        "2 errors",
        "150 total",
        "22.0 tok/s last",
        "0.123 s last",
        "Ctrl+C",
    ):
        assert value in output
    assert "\x1b" not in output


def test_status_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = {
        "server": {"uptime_seconds": 0, "listen": "localhost:8080"},
        "requests": {"total": 0, "active": 0, "errors": 0},
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    _print_status("http://localhost:8080")
    assert "Models start on the first request" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("listen", "expected"),
    [
        ("0.0.0.0:8080", "listening on all interfaces (0.0.0.0:8080)"),
        (":::8080", "listening on all interfaces (:::8080)"),
        ("192.168.1.20:8080", "listening on 192.168.1.20:8080"),
        ("myhost.lan:8080", "listening on myhost.lan:8080"),
        ("127.0.0.1:8080", None),
        ("127.0.1.1:8080", None),
        ("[::1]:8080", None),
        ("::1:8080", None),
        ("localhost:8080", None),
    ],
)
def test_exposure_note_flags_non_loopback_binds(listen: str, expected: str | None) -> None:
    assert _exposure_note(listen) == expected


def test_status_loopback_bind_stays_concise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = {
        "server": {"uptime_seconds": 0, "listen": "127.0.0.1:8080"},
        "requests": {"total": 0, "active": 0, "errors": 0},
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    _print_status("http://127.0.0.1:8080")
    assert "listening on" not in capsys.readouterr().out


def test_status_offline_exit_and_watch_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline)
    result = CliRunner().invoke(app, ["server", "status"])
    assert result.exit_code == 1
    assert "Warning: Server is not running." in result.stderr
    assert "drove server" in result.stderr
    _print_status("http://localhost:8080", watch=True)  # Does not exit the refresh loop.


def _watch_status(monkeypatch: pytest.MonkeyPatch, *, terminal: bool) -> tuple[Mock, str]:
    """Run `server status --watch` for two snapshots; return the clear mock and output."""
    data = {
        "server": {"uptime_seconds": 65, "listen": "0.0.0.0:8080"},
        "models": [],
        "requests": {"total": 0, "active": 0, "errors": 0},
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
    }
    clear = Mock()
    stream = StringIO()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    monkeypatch.setattr("drove.cli.server.click.clear", clear)
    monkeypatch.setattr(
        "drove.cli.server.time.sleep", Mock(side_effect=[None, KeyboardInterrupt()])
    )
    if terminal:
        monkeypatch.setattr(
            "drove.cli.server.console",
            lambda: Console(file=stream, force_terminal=True, color_system=None),
        )
    result = CliRunner().invoke(app, ["server", "status", "--watch", "0.1"])
    assert result.exit_code == 0
    return clear, (stream.getvalue() if terminal else result.stdout)


def test_status_watch_does_not_clear_redirected_output(monkeypatch: pytest.MonkeyPatch) -> None:
    clear, output = _watch_status(monkeypatch, terminal=False)
    clear.assert_not_called()
    assert output.count("running · up 1m") == 2  # two snapshots before the interrupt
    assert "\x1b" not in output


def test_status_watch_clears_terminal_each_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    clear, output = _watch_status(monkeypatch, terminal=True)
    assert clear.call_count == 2
    assert output.count("running · up 1m") == 2


def test_download_report_preserves_names_and_resume_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = DownloadPlan(
        repo_id="org/repo",
        local_name="model",
        sharded=False,
        files={"[red]model.gguf": 100, "second.gguf": 200},
        extra_files={"tokens.txt": 50},
    )
    _print_download_plan(
        plan,
        tmp_path,
        {
            "[red]model.gguf": (FileStatus.COMPLETE, 100),
            "second.gguf": (FileStatus.INCOMPLETE, 100),
        },
        verbose=True,
    )
    output = capsys.readouterr().out
    for value in ("[red]model.gguf", "complete", "50% local", "tokens.txt", "pending"):
        assert value in output


def test_status_default_prioritizes_activity_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {
        "server": {"uptime_seconds": 65, "listen": "0.0.0.0:8080"},
        "models": [
            {"name": "busy", "active_requests": 2, "idle_seconds": 99, "idle_timeout_seconds": 60},
            {"name": "idle", "active_requests": 0, "idle_seconds": 10, "idle_timeout_seconds": 60},
            {
                "name": "expired",
                "active_requests": 0,
                "idle_seconds": 60,
                "idle_timeout_seconds": 60,
            },
        ],
        "requests": {"total": 10, "active": 2, "errors": 3},
        "tokens": {"prompt": 100, "completion": 50, "total": 150},
        "process": {"memory_rss_bytes": 2**30, "cpu_percent": 12.5},
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    runner = CliRunner()
    result = runner.invoke(app, ["server", "status"])
    assert result.exit_code == 0
    assert "3 errors since startup" in result.stdout
    assert "~50s" in result.stdout
    busy = next(line for line in result.stdout.splitlines() if "busy" in line)
    assert "2 active" in busy and "—" in busy and "due" not in busy
    expired = next(line for line in result.stdout.splitlines() if "expired" in line)
    assert expired.split()[-1] == "due"  # idle == timeout is already expired
    idle = next(line for line in result.stdout.splitlines() if line.split()[:1] == ["idle"])
    assert idle.split()[-1] == "~50s"
    assert "Tokens" not in result.stdout and "Processes" not in result.stdout
    assert "listening on all interfaces (0.0.0.0:8080)" in result.stdout
    detailed = runner.invoke(app, ["server", "status", "--verbose"])
    assert detailed.exit_code == 0
    assert "Tokens" in detailed.stdout and "Processes" in detailed.stdout


def test_config_hides_unset_but_preserves_false_and_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    defaults = {"ctx_size": None, "embedding": None, "threads": None}
    layers = [("global", {"ctx_size": 8192}), ("model", {"ctx_size": 0, "embedding": False})]
    _print_resolved_config(defaults, layers)
    output = capsys.readouterr().out
    assert "ctx_size" in output and "False" in output and "model" in output
    assert "8192" not in output and "threads" not in output
    assert "1 unset parameters hidden" in output
    _print_resolved_config(defaults, layers, show_all=True)
    output = capsys.readouterr().out
    assert "threads" in output and "(not set)" in output
    assert "hidden" not in output


def test_download_summary_is_bounded_and_calculates_remaining(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = DownloadPlan(
        repo_id="org/repo",
        local_name="model",
        sharded=False,
        files={f"file-{i}.gguf": 1048576 for i in range(100)},
    )
    statuses = {
        "file-0.gguf": (FileStatus.COMPLETE, 2 * 1048576),
        "file-1.gguf": (FileStatus.INCOMPLETE, 524288),
    }
    _print_download_plan(plan, tmp_path, statuses)
    output = capsys.readouterr().out
    assert "100.0 MB" in output and "98.5 MB" in output
    assert "1/100 files complete" in output
    assert "file-0.gguf" not in output
    assert len(output.splitlines()) < 15


def test_model_config_all_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    runner = CliRunner()
    compact = runner.invoke(app, ["models", "config", "--global"])
    full = runner.invoke(app, ["models", "config", "--global", "--all"])
    assert compact.exit_code == full.exit_code == 0
    assert "(not set)" not in compact.stdout
    assert "(not set)" in full.stdout
    assert len(compact.stdout.splitlines()) < len(full.stdout.splitlines())


def test_model_info_storage_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    runner = CliRunner()
    compact = runner.invoke(app, ["models", "info", "model"])
    full = runner.invoke(app, ["models", "info", "model", "-V"])
    assert compact.exit_code == full.exit_code == 0
    assert "Capabilities" in compact.stdout and "On disk" in compact.stdout
    assert "Primary" not in compact.stdout and "Primary" in full.stdout


@pytest.fixture
def default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the default config path into tmp so a regression can't touch the real one."""
    default = tmp_path / "default" / "config.toml"
    default.parent.mkdir()
    default.write_text('listen_port = 7777\ntui_theme = "gruvbox"\n')
    monkeypatch.setattr("drove.config.DEFAULT_CONFIG_PATH", default)
    # Also catch a CLI module that imported the constant by name.
    monkeypatch.setattr("drove.cli.main.DEFAULT_CONFIG_PATH", default, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DROVE_CONFIG", raising=False)
    return default


def test_config_set_writes_to_config_option(tmp_path: Path, default_config: Path) -> None:
    custom = tmp_path / "custom.toml"
    original = default_config.read_text()
    runner = CliRunner()
    assert runner.invoke(app, ["-c", str(custom), "init"]).exit_code == 0

    result = runner.invoke(app, ["-c", str(custom), "config", "startup_timeout_seconds", "1800"])

    assert result.exit_code == 0, result.output
    assert "startup_timeout_seconds = 1800" in custom.read_text()
    assert default_config.read_text() == original
    read_back = runner.invoke(app, ["-c", str(custom), "config", "startup_timeout_seconds"])
    assert read_back.stdout == "1800\n"


def test_config_set_writes_to_drove_config_env(
    tmp_path: Path, default_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "env.toml"
    custom.write_text("idle_timeout_seconds = 60\n")
    original = default_config.read_text()
    monkeypatch.setenv("DROVE_CONFIG", str(custom))

    result = CliRunner().invoke(app, ["config", "max_loaded_models", "3"])

    assert result.exit_code == 0, result.output
    written = custom.read_text()
    assert "max_loaded_models = 3" in written
    assert "idle_timeout_seconds = 60" in written
    assert default_config.read_text() == original


@pytest.mark.parametrize("value", ["3600", "0", "-1"])
def test_config_set_validates_speech_timeout(default_config: Path, value: str) -> None:
    original = default_config.read_text()
    result = CliRunner().invoke(app, ["config", "speech_timeout_seconds", "--", value])
    if value == "3600":
        assert result.exit_code == 0, result.output
        assert "speech_timeout_seconds = 3600" in default_config.read_text()
    else:
        assert result.exit_code == 1, result.output
        assert "greater than 0" in result.output
        assert default_config.read_text() == original


def test_config_set_without_custom_file_refuses(tmp_path: Path, default_config: Path) -> None:
    missing = tmp_path / "missing.toml"
    original = default_config.read_text()

    result = CliRunner().invoke(app, ["-c", str(missing), "config", "listen_port", "9000"])

    assert result.exit_code == 1
    assert not missing.exists()
    assert default_config.read_text() == original


@pytest.mark.parametrize("verbose", [False, True])
def test_status_displays_audio(monkeypatch: pytest.MonkeyPatch, capsys, verbose: bool) -> None:
    data = {
        "server": {"uptime_seconds": 65, "listen": "127.0.0.1:8080"},
        "requests": {"total": 3, "active": 0, "errors": 0},
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "audio": {
            "transcriptions": {"requests": 2, "audio_seconds": 12.5, "processing_seconds": 1.2},
            "speech": {"requests": 1, "audio_seconds": 3.5, "processing_seconds": 2.1},
        },
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: Mock(json=lambda: data))
    _print_status("http://localhost:8080", verbose=verbose)
    output = capsys.readouterr().out
    assert "Transcription" in output and "Speech" in output
    assert "12.5s audio" in output and "3.5s audio" in output
    assert "1.2s processing" in output and "2.1s processing" in output


def test_config_default_model_set_get_clear(tmp_path: Path) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text("")
    runner = CliRunner()
    prefix = ["-c", str(custom), "config", "default_model"]
    assert runner.invoke(app, [*prefix, "org/model:Q4_K_M"]).exit_code == 0
    assert runner.invoke(app, prefix).stdout == "org/model:Q4_K_M\n"
    assert 'default_model = "org/model:Q4_K_M"' in custom.read_text()
    assert runner.invoke(app, [*prefix, ""]).exit_code == 0
    assert "default_model" not in custom.read_text()
    assert runner.invoke(app, prefix).stdout == "None\n"


@pytest.mark.parametrize(
    "answer,accepted", [("\n", False), ("n\n", False), ("", False), ("wrong\ny\n", True)]
)
def test_confirmation_literal_safe_input(monkeypatch, answer, accepted):
    import typer

    from drove.cli.output import confirm_action

    stream = StringIO()
    out = Console(file=stream, width=40, color_system=None)
    monkeypatch.setattr("sys.stdin", StringIO(answer))
    if accepted:
        confirm_action(out, "Delete '[red]literal'?")
    else:
        with pytest.raises(typer.Abort):
            confirm_action(out, "Delete '[red]literal'?")
    output = stream.getvalue()
    assert "[red]literal" in output and "(n)" in output
    assert "\x1b" not in output


def test_download_remaining_all_categories_and_oversized():
    from drove.cli.models import _download_remaining

    plan = DownloadPlan(
        repo_id="org/repo",
        local_name="m",
        sharded=False,
        files={"one.gguf": 100, "two.gguf": 100},
        mmproj_files={"mmproj.gguf": 50},
        extra_files={"tokens": 25},
    )
    assert _download_remaining(
        plan,
        {
            "one.gguf": (FileStatus.COMPLETE, 200),
            "two.gguf": (FileStatus.INCOMPLETE, 120),
            "mmproj.gguf": (FileStatus.INCOMPLETE, 20),
        },
    ) == (55, 1)


def _toml(path: Path) -> dict[str, object]:
    import tomllib

    return tomllib.loads(path.read_text())


def test_config_set_keeps_env_overrides_and_defaults_out_of_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text('models_dir = "~/mine"\nlisten_host = "127.0.0.1"\nfuture_key = 1\n')
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    monkeypatch.setenv("DROVE_LISTEN_HOST", "10.0.0.1")
    monkeypatch.setenv("DROVE_LLAMA_SERVER__THREADS", "4")

    result = CliRunner().invoke(app, ["-c", str(custom), "config", "idle_timeout_seconds", "600"])

    assert result.exit_code == 0, result.output
    assert _toml(custom) == {
        "models_dir": "~/mine",
        "listen_host": "127.0.0.1",
        "future_key": 1,
        "idle_timeout_seconds": 600,
    }


def test_config_set_overridden_key_writes_file_value_while_env_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text("listen_port = 7000\n")
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    runner = CliRunner()

    assert runner.invoke(app, ["-c", str(custom), "config", "listen_port", "7100"]).exit_code == 0

    assert _toml(custom) == {"listen_port": 7100}
    assert runner.invoke(app, ["-c", str(custom), "config", "listen_port"]).stdout == "9000\n"


def test_config_set_nested_key_keeps_siblings_and_ignores_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text("listen_port = 7000\n[llama_server]\nthreads = 8\n")
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    monkeypatch.setenv("DROVE_LLAMA_SERVER__THREADS", "2")
    monkeypatch.setenv("DROVE_LLAMA_N_GPU_LAYERS", "5")

    result = CliRunner().invoke(
        app, ["-c", str(custom), "config", "llama_server.n_gpu_layers", "20"]
    )

    assert result.exit_code == 0, result.output
    assert _toml(custom) == {
        "listen_port": 7000,
        "llama_server": {"threads": 8, "n_gpu_layers": 20},
    }


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize(
    "env",
    [
        {"DROVE_LISTEN_PORT": "9000"},
        {"DROVE_LLAMA_SERVER__N_GPU_LAYERS": "12", "DROVE_LLAMA_N_GPU_LAYERS": "13"},
        {"DROVE_MAX_MEMORY": "24XB"},
    ],
)
def test_init_writes_true_defaults_not_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool, env: dict[str, str]
) -> None:
    custom = tmp_path / "config.toml"
    if force:
        custom.write_text("listen_port = 7000\nidle_timeout_seconds = 5\n")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    args = ["-c", str(custom), "init", *(["--force"] if force else [])]
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    written = _toml(custom)
    assert written["listen_port"] == 8080
    assert written["idle_timeout_seconds"] == 1800
    assert written["max_memory"] == "0"
    assert written["llama_server"] == {"n_gpu_layers": -1}
    assert "0.0.0.0:8080" in result.stdout


def test_init_without_force_leaves_existing_file(tmp_path: Path) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text("listen_port = 7000\n")
    result = CliRunner().invoke(app, ["-c", str(custom), "init"])
    assert result.exit_code == 1
    assert custom.read_text() == "listen_port = 7000\n"


def _no_traceback(result) -> None:  # type: ignore[no-untyped-def]
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_invalid_file_value_is_readable_and_repairable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text(f'models_dir = "{tmp_path}"\nmax_memory = "24XB"\nlisten_port = 7000\n')
    monkeypatch.delenv("DROVE_MAX_MEMORY", raising=False)
    runner = CliRunner()

    broken = runner.invoke(app, ["-c", str(custom), "models", "list"])
    assert broken.exit_code == 1
    _no_traceback(broken)
    assert str(custom) in broken.output.replace("\n", "")
    assert "max_memory" in broken.output

    repair = runner.invoke(app, ["-c", str(custom), "config", "max_memory", "24GB"])
    assert repair.exit_code == 0, repair.output
    assert _toml(custom) == {"models_dir": str(tmp_path), "max_memory": "24GB", "listen_port": 7000}
    assert runner.invoke(app, ["-c", str(custom), "models", "list"]).exit_code == 0


def test_invalid_replacement_value_leaves_file_unchanged(tmp_path: Path) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text('max_memory = "24XB"\n')
    result = CliRunner().invoke(app, ["-c", str(custom), "config", "max_memory", "lots"])
    assert result.exit_code == 1
    _no_traceback(result)
    assert custom.read_text() == 'max_memory = "24XB"\n'


def test_config_set_refuses_while_another_file_key_is_invalid(tmp_path: Path) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text('max_memory = "24XB"\n')
    result = CliRunner().invoke(app, ["-c", str(custom), "config", "listen_port", "7000"])
    assert result.exit_code == 1
    _no_traceback(result)
    assert "max_memory" in result.output
    assert custom.read_text() == 'max_memory = "24XB"\n'


def test_config_set_validates_file_value_masked_by_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text('max_memory = "24XB"\n')
    monkeypatch.setenv("DROVE_MAX_MEMORY", "8GB")
    result = CliRunner().invoke(app, ["-c", str(custom), "config", "listen_port", "7000"])
    assert result.exit_code == 1
    assert "max_memory" in result.output
    assert custom.read_text() == 'max_memory = "24XB"\n'


def test_malformed_toml_is_readable_and_not_rewritten(tmp_path: Path) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text("listen_port = = 1\n")
    runner = CliRunner()

    for args in (["models", "list"], ["config"], ["config", "listen_port", "7000"]):
        result = runner.invoke(app, ["-c", str(custom), *args])
        assert result.exit_code == 1, args
        _no_traceback(result)
        assert str(custom) in result.output.replace("\n", "")
        assert "line 1" in result.output
    assert custom.read_text() == "listen_port = = 1\n"


@pytest.mark.parametrize(
    "name,value,key",
    [
        ("DROVE_LISTEN_PORT", "abc", "listen_port"),
        ("DROVE_ALLOWED_TOOLS", "not-json", "allowed_tools"),
        ("DROVE_LLAMA_SERVER__THREADS", "many", "llama_server.threads"),
        ("DROVE_LLAMA_N_GPU_LAYERS", "all", "llama_server.n_gpu_layers"),
        ("DROVE_ALLOWED_TOOLS", '["fetch_url", 3]', "allowed_tools.1"),
        ("DROVE_LLAMA_SERVER", '{"threads": "many"}', "llama_server.threads"),
    ],
)
def test_invalid_env_value_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str, key: str
) -> None:
    custom = tmp_path / "config.toml"
    custom.write_text(f'models_dir = "{tmp_path}"\n')
    monkeypatch.setenv(name, value)
    runner = CliRunner()

    result = runner.invoke(app, ["-c", str(custom), "models", "list"])

    assert result.exit_code == 1
    _no_traceback(result)
    flat = result.output.replace("\n", " ")
    assert name in flat and key in flat
    # A key write still works (the file is fine) but does not claim to fix the env.
    assert runner.invoke(app, ["-c", str(custom), "config", "listen_port", "7000"]).exit_code == 0
    assert _toml(custom) == {"models_dir": str(tmp_path), "listen_port": 7000}
