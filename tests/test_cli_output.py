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
from drove.cli.models import _print_download_plan
from drove.cli.output import fields, heading, table
from drove.cli.server import _print_status
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


def test_model_list_empty_and_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROVE_MODELS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "drove models download" in result.stdout
    primary = tmp_path / "[red]model.gguf"
    primary.write_bytes(b"GGUF")
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "[red]model" in result.stdout
    assert "defaults" in result.stdout
    assert "1 model(s)" in result.stdout
    assert "\x1b" not in result.stdout


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
    _print_status("http://localhost:8080", watch=True)
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


def test_status_offline_exit_and_watch_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline)
    result = CliRunner().invoke(app, ["server", "status"])
    assert result.exit_code == 1
    assert "drove server" in result.stdout
    _print_status("http://localhost:8080", watch=True)  # Does not exit the refresh loop.


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
    )
    output = capsys.readouterr().out
    for value in ("[red]model.gguf", "complete", "50% local", "tokens.txt", "pending"):
        assert value in output
