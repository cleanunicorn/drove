"""Transfer display tests use deterministic clocks and real Rich rendering."""

from io import StringIO

from rich.console import Console

from drove.cli.download_progress import DownloadDisplay
from drove.downloader import DownloadEvent


def test_plain_progress_is_bounded_literal_and_excludes_existing_bytes():
    output = StringIO()
    now = [0.0]
    display = DownloadDisplay(200, out=Console(file=output, width=200), clock=lambda: now[0])
    with display:
        display.update(DownloadEvent("[bold]a.gguf", 1, 2, "skipped", 100, 200))
        display.update(DownloadEvent("b.gguf", 2, 2, "resumed", 150, 200))
        now[0] = 5
        for i in range(50):
            display.update(DownloadEvent("b.gguf", 2, 2, "progress", 151 + i, 200, 1))
        display.update(DownloadEvent("b.gguf", 2, 2, "complete", 200, 200))
    text = output.getvalue()
    assert "[bold]a.gguf" in text
    assert "(skipped)" in text and "(resumed)" in text and "(complete)" in text
    assert "200/200 bytes" in text
    assert len(text.splitlines()) == 4
    assert "\x1b" not in text and "\r" not in text
    assert display.transferred == 50
    assert display.progress.tasks[0].fields["eta"] == "0s"


def test_terminal_display_has_filename_speed_eta_and_single_live_region(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    now = [0.0]
    display = DownloadDisplay(
        4_194_304,
        out=Console(file=output, force_terminal=True, width=180, color_system=None),
        clock=lambda: now[0],
    )
    with display:
        now[0] = 2
        display.update(
            DownloadEvent("model.gguf", 1, 1, "progress", 2_097_152, 4_194_304, 2_097_152)
        )
        display.progress.refresh()
        task = display.progress.tasks[0]
        assert task.completed == 2_097_152
        assert task.fields["rate"] == "1.00 MiB/s"
        assert task.fields["eta"] == "2s"
    text = output.getvalue()
    assert "model.gguf" in text and "ETA" in text and "MiB/s" in text
    assert text.count("\x1b[?25l") == 1  # One live display hides the cursor once.
    assert text.count("\x1b[?25h") == 1


def test_unknown_rate_and_total_remain_unknown():
    output = StringIO()
    display = DownloadDisplay(0, out=Console(file=output), clock=lambda: 1)
    with display:
        display.update(DownloadEvent("empty", 1, 1, "new", 0, 0))
    assert "ETA —" in output.getvalue()
    assert display.progress.tasks[0].total is None


def test_no_color_redirected_output_is_plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    output = StringIO()
    display = DownloadDisplay(1, out=Console(file=output))
    with display:
        display.update(DownloadEvent("[red]file", 1, 1, "complete", 1, 1))
    assert "[red]file" in output.getvalue()
    assert "\x1b" not in output.getvalue() and "\r" not in output.getvalue()
