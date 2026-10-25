"""One terminal progress display, or bounded plain records for redirected output."""

from collections.abc import Callable
from time import monotonic
from types import TracebackType

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from drove.cli.output import console
from drove.downloader import DownloadEvent


class DownloadDisplay:
    def __init__(
        self, total: int, *, out: Console | None = None, clock: Callable[[], float] = monotonic
    ) -> None:
        self.out = out if out is not None else console()
        self.clock = clock
        self.started = clock()
        self.last_line = self.started
        self.transferred = 0.0
        self.filename: str | None = None
        self.stage = "download"
        self.file_state = "new"
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[filename]}", markup=False),
            BarColumn(),
            DownloadColumn(),
            TextColumn("{task.fields[rate]}", markup=False),
            TextColumn("ETA {task.fields[eta]}", markup=False),
            console=self.out,
            disable=not self.out.is_terminal,
        )
        self.task = self.progress.add_task(
            "Download", total=total or None, filename="Preparing", rate="— B/s", eta="—"
        )

    def __enter__(self) -> DownloadDisplay:
        if self.out.is_terminal:
            self.progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.out.is_terminal:
            self.progress.stop()

    def update(self, event: DownloadEvent) -> None:
        if event.filename != self.filename:
            self.file_state = "new"
        if event.state != "progress":
            self.file_state = event.state
        self.filename = event.filename
        if event.state == "configuring":
            self.stage = "configuration"
        now = self.clock()
        self.transferred += event.transferred
        elapsed = now - self.started
        rate = self.transferred / elapsed if elapsed > 0 else 0
        rate_text = f"{rate / 1_048_576:.2f} MiB/s" if rate else "— B/s"
        eta = (
            f"{max(0, event.total - event.completed) / rate:.0f}s" if rate and event.total else "—"
        )
        label = f"[{event.index}/{event.count}] {event.filename} ({self.file_state})"
        self.progress.update(
            self.task, completed=event.completed, filename=label, rate=rate_text, eta=eta
        )
        if not self.out.is_terminal and (event.state != "progress" or now - self.last_line >= 5):
            self.out.print(
                Text(f"{label} · {event.completed:g}/{event.total} bytes · {rate_text} · ETA {eta}")
            )
            self.last_line = now
