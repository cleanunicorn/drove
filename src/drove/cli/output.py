"""Shared, terminal-aware presentation for human-readable CLI reports.

Use Text for all dynamic values: model names and paths are not Rich markup.
Console detects redirected output and NO_COLOR without application settings.
"""

from collections.abc import Iterable

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


def console() -> Console:
    return Console(highlight=False)


def heading(out: Console, title: str, detail: str | None = None) -> None:
    out.print(Text.assemble(("drove", "bold cyan"), " / ", (title, "bold")))
    if detail:
        out.print(Text(detail, style="dim"))
    out.print()


def section(out: Console, title: str) -> None:
    out.print()
    out.print(Text(title, style="bold cyan"))


def table(*columns: str) -> Table:
    result = Table(
        box=box.SIMPLE,
        padding=(0, 1),
        collapse_padding=True,
        show_edge=False,
        header_style="bold",
        expand=False,
    )
    for column in columns:
        result.add_column(column, overflow="fold")
    return result


def fields(out: Console, rows: Iterable[tuple[str, object]]) -> None:
    result = Table.grid(padding=(0, 2))
    result.add_column(style="dim")
    result.add_column(overflow="fold")
    for label, value in rows:
        result.add_row(Text(label), Text(str(value)))
    out.print(result)


def hint(out: Console, message: str) -> None:
    out.print(Text(message, style="dim"))
