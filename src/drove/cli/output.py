"""Shared, terminal-aware presentation for human-readable CLI reports.

Use Text for all dynamic values: model names and paths are not Rich markup.
Console detects redirected output and NO_COLOR without application settings.
"""

from collections.abc import Iterable
from difflib import get_close_matches
from typing import Literal

import typer
from rich import box
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text


def console(*, stderr: bool = False) -> Console:
    return Console(highlight=False, stderr=stderr)


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


def confirm_action(out: Console, action: str) -> None:
    """Ask for explicit consent; rejection and EOF retain CLI abort semantics."""
    try:
        accepted = Confirm.ask(Text(action, style="bold"), console=out, default=False)
    except EOFError:
        raise typer.Abort() from None
    if not accepted:
        raise typer.Abort()


def notice(
    level: Literal["Success", "Warning", "Error", "Info"],
    message: str,
    *,
    next_step: str | None = None,
    out: Console | None = None,
) -> None:
    """Print literal human feedback whose meaning does not depend on color."""
    target = out if out is not None else console(stderr=level in {"Warning", "Error"})
    style = {"Success": "green", "Warning": "yellow", "Error": "red", "Info": "cyan"}[level]
    target.print(Text.assemble((f"{level}: ", f"bold {style}"), message))
    if next_step:
        target.print(Text.assemble(("Next: ", "bold"), next_step))


def success(message: str, *, next_step: str | None = None) -> None:
    notice("Success", message, next_step=next_step)


def warning(message: str, *, next_step: str | None = None) -> None:
    notice("Warning", message, next_step=next_step)


def error(message: str, *, next_step: str | None = None) -> None:
    notice("Error", message, next_step=next_step)


def info(message: str, *, next_step: str | None = None) -> None:
    notice("Info", message, next_step=next_step)


def unknown(kind: str, value: str, candidates: Iterable[str]) -> str:
    """Describe an unknown value with one deterministic, plausible correction."""
    matches = get_close_matches(value, sorted(set(candidates)), n=1, cutoff=0.6)
    suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
    return f"Unknown {kind} '{value}'.{suggestion}"
