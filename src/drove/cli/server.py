"""CLI subcommands for server management."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Annotated

import click
import typer
import uvicorn
from rich.text import Text
from typer.core import TyperCommand

from drove.cli.output import console, fields, heading, hint, section, table

server_app = typer.Typer(
    help="Manage the drove server.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@server_app.callback()
def _server_default(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="Listen host.")] = None,
    port: Annotated[int | None, typer.Option(help="Listen port.")] = None,
) -> None:
    """Start the drove proxy server (default when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        return

    from drove.proxy import create_app

    config = ctx.obj["config"]
    if host:
        config = config.model_copy(update={"listen_host": host})
    if port:
        config = config.model_copy(update={"listen_port": port})

    config.models_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which(str(config.llama_server_bin)):
        typer.secho(
            f"Warning: llama-server binary not found: {config.llama_server_bin!r}. "
            "Model requests will fail until it is installed and on PATH "
            "(or set llama_server_bin in the config).",
            fg=typer.colors.YELLOW,
            err=True,
        )

    cfg_path: Path = ctx.obj["config_path"]

    out = console()
    heading(out, "Starting server")
    fields(
        out,
        [
            ("Listen", f"{config.listen_host}:{config.listen_port}"),
            ("Endpoint", f"{_base_url(ctx, host, port)}/v1"),
            ("Models", config.models_dir),
            ("Idle timeout", _fmt_duration(config.idle_timeout_seconds)),
            ("Config", f"{cfg_path} (watching for changes)" if cfg_path.exists() else "defaults"),
        ],
    )
    out.print()

    fastapi_app = create_app(config, config_path=cfg_path if cfg_path.exists() else None)
    uvicorn.run(
        fastapi_app,
        host=config.listen_host,
        port=config.listen_port,
        log_level="warning",
    )


def _base_url(ctx: typer.Context, host: str | None, port: int | None) -> str:
    """Build the base URL for the running drove server."""
    config = ctx.obj["config"]
    base = f"http://{host or config.listen_host}:{port or config.listen_port}"
    return base.replace("//0.0.0.0:", "//127.0.0.1:")


class _OptionalValueCommand(TyperCommand):
    """Allow ``--watch`` to be used as a bare flag or with a value.

    When ``--watch`` appears without a following numeric argument, the default
    value ``2.0`` is injected so Click sees ``--watch 2.0``.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if arg in ("--watch", "-w"):
                next_idx = i + 1
                if next_idx >= len(new_args) or new_args[next_idx].startswith("-"):
                    new_args.insert(next_idx, "2.0")
                break
        return super().parse_args(ctx, new_args)


@server_app.command(cls=_OptionalValueCommand)
def status(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="drove host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="drove port (overrides config).")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-V", help="Show process, timing, and token details.")
    ] = False,
    watch: Annotated[
        float | None,
        typer.Option(
            "--watch",
            "-w",
            help="Continuously refresh every N seconds (default 2).",
        ),
    ] = None,
) -> None:
    """Show the status of the running drove server.

    Use --watch to continuously refresh. Examples:

        drove server status --watch        # refresh every 2s
        drove server status --watch 5      # refresh every 5s
    """
    base = _base_url(ctx, host, port)

    if watch is not None:
        try:
            while True:
                if console().is_terminal:
                    click.clear()
                _print_status(base, watch=True, verbose=verbose)
                time.sleep(watch)
        except KeyboardInterrupt:
            pass
    else:
        _print_status(base, verbose=verbose)


def _print_status(base: str, *, watch: bool = False, verbose: bool = False) -> None:
    """Fetch and print the server status.

    When *watch* is True, connection errors are printed but do not exit,
    so the caller can retry on the next refresh cycle.
    """
    import httpx

    try:
        resp = httpx.get(f"{base}/status", timeout=5.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        out = console()
        heading(out, "Server status", base)
        out.print(Text("Server is not running.", style="yellow"))
        hint(out, "Start the server: drove server")
        if not watch:
            raise typer.Exit(1)
        return
    except httpx.HTTPError as e:
        typer.echo(f"Failed to connect: {e}", err=True)
        if not watch:
            raise typer.Exit(1)
        return

    data = resp.json()

    out = console()
    server = data["server"]
    req = data["requests"]
    tok = data["tokens"]
    heading(out, "Server", f"{base}/v1")
    summary = Text(f"running · up {_fmt_duration(server['uptime_seconds'])}", style="green")
    summary.append(f" · {req['active']:,} active requests", style="bold" if req["active"] else "")
    if req["errors"]:
        summary.append(f" · {req['errors']:,} errors since startup", style="bold yellow")
    out.print(summary)

    models = data.get("models", [])
    section(out, f"Loaded models · {len(models)}")
    if models:
        listing = table("MODEL", "ACTIVITY", "IDLE UNLOAD IN")
        if verbose:
            listing.add_column("LOADED", overflow="fold")
            listing.add_column("IDLE / LIMIT", overflow="fold")
        listing.columns[0].style = "cyan"
        for model in models:
            active = model.get("active_requests", 0)
            idle = model.get("idle_seconds", 0)
            timeout = model.get("idle_timeout_seconds")
            unload = (
                "—"
                if active or timeout is None
                else "due"
                if idle >= timeout
                else "~" + _fmt_duration(timeout - idle)
            )
            row = [
                Text(model["name"]),
                Text(f"{active} active" if active else "idle"),
                Text(unload),
            ]
            if verbose:
                loaded = model.get("loaded_seconds")
                row.extend(
                    [
                        Text(_fmt_duration(loaded) if loaded is not None else "—"),
                        Text(
                            f"{_fmt_duration(idle)} / "
                            + (_fmt_duration(timeout) if timeout is not None else "—")
                        ),
                    ]
                )
            listing.add_row(*row)
        out.print(listing)
    else:
        hint(out, "No models loaded. Models start on the first request.")

    if not verbose:
        if req["total"]:
            hint(out, f"{req['total']:,} requests since startup")
        hint(out, "Use --verbose for process, timing, and token details.")
        if watch:
            hint(out, "Watching · Ctrl+C to stop")
        return

    section(out, "Connection")
    fields(out, [("Listen", server["listen"])])
    proc = data.get("process")
    if proc:
        section(out, "Processes")
        processes = (
            {models[0]["name"] if len(models) == 1 else "backend": proc}
            if "memory_rss_bytes" in proc
            else proc
        )
        listing = table("PROCESS", "MEMORY (RSS)", "CPU")
        for name, stats in processes.items():
            if stats:
                listing.add_row(
                    Text(name), _fmt_bytes(stats["memory_rss_bytes"]), f"{stats['cpu_percent']}%"
                )
        out.print(listing)

    section(out, "Traffic since startup")
    traffic: list[tuple[str, object]] = [
        (
            "Requests",
            f"{req['total']:,} total · {req['active']:,} active · {req['errors']:,} errors",
        ),
        ("Tokens", f"{tok['prompt']:,} in · {tok['completion']:,} out · {tok['total']:,} total"),
    ]
    for label, metrics, last, avg, unit, precision in [
        ("Speed", tok.get("speed", {}), "last_tok_per_sec", "avg_tok_per_sec", "tok/s", 1),
        ("TTFT", tok.get("ttft", {}), "last_seconds", "avg_seconds", "s", 3),
    ]:
        if metrics.get(last) is not None:
            value = f"{metrics[last]:.{precision}f} {unit} last"
            if metrics.get(avg) is not None:
                value += f" · {metrics[avg]:.{precision}f} {unit} avg"
            traffic.append((label, value))
    fields(out, traffic)
    if watch:
        out.print()
        hint(out, "Watching · Ctrl+C to stop")


def _fmt_bytes(b: int) -> str:
    """Format bytes into a human-readable size string."""
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.1f} GB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.1f} MB"
    return f"{b / 1024:.0f} KB"


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m"
