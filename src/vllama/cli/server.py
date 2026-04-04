"""CLI subcommands for server management."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import click
import typer
import uvicorn
from typer.core import TyperCommand

server_app = typer.Typer(
    help="Manage the vllama server.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@server_app.callback()
def _server_default(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="Listen host.")] = None,
    port: Annotated[int | None, typer.Option(help="Listen port.")] = None,
) -> None:
    """Start the vllama proxy server (default when no subcommand is given)."""
    if ctx.invoked_subcommand is not None:
        return

    from vllama.proxy import create_app

    config = ctx.obj["config"]
    if host:
        config = config.model_copy(update={"listen_host": host})
    if port:
        config = config.model_copy(update={"listen_port": port})

    config.models_dir.mkdir(parents=True, exist_ok=True)

    cfg_path: Path = ctx.obj["config_path"]

    typer.echo(f"Starting vllama on {config.listen_host}:{config.listen_port}")
    typer.echo(f"Models directory: {config.models_dir}")
    typer.echo(f"Idle timeout: {config.idle_timeout_seconds}s")
    if cfg_path.exists():
        typer.echo(f"Config file: {cfg_path} (watching for changes)")

    fastapi_app = create_app(config, config_path=cfg_path if cfg_path.exists() else None)
    uvicorn.run(
        fastapi_app,
        host=config.listen_host,
        port=config.listen_port,
        log_level="warning",
    )


def _base_url(ctx: typer.Context, host: str | None, port: int | None) -> str:
    """Build the base URL for the running vllama server."""
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
    host: Annotated[str | None, typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="vllama port (overrides config).")] = None,
    watch: Annotated[
        float | None,
        typer.Option(
            "--watch",
            "-w",
            help="Continuously refresh every N seconds (default 2).",
        ),
    ] = None,
) -> None:
    """Show the status of the running vllama server.

    Use --watch to continuously refresh. Examples:

        vllama server status --watch        # refresh every 2s
        vllama server status --watch 5      # refresh every 5s
    """
    base = _base_url(ctx, host, port)

    if watch is not None:
        try:
            while True:
                typer.clear()
                _print_status(base)
                time.sleep(watch)
        except KeyboardInterrupt:
            pass
    else:
        _print_status(base)



def _print_status(base: str) -> None:
    """Fetch and print the server status. Exits on connection failure."""
    import httpx

    try:
        resp = httpx.get(f"{base}/status", timeout=5.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        typer.echo("Server is not running.")
        raise typer.Exit(1)
    except httpx.HTTPError as e:
        typer.echo(f"Failed to connect: {e}", err=True)
        raise typer.Exit(1)

    data = resp.json()

    # Server
    server = data["server"]
    typer.echo(f"Server:    running ({_fmt_duration(server['uptime_seconds'])} uptime)")
    typer.echo(f"Listen:    {server['listen']}")
    typer.echo(f"Endpoint:  {base}/v1")

    # Models
    models = data.get("models", [])
    if models:
        typer.echo(f"Models:    {len(models)} loaded")
        for model in models:
            typer.echo(f"  - {model['name']}")
            if model.get("loaded_seconds") is not None:
                typer.echo(
                    f"    Loaded:  {_fmt_duration(model['loaded_seconds'])} ago"
                )
            idle = model.get("idle_seconds", 0)
            timeout = model.get("idle_timeout_seconds", 0)
            typer.echo(
                f"    Idle:    {_fmt_duration(idle)} / {_fmt_duration(timeout)}"
            )
            if model.get("active_requests", 0) > 0:
                typer.echo(f"    Active:  {model['active_requests']} request(s)")
    else:
        typer.echo("Models:    (none loaded)")

    # Process
    proc = data.get("process")
    if proc:
        if "memory_rss_bytes" in proc:
            rss = _fmt_bytes(proc["memory_rss_bytes"])
            typer.echo(f"Process:   {rss} RSS, {proc['cpu_percent']}% CPU")
        else:
            for name, pstats in proc.items():
                if pstats:
                    rss = _fmt_bytes(pstats["memory_rss_bytes"])
                    cpu = pstats["cpu_percent"]
                    typer.echo(f"Process ({name}): {rss} RSS, {cpu}% CPU")

    # Requests
    req = data["requests"]
    typer.echo(f"Requests:  {req['total']} total, {req['active']} active, {req['errors']} errors")

    # Tokens
    tok = data["tokens"]
    if tok["total"] > 0:
        typer.echo(
            f"Tokens:    {tok['prompt']} in / {tok['completion']} out ({tok['total']} total)"
        )
        speed = tok.get("speed", {})
        if speed.get("last_tok_per_sec") is not None:
            parts = [f"{speed['last_tok_per_sec']} tok/s (last)"]
            if speed.get("avg_tok_per_sec") is not None:
                parts.append(f"{speed['avg_tok_per_sec']} tok/s (avg)")
            typer.echo(f"Speed:     {', '.join(parts)}")
        ttft = tok.get("ttft", {})
        if ttft.get("last_seconds") is not None:
            parts = [f"{ttft['last_seconds']:.3f}s (last)"]
            if ttft.get("avg_seconds") is not None:
                parts.append(f"{ttft['avg_seconds']:.3f}s (avg)")
            typer.echo(f"TTFT:      {', '.join(parts)}")
    else:
        typer.echo("Tokens:    (none)")


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
