"""CLI subcommands for server management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

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


@server_app.command()
def stop(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="vllama port (overrides config).")] = None,
) -> None:
    """Stop the currently loaded model."""
    import httpx

    base = _base_url(ctx, host, port)
    try:
        resp = httpx.post(f"{base}/server/stop", timeout=30.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        typer.echo("Server is not running.")
        raise typer.Exit(1)
    except httpx.HTTPError as e:
        typer.echo(f"Failed: {e}", err=True)
        raise typer.Exit(1)

    data = resp.json()
    if data.get("model"):
        typer.echo(f"Stopped model: {data['model']}")
    else:
        typer.echo("No model was loaded.")


@server_app.command()
def restart(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="vllama port (overrides config).")] = None,
) -> None:
    """Restart the currently loaded model."""
    import httpx

    base = _base_url(ctx, host, port)
    try:
        resp = httpx.post(f"{base}/server/restart", timeout=120.0)
    except httpx.ConnectError:
        typer.echo("Server is not running.")
        raise typer.Exit(1)
    except httpx.HTTPError as e:
        typer.echo(f"Failed: {e}", err=True)
        raise typer.Exit(1)

    data = resp.json()
    if resp.status_code == 400:
        typer.echo(data.get("detail", "No model is loaded to restart."))
        raise typer.Exit(1)

    typer.echo(f"Restarted model: {data.get('model')}")
