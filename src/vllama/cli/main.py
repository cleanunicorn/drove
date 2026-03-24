"""vllama CLI — root app and serve command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from vllama.cli.completions import completions_app
from vllama.cli.models import models_app
from vllama.config import DEFAULT_CONFIG_PATH, load_config

app = typer.Typer(
    name="vllama",
    help="llama.cpp server manager and proxy.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
app.add_typer(completions_app, name="completions")


@app.callback()
def _root(
    ctx: typer.Context,
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to config TOML file."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_file)


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[Optional[str], typer.Option(help="Listen host.")] = None,
    port: Annotated[Optional[int], typer.Option(help="Listen port.")] = None,
) -> None:
    """Start the vllama proxy server."""
    from vllama.proxy import create_app

    config = ctx.obj["config"]
    if host:
        config = config.model_copy(update={"listen_host": host})
    if port:
        config = config.model_copy(update={"listen_port": port})

    config.models_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Starting vllama on {config.listen_host}:{config.listen_port}")
    typer.echo(f"Models directory: {config.models_dir}")
    typer.echo(f"Idle timeout: {config.idle_timeout_seconds}s")

    fastapi_app = create_app(config)
    uvicorn.run(
        fastapi_app,
        host=config.listen_host,
        port=config.listen_port,
        log_level="warning",
    )


@app.command()
def config(ctx: typer.Context) -> None:
    """Show current configuration."""
    cfg_path = DEFAULT_CONFIG_PATH
    conf = ctx.obj["config"]

    typer.echo(f"Config file: {cfg_path} ({'exists' if cfg_path.exists() else 'not found, using defaults'})")
    typer.echo("")
    for field, value in conf.model_dump().items():
        typer.echo(f"  {field} = {value}")


@app.command("init")
def init_config(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing config.")] = False,
) -> None:
    """Create the config file at its default location with all default values.

    Writes to ~/.config/vllama/config.toml (or the path from --config / VLLAMA_CONFIG).
    """
    cfg_path = DEFAULT_CONFIG_PATH
    conf = ctx.obj["config"]

    if cfg_path.exists() and not force:
        typer.echo(f"Config already exists: {cfg_path}")
        typer.echo("Use --force to overwrite.")
        raise typer.Exit(1)

    conf.save(cfg_path)
    typer.echo(f"Config written to {cfg_path}")
    typer.echo("")
    typer.echo(cfg_path.read_text())
