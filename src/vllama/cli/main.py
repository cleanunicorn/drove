"""vllama CLI — root app and serve command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
import uvicorn

from vllama.cli.completions import completions_app
from vllama.cli.models import _complete_model_name, models_app
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
    ctx.obj["config_path"] = config_file or DEFAULT_CONFIG_PATH


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


@app.command()
def config(
    ctx: typer.Context,
    key: Annotated[Optional[str], typer.Argument(help="Config key to get or set.")] = None,
    value: Annotated[Optional[str], typer.Argument(help="Value to set.")] = None,
) -> None:
    """Show or edit configuration values.

    Examples:

        vllama config                              # show all values

        vllama config idle_timeout_seconds         # get one value

        vllama config idle_timeout_seconds 3600    # set a value
    """
    cfg_path = DEFAULT_CONFIG_PATH
    conf = ctx.obj["config"]

    if key is None:
        typer.echo(f"Config file: {cfg_path} ({'exists' if cfg_path.exists() else 'not found, using defaults'})")
        typer.echo("")
        flat = conf.model_dump()
        for field, val in flat.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    typer.echo(f"  {field}.{subkey} = {subval}")
            else:
                typer.echo(f"  {field} = {val}")
        return

    if value is None:
        # Get single key (supports dot notation for nested: llama_server.n_gpu_layers)
        val = _config_get(conf, key)
        if val is _MISSING:
            typer.echo(f"Unknown key '{key}'.", err=True)
            raise typer.Exit(1)
        typer.echo(str(val))
        return

    # Set key — load, update, save
    try:
        updated = _config_set(conf, key, value)
    except (ValueError, KeyError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if not cfg_path.exists():
        typer.echo(f"Config file not found at {cfg_path}. Run 'vllama init' first or use --config.", err=True)
        raise typer.Exit(1)

    updated.save(cfg_path)
    typer.echo(f"Set {key} = {value}")


_MISSING = object()


def _config_get(conf: "Config", key: str) -> object:  # type: ignore[name-defined]
    """Return the config value for key, or _MISSING if the key doesn't exist."""
    parts = key.split(".", 1)
    data = conf.model_dump()
    if len(parts) == 1:
        return data.get(parts[0], _MISSING)
    section = data.get(parts[0], _MISSING)
    if section is _MISSING or not isinstance(section, dict):
        return _MISSING
    return section.get(parts[1], _MISSING)


def _config_set(conf: "Config", key: str, raw_value: str) -> "Config":  # type: ignore[name-defined]
    from vllama.config import Config, LlamaServerDefaults

    parts = key.split(".", 1)

    if len(parts) == 2:
        section_name, subkey = parts
        if section_name != "llama_server":
            raise ValueError(f"Unknown section '{section_name}'. Only 'llama_server' is nested.")
        fields = LlamaServerDefaults.model_fields
        if subkey not in fields:
            valid = ", ".join(f"llama_server.{k}" for k in sorted(fields))
            raise ValueError(f"Unknown key '{key}'. Valid nested keys: {valid}")
        coerced = _coerce(fields[subkey].annotation, raw_value)
        new_llama = conf.llama_server.model_copy(update={subkey: coerced})
        return conf.model_copy(update={"llama_server": new_llama})

    fields = Config.model_fields
    if key not in fields:
        top = ", ".join(sorted(fields))
        nested = ", ".join(f"llama_server.{k}" for k in sorted(LlamaServerDefaults.model_fields))
        raise ValueError(f"Unknown key '{key}'.\nTop-level keys: {top}\nNested keys: {nested}")

    coerced = _coerce(fields[key].annotation, raw_value)
    return conf.model_copy(update={key: coerced})


def _coerce(annotation: object, raw: str) -> object:
    """Best-effort string → Python type coercion based on field annotation."""
    import types

    # Unwrap Optional[X] / X | None → X
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is types.UnionType or origin is type(None):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            annotation = non_none[0]

    if annotation is bool or annotation == bool:
        return raw.lower() in ("1", "true", "yes")
    if annotation is int or annotation == int:
        return int(raw)
    if annotation is float or annotation == float:
        return float(raw)
    return raw


@app.command()
def chat(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Model name to chat with.", autocompletion=_complete_model_name)],
    host: Annotated[Optional[str], typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[Optional[int], typer.Option(help="vllama port (overrides config).")] = None,
    system: Annotated[Optional[str], typer.Option("--system", "-s", help="System prompt.")] = None,
    resume: Annotated[bool, typer.Option("--resume", "-r", help="Resume the latest saved session.")] = False,
) -> None:
    """Open an interactive TUI chat session with the running vllama server.

    Use /help inside the chat to see available commands (/sessions, /theme, …).
    """
    from vllama.sessions import latest_session
    from vllama.tui import ChatApp

    config = ctx.obj["config"]
    base_url = f"http://{host or config.listen_host}:{port or config.listen_port}"
    base_url = base_url.replace("//0.0.0.0:", "//127.0.0.1:")

    resume_session = None
    if resume:
        resume_session = latest_session(config.sessions_dir, model)
        if resume_session is None:
            typer.echo(f"No saved sessions found for model '{model}'.")
            raise typer.Exit(1)

    tui = ChatApp(
        base_url=base_url,
        model=model,
        sessions_dir=config.sessions_dir,
        config_path=ctx.obj.get("config_path"),
        system_prompt=system,
        resume_session=resume_session,
        theme=config.tui_theme,
    )
    tui.run()


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
