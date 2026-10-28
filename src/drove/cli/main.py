"""drove CLI — root app and subcommand registration."""

from __future__ import annotations

import logging
import sys
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Annotated

import typer

from drove.cli.completions import completions_app
from drove.cli.integrate import integrate
from drove.cli.models import _complete_model_name, models_app
from drove.cli.output import console, error, fields, heading, info, success, unknown, warning
from drove.cli.server import server_app
from drove.config import (
    Config,
    ConfigError,
    load_config,
    resolve_config_path,
    set_config_value,
    write_default_config,
)

app = typer.Typer(
    name="drove",
    help="llama.cpp server manager and proxy.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
app.add_typer(completions_app, name="completions")
app.add_typer(server_app, name="server")
app.add_typer(server_app, name="serve", hidden=True)
app.command("integrate")(integrate)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"drove {_package_version('drove')}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config TOML file."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the drove version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    try:
        ctx.obj["config_path"] = resolve_config_path(config_file)
    except ConfigError as exc:
        error(str(exc), next_step=exc.next_step)
        raise typer.Exit(1) from exc
    # `config` and `init` load on their own terms, so they can repair or replace a
    # broken config file instead of failing on it.
    if ctx.invoked_subcommand not in ("config", "init"):
        ctx.obj["config"] = _load_or_exit(ctx.obj["config_path"])


def _load_or_exit(config_path: Path) -> Config:
    """Load the effective config, or explain what is wrong with it and exit 1."""
    try:
        return load_config(config_path)
    except ConfigError as exc:
        error(str(exc), next_step=exc.next_step)
        raise typer.Exit(1) from exc


def _complete_config_key(ctx: typer.Context, incomplete: str) -> list[str]:
    """Complete global config keys (including llama_server. nested keys)."""
    from drove.config import Config, LlamaServerDefaults

    keys: list[str] = []
    for field in Config.model_fields:
        if field == "llama_server":
            for subfield in LlamaServerDefaults.model_fields:
                keys.append(f"llama_server.{subfield}")
        else:
            keys.append(field)

    return [k for k in sorted(keys) if k.startswith(incomplete)]


@app.command()
def config(
    ctx: typer.Context,
    key: Annotated[
        str | None,
        typer.Argument(help="Config key to get or set.", autocompletion=_complete_config_key),
    ] = None,
    value: Annotated[str | None, typer.Argument(help="Value to set.")] = None,
) -> None:
    """Show or edit configuration values.

    Examples:

        drove config                              # show all values

        drove config idle_timeout_seconds         # get one value

        drove config idle_timeout_seconds 3600    # set a value
    """
    cfg_path: Path = ctx.obj["config_path"]

    if value is not None:
        _config_set_command(ctx, cfg_path, key, value)
        return

    conf = _load_or_exit(cfg_path)
    if key is None:
        out = console()
        heading(out, "Configuration", str(ctx.obj["config_path"]))
        rows: list[tuple[str, object]] = []
        for field, val in conf.model_dump().items():
            if isinstance(val, dict):
                rows.extend((f"{field}.{k}", v) for k, v in val.items())
            else:
                rows.append((field, val))
        fields(out, rows)
        return

    # Get single key (supports dot notation for nested: llama_server.n_gpu_layers)
    val = _config_get(conf, key)
    if val is _MISSING:
        error(
            unknown("key", key, _complete_config_key(ctx, "")),
            next_step="List valid keys: drove config",
        )
        raise typer.Exit(1)
    typer.echo(str(val))


def _config_set_command(ctx: typer.Context, cfg_path: Path, key: str | None, value: str) -> None:
    """Write one key to the file. Needs no loadable config, so it can repair one."""
    valid_keys = _complete_config_key(ctx, "")
    if key not in valid_keys:
        error(unknown("key", str(key), valid_keys), next_step="List valid keys: drove config")
        raise typer.Exit(1)

    if not cfg_path.exists():
        error(
            f"Config file not found at {cfg_path}.",
            next_step="Run drove init first or use --config.",
        )
        raise typer.Exit(1)

    try:
        _config_set(cfg_path, key, value)
    except ConfigError as exc:
        error(str(exc), next_step=exc.next_step)
        raise typer.Exit(1) from exc
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc
    except OSError as exc:
        error(
            f"Could not write configuration: {exc}",
            next_step="Check the config path and permissions.",
        )
        raise typer.Exit(1) from exc
    success(f"Set {key} = {value} in {cfg_path}.", next_step="Review configuration: drove config")


_MISSING = object()


def _config_get(conf: Config, key: str) -> object:
    """Return the config value for key, or _MISSING if the key doesn't exist."""
    parts = key.split(".", 1)
    data = conf.model_dump()
    if len(parts) == 1:
        return data.get(parts[0], _MISSING)
    section = data.get(parts[0], _MISSING)
    if section is _MISSING or not isinstance(section, dict):
        return _MISSING
    return section.get(parts[1], _MISSING)


def _config_set(cfg_path: Path, key: str, raw_value: str) -> None:
    """Coerce *raw_value* to the key's type and write only that key to *cfg_path*."""
    from drove.config import LlamaServerDefaults

    section, _, name = key.rpartition(".")
    fields = LlamaServerDefaults.model_fields if section else Config.model_fields
    set_config_value(key, _coerce(fields[name].annotation, raw_value), cfg_path)


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

    if annotation is bool:
        return raw.lower() in ("1", "true", "yes")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    return raw


@app.command()
def chat(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Argument(help="Model name to chat with.", autocompletion=_complete_model_name),
    ] = None,
    host: Annotated[str | None, typer.Option(help="drove host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="drove port (overrides config).")] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint", "-e", help="OpenAI-compatible base URL (e.g. https://api.openai.com/v1)."
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="API key for the endpoint."),
    ] = None,
    system: Annotated[str | None, typer.Option("--system", "-s", help="System prompt.")] = None,
    resume: Annotated[
        bool, typer.Option("--resume", "-r", help="Resume the latest saved session.")
    ] = False,
) -> None:
    """Open an interactive TUI chat session.

    By default connects to the local drove server. Use --endpoint to connect
    to any OpenAI-compatible API (OpenAI, Anthropic, Groq, etc.).

    Use /help inside the chat to see available commands (/sessions, /theme, …).
    """
    from drove.cli.tui_logging import textual_logging
    from drove.sessions import latest_session
    from drove.tui import ChatApp

    config = ctx.obj["config"]

    if endpoint:
        base_url = endpoint.rstrip("/")
        # Strip /v1 suffix if present — the TUI adds /v1/chat/completions
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
    else:
        base_url = f"http://{host or config.listen_host}:{port or config.listen_port}"
        base_url = base_url.replace("//0.0.0.0:", "//127.0.0.1:")

    if model is None:
        model = config.default_model
    if model is None:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            typer.echo("Pass a model name when not in a terminal: drove chat MODEL", err=True)
            raise typer.Exit(1)
        model = _select_model_from_endpoint(base_url, api_key, config.tui_theme)
        if model is None:
            raise typer.Exit()

    resume_session = None
    if resume:
        resume_session = latest_session(config.sessions_dir, model)
        if resume_session is None:
            warning(
                f"No saved sessions found for model '{model}'.",
                next_step="Start a new chat without --resume.",
            )
            raise typer.Exit(1)

    tui = ChatApp(
        base_url=base_url,
        model=model,
        sessions_dir=config.sessions_dir,
        config_path=ctx.obj.get("config_path"),
        system_prompt=system,
        resume_session=resume_session,
        theme=config.tui_theme,
        api_key=api_key,
    )
    with textual_logging():
        tui.run()


def _select_model_from_endpoint(
    base_url: str, api_key: str | None, theme: str = "textual-dark"
) -> str | None:
    """Fetch eligible models and show the shared keyboard picker."""
    from drove.chat_models import CatalogError, fetch_models
    from drove.cli.tui_logging import textual_logging
    from drove.model_picker import ModelPickerApp

    try:
        models = fetch_models(base_url, api_key)
    except CatalogError as exc:
        error(str(exc), next_step="Check the endpoint, server and available chat models.")
        raise typer.Exit(1) from exc
    try:
        with textual_logging():
            return ModelPickerApp(models, theme).run()
    except KeyboardInterrupt:
        return None


observe_app = typer.Typer(
    help="Browse logged API requests and responses.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(observe_app, name="observe")


@observe_app.callback()
def _observe_default(
    ctx: typer.Context,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Filter by model name."),
    ] = None,
) -> None:
    """Browse logged API requests and responses.

    Enable observation logging by setting `observe = true` in config.
    Logs are stored in the observe_dir (default ~/.local/share/drove/observe/).

    Run with no subcommand to open the TUI browser, or use `drove observe web`
    to start a web interface.

    Examples:

        drove observe                    # TUI, all models

        drove observe -m mymodel         # TUI, filter by model

        drove observe web                # web UI

        drove observe web -m mymodel     # web UI, filter by model
    """
    ctx.ensure_object(dict)
    ctx.obj["observe_model"] = model

    if ctx.invoked_subcommand is not None:
        return

    from drove.cli.tui_logging import textual_logging
    from drove.observe_tui import ObserveApp

    config = ctx.obj["config"]
    if not config.observe_dir.exists():
        warning(
            "No observe logs found.",
            next_step=(
                "Run drove config observe true, then make some requests."
                if not config.observe
                else "Make some requests through drove, then open observe again."
            ),
        )
        raise typer.Exit(1)

    tui = ObserveApp(
        observe_dir=config.observe_dir,
        model=model,
        theme=config.tui_theme,
    )
    with textual_logging():
        tui.run()


@observe_app.command()
def web(
    ctx: typer.Context,
    host: Annotated[str, typer.Option(help="Listen host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Listen port.")] = 8877,
) -> None:
    """Start a web UI for browsing observe logs.

    Examples:

        drove observe web
        drove observe web --port 9090
    """
    import uvicorn

    from drove.observe_web import create_observe_app

    config = ctx.obj["config"]
    model: str | None = ctx.obj.get("observe_model")

    if not config.observe_dir.exists():
        warning(
            "No observe logs found.",
            next_step=(
                "Run drove config observe true, then make some requests."
                if not config.observe
                else "Make some requests through drove, then open observe again."
            ),
        )
        raise typer.Exit(1)

    web_app = create_observe_app(config.observe_dir, model=model)
    info(f"Observe web UI: http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@app.command("init")
def init_config(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing config.")
    ] = False,
) -> None:
    """Create the config file at its default location with all default values.

    Writes to ~/.config/drove/config.toml (or the path from --config / DROVE_CONFIG).
    """
    cfg_path: Path = ctx.obj["config_path"]

    if cfg_path.exists() and not force:
        warning(
            f"Config already exists: {cfg_path}", next_step="Use drove init --force to overwrite."
        )
        raise typer.Exit(1)

    try:
        conf = write_default_config(cfg_path)
    except OSError as exc:
        error(
            f"Could not write configuration: {exc}",
            next_step="Check the config path and permissions.",
        )
        raise typer.Exit(1) from exc
    out = console()
    fields(
        out,
        [
            ("Config", cfg_path),
            ("Models", conf.models_dir),
            ("Listen", f"{conf.listen_host}:{conf.listen_port}"),
            ("Idle timeout", f"{conf.idle_timeout_seconds} seconds"),
        ],
    )
    success(
        f"Config written to {cfg_path}.", next_step="Review drove config, then start drove server."
    )
