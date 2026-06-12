"""drove CLI — root app and subcommand registration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from drove.cli.completions import completions_app
from drove.cli.models import _complete_model_name, models_app
from drove.cli.server import server_app
from drove.config import DEFAULT_CONFIG_PATH, Config, load_config

app = typer.Typer(
    name="drove",
    help="llama.cpp server manager and proxy.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
app.add_typer(completions_app, name="completions")
app.add_typer(server_app, name="server")
app.add_typer(server_app, name="serve", hidden=True)


@app.callback()
def _root(
    ctx: typer.Context,
    config_file: Annotated[
        Path | None,
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
    cfg_path = DEFAULT_CONFIG_PATH
    conf = ctx.obj["config"]

    if key is None:
        typer.echo(
            f"Config file: {cfg_path} "
            f"({'exists' if cfg_path.exists() else 'not found, using defaults'})"
        )
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
        typer.echo(
            f"Config file not found at {cfg_path}. Run 'drove init' first or use --config.",
            err=True,
        )
        raise typer.Exit(1)

    updated.save(cfg_path)
    typer.echo(f"Set {key} = {value}")


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


def _config_set(conf: Config, key: str, raw_value: str) -> Config:
    from drove.config import Config, LlamaServerDefaults

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
        model = _select_model_from_endpoint(base_url, api_key)
        if model is None:
            raise typer.Exit(1)

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
        api_key=api_key,
    )
    tui.run()


def _select_model_from_endpoint(base_url: str, api_key: str | None) -> str | None:
    """Fetch models from an OpenAI-compatible /v1/models endpoint and prompt selection."""
    import httpx

    url = f"{base_url}/v1/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(url, headers=headers, timeout=5.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        typer.echo(f"Could not connect to {base_url}", err=True)
        typer.echo("Is the server running?", err=True)
        return None
    except httpx.HTTPStatusError as e:
        typer.echo(f"Failed to list models: {e.response.status_code}", err=True)
        return None
    except httpx.HTTPError as e:
        typer.echo(f"Failed to list models: {e}", err=True)
        return None

    try:
        data = resp.json()
        model_list = data.get("data", [])
    except Exception:
        typer.echo("Invalid response from /v1/models.", err=True)
        return None

    names: list[str] = [m["id"] for m in model_list if "id" in m]
    if not names:
        typer.echo("No models available at this endpoint.", err=True)
        return None

    if len(names) == 1:
        typer.echo(f"Using model: {names[0]}")
        return names[0]

    typer.echo("Available models:")
    typer.echo("")
    for i, name in enumerate(names, 1):
        typer.echo(f"  {i}. {name}")
    typer.echo("")

    while True:
        try:
            choice = int(typer.prompt("Select a model", default="1"))
            if 1 <= choice <= len(names):
                return names[choice - 1]
            typer.echo(f"Please enter a number between 1 and {len(names)}")
        except ValueError, TypeError:
            typer.echo("Please enter a valid number")


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

    from drove.observe_tui import ObserveApp

    config = ctx.obj["config"]
    if not config.observe_dir.exists():
        typer.echo("No observe logs found.")
        typer.echo(
            "Enable observation by setting 'observe = true' in your config, "
            "then make some requests."
        )
        raise typer.Exit(1)

    tui = ObserveApp(
        observe_dir=config.observe_dir,
        model=model,
        theme=config.tui_theme,
    )
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
        typer.echo("No observe logs found.")
        typer.echo(
            "Enable observation by setting 'observe = true' in your config, "
            "then make some requests."
        )
        raise typer.Exit(1)

    web_app = create_observe_app(config.observe_dir, model=model)
    typer.echo(f"Observe web UI: http://{host}:{port}")
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
