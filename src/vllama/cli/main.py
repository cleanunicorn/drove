"""vllama CLI — root app and serve command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from vllama.cli.completions import completions_app
from vllama.cli.models import _complete_model_name, models_app
from vllama.config import DEFAULT_CONFIG_PATH, Config, load_config

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


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="Listen host.")] = None,
    port: Annotated[int | None, typer.Option(help="Listen port.")] = None,
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
def status(
    ctx: typer.Context,
    host: Annotated[str | None, typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="vllama port (overrides config).")] = None,
) -> None:
    """Show the status of the running vllama server."""
    import httpx

    config = ctx.obj["config"]
    base = f"http://{host or config.listen_host}:{port or config.listen_port}"
    base = base.replace("//0.0.0.0:", "//127.0.0.1:")

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

    # Model
    model = data["model"]
    if model.get("loaded") and model.get("name"):
        typer.echo(f"Model:     {model['name']}")
        if model.get("loaded_seconds") is not None:
            typer.echo(f"  Loaded:  {_fmt_duration(model['loaded_seconds'])} ago")
        idle = model.get("idle_seconds", 0)
        timeout = model.get("idle_timeout_seconds", 0)
        typer.echo(f"  Idle:    {_fmt_duration(idle)} / {_fmt_duration(timeout)}")
    else:
        typer.echo("Model:     (none loaded)")

    # Process
    proc = data.get("process")
    if proc:
        rss = proc["memory_rss_bytes"]
        cpu = proc["cpu_percent"]
        typer.echo(f"Process:   {_fmt_bytes(rss)} RSS, {cpu}% CPU")

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


@app.command()
def config(
    ctx: typer.Context,
    key: Annotated[str | None, typer.Argument(help="Config key to get or set.")] = None,
    value: Annotated[str | None, typer.Argument(help="Value to set.")] = None,
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
            f"Config file not found at {cfg_path}. Run 'vllama init' first or use --config.",
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
    host: Annotated[str | None, typer.Option(help="vllama host (overrides config).")] = None,
    port: Annotated[int | None, typer.Option(help="vllama port (overrides config).")] = None,
    endpoint: Annotated[
        str | None,
        typer.Option("--endpoint", "-e", help="OpenAI-compatible base URL (e.g. https://api.openai.com/v1)."),
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

    By default connects to the local vllama server. Use --endpoint to connect
    to any OpenAI-compatible API (OpenAI, Anthropic, Groq, etc.).

    Use /help inside the chat to see available commands (/sessions, /theme, …).
    """
    from vllama.sessions import latest_session
    from vllama.tui import ChatApp

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

    names = [m["id"] for m in model_list if "id" in m]
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
        except (ValueError, TypeError):
            typer.echo("Please enter a valid number")


@app.command("init")
def init_config(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing config.")
    ] = False,
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
