"""`drove integrate` — register drove models with local coding agents."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drove.cli.models import _complete_model_name
from drove.config import Config
from drove.integrations import (
    AGENTS,
    DEFAULT_PROVIDER_ID,
    Agent,
    ConfigFiles,
    IncompleteRollbackError,
    collect_model_specs,
    default_base_url,
    dump_json,
    get_agent,
    has_comments,
    write_configs,
)


def _complete_agent(incomplete: str) -> list[str]:
    return [k for k in sorted(AGENTS) if k.startswith(incomplete.lower())]


def integrate(
    ctx: typer.Context,
    agent: Annotated[
        str,
        typer.Argument(
            help=f"Agent to configure: {', '.join(sorted(AGENTS))}.",
            autocompletion=_complete_agent,
        ),
    ],
    models: Annotated[
        list[str] | None,
        typer.Argument(
            help="Models to add (default: every chat model drove has).",
            autocompletion=_complete_model_name,
        ),
    ] = None,
    default: Annotated[
        str | None,
        typer.Option(
            "--default",
            "-d",
            help="Also make this model the agent's default.",
            autocompletion=_complete_model_name,
        ),
    ] = None,
    remove: Annotated[
        bool,
        typer.Option(
            "--remove", help="Remove the given models, or the whole drove provider if none."
        ),
    ] = False,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="drove URL the agent connects to (default: from config)."),
    ] = None,
    provider_id: Annotated[
        str, typer.Option("--provider-id", help="Provider name used in the agent's config.")
    ] = DEFAULT_PROVIDER_ID,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Agent config file to edit (default: auto-detected)."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Print the resulting config without writing.")
    ] = False,
) -> None:
    """Add drove models, fully configured, to a local coding agent.

    Writes a `drove` provider pointing at this server into the agent's config,
    with each model's context window, output limit, vision, tool-calling and
    reasoning support filled in. Other providers and settings are left alone.

    With no model names every chat model is added and models that no longer
    exist in drove are dropped from the provider. Re-run after downloading or
    reconfiguring models to keep the agent in sync.

    Examples:

        drove integrate opencode                     # all models

        drove integrate pi qwen3-coder -d qwen3-coder

        drove integrate kilo --dry-run

        drove integrate opencode --remove            # undo
    """
    config = ctx.obj["config"]
    try:
        target = get_agent(agent)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    path = (file.expanduser() if file else None) or target.config_path()
    files = ConfigFiles()
    try:
        _run(target, files, path, config, models, default, remove, base_url, provider_id)
    except ValueError as e:  # JSONDecodeError, or a file that is not a JSON object
        typer.echo(f"Could not read an agent config file: {e}", err=True)
        raise typer.Exit(1)

    changed = files.changed()
    if dry_run:
        for out_path, out_data in changed:
            typer.echo(f"\n# {out_path}")
            typer.echo(dump_json(out_data), nl=False)
        if not changed:
            typer.echo(f"{target.display_name} is already up to date: {path}")
        return

    for out_path, _ in changed:
        if has_comments(out_path):
            typer.echo(f"Note: comments in {out_path} are not preserved.", err=True)
    try:
        backups = write_configs(changed)
    except IncompleteRollbackError as e:
        typer.echo(f"Could not write agent config, and the rollback failed: {e}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"Could not write agent config; nothing was changed: {e}", err=True)
        raise typer.Exit(1)
    for backup in backups:
        typer.echo(f"Backup: {backup}")
    if changed:
        typer.echo(f"Updated {target.display_name}: {', '.join(str(p) for p, _ in changed)}")
    else:
        typer.echo(f"{target.display_name} is already up to date: {path}")


def _run(
    target: Agent,
    files: ConfigFiles,
    path: Path,
    config: Config,
    models: list[str] | None,
    default: str | None,
    remove: bool,
    base_url: str | None,
    provider_id: str,
) -> None:
    """Apply the requested edit to *files*; exits on invalid input."""
    if remove:
        if default:
            typer.echo("--default cannot be combined with --remove.", err=True)
            raise typer.Exit(1)
        target.remove(files, path, models or None, provider_id=provider_id)
        return

    try:
        specs = collect_model_specs(config, models or None)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if not specs:
        # Full sync with nothing left to offer: drop what is still registered
        # rather than leaving the agent pointing at deleted models.
        stale = target.model_ids(files, path, provider_id=provider_id)
        if not stale or default:
            typer.echo("No chat models found. Download one with 'drove models download'.", err=True)
            raise typer.Exit(1)
        typer.echo(f"No chat models left; removing {len(stale)} stale model(s).")
        target.remove(files, path, stale, provider_id=provider_id)
        return

    if default and default not in {s.id for s in specs}:
        typer.echo(f"--default '{default}' is not among the models being added.", err=True)
        raise typer.Exit(1)

    target.apply(
        files,
        path,
        specs,
        provider_id=provider_id,
        base_url=base_url or default_base_url(config),
        prune=not models,
    )
    if default:
        target.set_default(files, path, default, provider_id=provider_id)

    for s in specs:
        caps = [
            c
            for c, on in (("vision", s.vision), ("tools", s.tool_call), ("reasoning", s.reasoning))
            if on
        ]
        typer.echo(
            f"  {s.id}: context {s.context_window:,}, output {s.max_output:,}"
            + (f", {', '.join(caps)}" if caps else "")
        )
