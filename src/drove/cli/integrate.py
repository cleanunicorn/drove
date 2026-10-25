"""`drove integrate` — register drove models with local coding agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from drove.cli.models import _complete_model_name, _find_model
from drove.cli.output import console, error, fields, heading, success, table, unknown, warning
from drove.config import Config
from drove.integrations import (
    AGENTS,
    DEFAULT_PROVIDER_ID,
    Agent,
    ConfigFiles,
    IncompleteRollbackError,
    ModelSpec,
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
        error(str(e))
        raise typer.Exit(1)

    path = (file.expanduser() if file else None) or target.config_path()
    files = ConfigFiles()
    try:
        result = _run(target, files, path, config, models, default, remove, base_url, provider_id)
    except (ValueError, OSError) as e:  # Unreadable file or invalid JSON object
        error(
            f"Could not read an agent config file: {e}",
            next_step="Check the config path and its JSON syntax.",
        )
        raise typer.Exit(1)

    changed = files.changed()
    if dry_run:
        _print_plain_result(result)
        for out_path, out_data in changed:
            typer.echo(f"\n# {out_path}")
            typer.echo(dump_json(out_data), nl=False)
        if not changed:
            typer.echo(f"{target.display_name} is already up to date: {path}")
        return

    for out_path, _ in changed:
        if has_comments(out_path):
            warning(f"Comments in {out_path} are not preserved.")
    try:
        backups = write_configs(changed)
    except IncompleteRollbackError as e:
        error(
            f"Could not write agent config, and the rollback failed: {e}",
            next_step="Inspect the config and backup files before retrying.",
        )
        raise typer.Exit(1)
    except OSError as e:
        error(
            f"Could not write agent config; nothing was changed: {e}",
            next_step="Check permissions and retry.",
        )
        raise typer.Exit(1)
    out = console()
    heading(out, "Integration", target.display_name)
    if result.specs:
        listing = table("MODEL", "CONTEXT", "OUTPUT", "VISION", "TOOLS", "REASONING")
        for spec in result.specs:
            listing.add_row(
                Text(spec.id),
                f"{spec.context_window:,}",
                f"{spec.max_output:,}",
                "yes" if spec.vision else "no",
                "yes" if spec.tool_call else "no",
                "yes" if spec.reasoning else "no",
            )
        out.print(listing)
    fields(out, [("Written", p) for p, _ in changed] + [("Backup", p) for p in backups])
    if changed:
        action = result.action
        success(
            f"{action}. Updated {target.display_name}: {', '.join(str(p) for p, _ in changed)}",
            next_step=f"Restart {target.display_name} to load the updated configuration.",
        )
    else:
        success(
            f"{target.display_name} is already up to date: {path}",
            next_step=f"Open {target.display_name} to use its configuration.",
        )


@dataclass(frozen=True)
class _IntegrationResult:
    specs: list[ModelSpec]
    note: str | None = None
    action: str = "Registered models"


def _print_plain_result(result: _IntegrationResult) -> None:
    """Keep the existing dry-run preamble byte-for-byte plain."""
    if result.note:
        typer.echo(result.note)
    for spec in result.specs:
        caps = [
            c
            for c, on in (
                ("vision", spec.vision),
                ("tools", spec.tool_call),
                ("reasoning", spec.reasoning),
            )
            if on
        ]
        typer.echo(
            f"  {spec.id}: context {spec.context_window:,}, output {spec.max_output:,}"
            + (f", {', '.join(caps)}" if caps else "")
        )


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
) -> _IntegrationResult:
    """Prepare edits and return display facts without claiming a successful write."""
    if remove:
        if default:
            error("--default cannot be combined with --remove.")
            raise typer.Exit(1)
        target.remove(files, path, models or None, provider_id=provider_id)
        return _IntegrationResult([], action="Removed drove integration")

    for name in models or []:
        _find_model(config.models_dir, name)
    try:
        specs = collect_model_specs(config, models or None)
    except (FileNotFoundError, ValueError) as e:
        error(str(e))
        raise typer.Exit(1)

    if not specs:
        # Full sync with nothing left to offer: drop what is still registered
        # rather than leaving the agent pointing at deleted models.
        stale = target.model_ids(files, path, provider_id=provider_id)
        if not stale or default:
            error("No chat models found.", next_step="Download one with drove models download.")
            raise typer.Exit(1)
        target.remove(files, path, stale, provider_id=provider_id)
        return _IntegrationResult(
            [],
            f"No chat models left; removing {len(stale)} stale model(s).",
            f"Removed {len(stale)} stale model(s)",
        )

    if default and default not in {s.id for s in specs}:
        error(
            unknown("default model", default, (s.id for s in specs)),
            next_step="Select a model being added; list models with drove models list.",
        )
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

    return _IntegrationResult(specs)
