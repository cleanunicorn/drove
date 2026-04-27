"""Shell completion generation and installation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

completions_app = typer.Typer(help="Manage shell completions.", no_args_is_help=True)

_SHELLS = ("bash", "zsh", "fish", "powershell")

# Where to write the completion script for each shell
_COMPLETION_DIRS: dict[str, Path] = {
    "zsh": Path.home() / ".zfunc",
    "fish": Path.home() / ".config" / "fish" / "completions",
}

_COMPLETION_FILES: dict[str, str] = {
    "bash": "drove",
    "zsh": "_drove",
    "fish": "drove.fish",
    "powershell": "drove.ps1",
}

# Lines that need to be present in shell config files to activate completions
_ACTIVATION: dict[str, list[str]] = {
    "zsh": [
        "fpath=(~/.zfunc $fpath)",
        "autoload -Uz compinit && compinit",
    ],
    "bash": [
        "source ~/.bash_completions/drove",
    ],
    "fish": [],  # fish auto-loads from ~/.config/fish/completions/
    "powershell": [
        ". ~/.config/powershell/drove.ps1",
    ],
}

_SHELL_RC: dict[str, Path] = {
    "zsh": Path.home() / ".zshrc",
    "bash": Path.home() / ".bashrc",
    "powershell": Path.home() / ".config" / "powershell" / "Microsoft.PowerShell_profile.ps1",
}


def _generate_script(shell: str) -> str:
    """Generate the completion script for the given shell via Click's API."""
    from click.shell_completion import get_completion_class
    from typer.main import get_command

    from drove.cli.main import app as drove_app  # avoid circular at import time

    prog_name = "drove"
    cli = get_command(drove_app)
    complete_var = f"_{prog_name.upper()}_COMPLETE"

    complete_cls = get_completion_class(shell)
    if complete_cls is None:
        raise ValueError(f"Unsupported shell '{shell}'. Choose from: {', '.join(_SHELLS)}")

    complete = complete_cls(cli, {}, prog_name, complete_var)
    return complete.source()


@completions_app.command("generate")
def generate(
    shell: Annotated[
        str,
        typer.Argument(help=f"Shell name: {', '.join(_SHELLS)}"),
    ] = "",
) -> None:
    """Print the completion script to stdout.

    Pipe it wherever you need:

        drove completions generate zsh > ~/.zfunc/_drove

        drove completions generate bash | sudo tee /etc/bash_completion.d/drove
    """
    if not shell:
        detected = _detect_shell()
        shell = detected or "zsh"
        if detected:
            typer.echo(f"# Detected shell: {shell}", err=True)

    shell = shell.lower()
    try:
        script = _generate_script(shell)
    except Exception as e:
        typer.echo(f"Error generating completion: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(script, nl=False)


@completions_app.command("install")
def install(
    shell: Annotated[
        str,
        typer.Argument(help=f"Shell name: {', '.join(_SHELLS)}. Omit to auto-detect."),
    ] = "",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without writing files."),
    ] = False,
) -> None:
    """Install completions for the given shell.

    Writes the completion script to the appropriate location and prints any
    additional steps needed to activate it (e.g. sourcing in ~/.zshrc).
    """
    if not shell:
        detected = _detect_shell()
        if not detected:
            typer.echo(
                f"Could not detect current shell. Please specify one of: {', '.join(_SHELLS)}",
                err=True,
            )
            raise typer.Exit(1)
        shell = detected
        typer.echo(f"Detected shell: {shell}")

    shell = shell.lower()
    if shell not in _SHELLS:
        typer.echo(f"Unknown shell '{shell}'. Choose from: {', '.join(_SHELLS)}", err=True)
        raise typer.Exit(1)

    try:
        script = _generate_script(shell)
    except Exception as e:
        typer.echo(f"Error generating completion: {e}", err=True)
        raise typer.Exit(1)

    dest = _completion_path(shell)
    typer.echo(f"Writing completion script → {dest}")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(script)

    # Report activation steps
    activation = _ACTIVATION.get(shell, [])
    rc_file = _SHELL_RC.get(shell)

    if activation and rc_file:
        missing = _missing_activation_lines(rc_file, activation)
        if missing:
            typer.echo("")
            typer.echo(f"Add the following to {rc_file} to activate completions:")
            typer.echo("")
            for line in missing:
                typer.echo(f"    {line}")
            typer.echo("")
            if not dry_run:
                _append_activation(rc_file, missing)
                typer.echo(f"(Added automatically to {rc_file})")
        else:
            typer.echo(f"Activation lines already present in {rc_file}.")

    if shell == "fish":
        typer.echo("Fish loads completions automatically — no further steps needed.")

    typer.echo("")
    typer.echo(f"Restart your shell or run:  source {rc_file or '~/.zshrc'}")


@completions_app.command("shells")
def list_shells() -> None:
    """List supported shells."""
    for s in _SHELLS:
        marker = " (detected)" if s == _detect_shell() else ""
        typer.echo(f"  {s}{marker}")


def _completion_path(shell: str) -> Path:
    filename = _COMPLETION_FILES[shell]
    if shell == "bash":
        return Path.home() / ".bash_completions" / filename
    if shell == "powershell":
        return Path.home() / ".config" / "powershell" / filename
    return _COMPLETION_DIRS.get(shell, Path.home() / f".{shell}_completions") / filename


def _detect_shell() -> str | None:
    shell_bin = os.environ.get("SHELL", "")
    name = Path(shell_bin).name.lower()
    if name in _SHELLS:
        return name
    if "zsh" in name:
        return "zsh"
    if "bash" in name:
        return "bash"
    if "fish" in name:
        return "fish"
    return None


def _missing_activation_lines(rc: Path, lines: list[str]) -> list[str]:
    if not rc.exists():
        return lines
    content = rc.read_text()
    return [line for line in lines if line not in content]


def _append_activation(rc: Path, lines: list[str]) -> None:
    rc.parent.mkdir(parents=True, exist_ok=True)
    with rc.open("a") as f:
        f.write("\n# drove shell completions\n")
        for line in lines:
            f.write(line + "\n")
