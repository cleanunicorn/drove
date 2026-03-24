"""CLI subcommands for model management."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from vllama.model_config import (
    ModelConfig,
    config_path_for_model,
    load_model_config,
    save_model_config,
    set_model_config_key,
)

models_app = typer.Typer(help="Manage models.", no_args_is_help=True)


def _models_dir(ctx: typer.Context) -> Path:
    return ctx.obj["config"].models_dir


def _complete_model_name(ctx: typer.Context, incomplete: str) -> list[str]:
    """Shell completion callback: return model names matching the incomplete string."""
    from vllama.config import DEFAULT_MODELS_DIR, load_config

    try:
        models_dir = ctx.obj["config"].models_dir if ctx.obj else load_config().models_dir
    except Exception:
        models_dir = DEFAULT_MODELS_DIR

    if not models_dir.exists():
        return []

    names: list[str] = []
    for p in sorted(models_dir.iterdir()):
        if p.suffix.lower() == ".gguf" and p.is_file():
            names.append(p.stem)
        elif p.is_dir() and not p.name.startswith("."):
            names.append(p.name)

    return [n for n in names if n.lower().startswith(incomplete.lower())]


_MODEL_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}


def _model_root(models_dir: Path, name: str) -> Path | None:
    """Return the root path for a model (file or directory), or None if absent."""
    candidate = models_dir / f"{name}.gguf"
    if candidate.exists():
        return candidate
    subdir = models_dir / name
    if subdir.is_dir():
        return subdir
    return None


def _find_model(models_dir: Path, name: str) -> Path:
    """Locate the primary model file by name.

    Returns the primary file path:
    - Single GGUF: models_dir/<name>.gguf
    - Sharded / multi-file: first shard inside models_dir/<name>/ (recursive)
    """
    candidate = models_dir / f"{name}.gguf"
    if candidate.exists():
        return candidate

    subdir = models_dir / name
    if subdir.is_dir():
        # Search recursively so nested cache dirs (e.g. .cache/huggingface/) are found
        shards = sorted(p for p in subdir.rglob("*.gguf"))
        if shards:
            return shards[0]
        others = sorted(p for p in subdir.rglob("*") if p.suffix.lower() in _MODEL_EXTS)
        if others:
            return others[0]

    typer.echo(f"Model '{name}' not found.", err=True)
    raise typer.Exit(1)


def _iter_models(models_dir: Path) -> list[tuple[str, Path, int]]:
    """Yield (name, primary_path, total_bytes) for each model."""
    results = []

    if not models_dir.exists():
        return results

    for p in sorted(models_dir.iterdir()):
        if p.suffix.lower() == ".gguf" and p.is_file():
            results.append((p.stem, p, p.stat().st_size))
        elif p.is_dir() and not p.name.startswith("."):
            files = [f for f in p.rglob("*") if f.is_file()]
            total = sum(f.stat().st_size for f in files)
            # Primary file = first shard or first file
            primary = sorted(f for f in files if f.suffix.lower() == ".gguf")
            if not primary:
                primary = sorted(files)
            if primary:
                results.append((p.name, primary[0], total))

    return results


@models_app.command("list")
def list_models(ctx: typer.Context) -> None:
    """List all downloaded models."""
    models_dir = _models_dir(ctx)
    models = _iter_models(models_dir)

    if not models:
        typer.echo("No models found.")
        return

    typer.echo(f"{'NAME':<45} {'SIZE':>10}  LOCATION")
    typer.echo("-" * 90)
    for name, primary, total_bytes in models:
        size_mb = total_bytes / 1_048_576
        has_cfg = config_path_for_model(primary).exists()
        cfg_marker = " [cfg]" if has_cfg else ""
        location = primary.parent if primary.parent != models_dir else primary
        typer.echo(f"{name:<45} {size_mb:>9.1f}M  {location}{cfg_marker}")


@models_app.command("info")
def model_info(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Model name.", autocompletion=_complete_model_name)],
) -> None:
    """Show info and configuration for a model."""
    models_dir = _models_dir(ctx)
    primary = _find_model(models_dir, name)

    # Collect all files belonging to this model
    if primary.parent == models_dir:
        all_files = [primary]
        total_bytes = primary.stat().st_size
    else:
        all_files = sorted(primary.parent.rglob("*"))
        total_bytes = sum(f.stat().st_size for f in all_files if f.is_file())

    typer.echo(f"Name:    {name}")
    typer.echo(f"Files:   {len(all_files)}")
    typer.echo(f"Size:    {total_bytes / 1_048_576:.1f} MB")
    typer.echo(f"Primary: {primary}")

    cfg = load_model_config(primary)
    cfg_path = config_path_for_model(primary)
    typer.echo(f"Config:  {cfg_path} ({'exists' if cfg_path.exists() else 'not set'})")

    params = cfg.to_dict()
    if params:
        typer.echo("\nModel parameters:")
        for k, v in params.items():
            typer.echo(f"  {k} = {v}")
    else:
        typer.echo("\nNo model-specific parameters set (using global defaults).")


@models_app.command("delete")
def delete_model(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Model name.", autocompletion=_complete_model_name)],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a model and its config."""
    import shutil

    models_dir = _models_dir(ctx)
    root = _model_root(models_dir, name)

    if root is None:
        typer.echo(f"Model '{name}' not found.", err=True)
        raise typer.Exit(1)

    target_str = f"{root}/" if root.is_dir() else str(root)

    if not yes:
        typer.confirm(f"Delete model '{name}' at {target_str}?", abort=True)

    if root.is_dir():
        shutil.rmtree(root)
    else:
        root.unlink()
    typer.echo(f"Deleted {target_str}")

    # Remove sidecar config if present (stored next to the root, not inside it)
    cfg_path = models_dir / f"{name}.toml"
    if cfg_path.exists():
        cfg_path.unlink()
        typer.echo(f"Deleted config {cfg_path}")


def _fmt_size(b: int) -> str:
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    return f"{b / 1_048_576:.1f} MB"


def _print_download_plan(plan: "DownloadPlan", models_dir: Path) -> None:  # type: ignore[name-defined]
    from vllama.downloader import DownloadPlan  # noqa: F401

    dest = plan.destination(models_dir)
    col = 60

    typer.echo("")
    typer.echo(f"  Repo        {plan.repo_id}")
    typer.echo(f"  Model name  {plan.local_name}")
    typer.echo(f"  Destination {dest}")
    typer.echo(f"  Files       {len(plan.files)}  ({_fmt_size(plan.total_bytes)} total)")
    typer.echo("")

    for fname, size in plan.files.items():
        typer.echo(f"    {fname:<{col}}  {_fmt_size(size):>10}")

    typer.echo("")


@models_app.command("download")
def download_model(
    ctx: typer.Context,
    model_ref: Annotated[
        str,
        typer.Argument(
            help=(
                "HuggingFace repo reference. "
                "Format: 'org/repo' or 'org/repo:QUANT'. "
                "Examples: unsloth/Qwen3-8B-GGUF  "
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"
            )
        ),
    ],
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Override local model name."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Download a model from HuggingFace Hub.

    Automatically discovers files in the repo. If a quantization tag is
    provided (e.g. :Q4_K_M), only matching files are downloaded. Sharded
    models (multiple files) are stored in a named subdirectory.

    Examples:

        vllama models download unsloth/Qwen3-8B-GGUF

        vllama models download unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M

        vllama models download unsloth/Qwen3-8B-GGUF:Q8_0 --name qwen3-8b-q8
    """
    from vllama.downloader import resolve_download

    models_dir = _models_dir(ctx)
    models_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Resolving {model_ref} ...")
    try:
        plan = resolve_download(model_ref, name_override=name)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Failed to resolve repo: {e}", err=True)
        raise typer.Exit(1)

    if plan.already_exists(models_dir):
        typer.echo(f"Model '{plan.local_name}' already exists at {plan.destination(models_dir)}.")
        raise typer.Exit(1)

    _print_download_plan(plan, models_dir)

    if not yes:
        typer.confirm("Proceed with download?", abort=True)

    def progress(current: int, total: int, filename: str) -> None:
        typer.echo(f"  [{current}/{total}] {filename}")

    try:
        primary = plan.execute(models_dir, progress_cb=progress)
    except Exception as e:
        typer.echo(f"Download failed: {e}", err=True)
        raise typer.Exit(1)

    size_mb = sum(f.stat().st_size for f in primary.parent.rglob("*") if f.is_file()) / 1_048_576
    typer.echo(f"\nSaved as '{plan.local_name}'  ({size_mb:.1f} MB)")


@models_app.command("config")
def model_config_cmd(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Model name.", autocompletion=_complete_model_name)],
    key: Annotated[Optional[str], typer.Argument(help="Config key to get/set.")] = None,
    value: Annotated[Optional[str], typer.Argument(help="Value to set.")] = None,
    unset: Annotated[
        Optional[str],
        typer.Option("--unset", help="Remove a config key."),
    ] = None,
) -> None:
    """Get or set per-model configuration parameters.

    Examples:

        vllama models config mymodel                  # show all params

        vllama models config mymodel ctx_size          # get one param

        vllama models config mymodel ctx_size 8192     # set a param

        vllama models config mymodel --unset ctx_size  # remove a param
    """
    models_dir = _models_dir(ctx)
    model_path = _find_model(models_dir, name)

    if unset:
        cfg = load_model_config(model_path)
        if unset not in ModelConfig.model_fields:
            typer.echo(f"Unknown key '{unset}'.", err=True)
            raise typer.Exit(1)
        updated = cfg.model_copy(update={unset: None})
        save_model_config(model_path, updated)
        typer.echo(f"Unset '{unset}' for model '{name}'.")
        return

    if key is None:
        cfg = load_model_config(model_path)
        params = cfg.to_dict()
        if not params:
            typer.echo(f"No config set for model '{name}'.")
        else:
            for k, v in params.items():
                typer.echo(f"{k} = {v}")
        return

    if value is None:
        cfg = load_model_config(model_path)
        params = cfg.to_dict()
        typer.echo(str(params[key]) if key in params else "(not set)")
        return

    try:
        set_model_config_key(model_path, key, value)
        typer.echo(f"Set '{key}' = {value} for model '{name}'.")
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
