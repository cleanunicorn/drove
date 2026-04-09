"""CLI subcommands for model management."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from vllama.downloader import DownloadPlan

from vllama.model_config import (
    DownloadInfo,
    ModelConfig,
    config_path_for_model,
    load_download_info,
    load_global_model_config,
    load_model_config,
    resolve_model_alias,
    save_download_info,
    save_global_model_config,
    save_model_config,
    set_global_model_config_key,
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
        if p.is_dir() and not p.name.startswith("."):
            names.append(p.name)
        elif p.suffix.lower() == ".gguf" and p.is_file():
            # Legacy: single file without directory
            names.append(p.stem)

    return [n for n in names if n.lower().startswith(incomplete.lower())]


_MODEL_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}


def _resolve_name(models_dir: Path, name: str) -> str:
    """Resolve a model name or HuggingFace reference to a local name."""
    if "/" in name:
        local = resolve_model_alias(models_dir, name)
        if local:
            return local
    return name


def _model_root(models_dir: Path, name: str) -> Path | None:
    """Return the model directory, or None if absent."""
    name = _resolve_name(models_dir, name)
    subdir = models_dir / name
    if subdir.is_dir():
        return subdir
    # Legacy: single file without directory
    candidate = models_dir / f"{name}.gguf"
    if candidate.exists():
        return candidate
    return None


def _find_model(models_dir: Path, name: str) -> Path:
    """Locate the primary model file by name or HuggingFace reference.

    Returns the primary file path (first GGUF inside the model directory).
    """
    name = _resolve_name(models_dir, name)
    subdir = models_dir / name
    if subdir.is_dir():
        shards = sorted(p for p in subdir.rglob("*.gguf"))
        if shards:
            return shards[0]
        others = sorted(p for p in subdir.rglob("*") if p.suffix.lower() in _MODEL_EXTS)
        if others:
            return others[0]

    # Legacy: single file without directory
    candidate = models_dir / f"{name}.gguf"
    if candidate.exists():
        return candidate

    typer.echo(f"Model '{name}' not found.", err=True)
    raise typer.Exit(1)


def _iter_models(models_dir: Path) -> list[tuple[str, Path, int]]:
    """Return (name, primary_path, total_bytes) for each model."""
    results = []

    if not models_dir.exists():
        return results

    for p in sorted(models_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            files = [f for f in p.rglob("*") if f.is_file()]
            total = sum(f.stat().st_size for f in files)
            primary = sorted(f for f in files if f.suffix.lower() == ".gguf")
            if not primary:
                primary = sorted(files)
            if primary:
                results.append((p.name, primary[0], total))
        elif p.suffix.lower() == ".gguf" and p.is_file():
            # Legacy: single file without directory
            results.append((p.stem, p, p.stat().st_size))

    return results


@models_app.command("list")
def list_models(
    ctx: typer.Context,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-V", help="Show download origin info.")
    ] = False,
) -> None:
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

        if verbose:
            dl = load_download_info(primary)
            if dl:
                org, repo = dl.repo_id.split("/", 1)
                typer.echo(f"  {'origin:':<10} {org}/{repo}")
                for fname in dl.files:
                    typer.echo(f"  {'file:':<10} {fname}")


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

    dl = load_download_info(primary)
    if dl:
        org, repo = dl.repo_id.split("/", 1)
        typer.echo(f"Origin:  {org}/{repo}")
        for fname in dl.files:
            typer.echo(f"         {fname}")

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

    target_str = f"{root}/"

    if not yes:
        typer.confirm(f"Delete model '{name}' at {target_str}?", abort=True)

    if root.is_dir():
        shutil.rmtree(root)
    else:
        # Legacy: single file without directory
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


def _print_download_plan(
    plan: DownloadPlan,
    models_dir: Path,
    statuses: dict[str, tuple[object, int]] | None = None,
) -> None:
    from vllama.downloader import FileStatus

    dest = plan.destination(models_dir)
    col = 60
    total_files = len(plan.files) + len(plan.mmproj_files)

    typer.echo("")
    typer.echo(f"  Repo        {plan.repo_id}")
    typer.echo(f"  Model name  {plan.local_name}")
    typer.echo(f"  Destination {dest}")
    typer.echo(f"  Files       {total_files}  ({_fmt_size(plan.total_bytes)} total)")
    typer.echo("")

    def _status_label(fname: str, remote_size: int) -> str:
        if statuses is None:
            return ""
        status, local_size = statuses.get(fname, (FileStatus.MISSING, 0))
        if status == FileStatus.COMPLETE:
            return "  [complete]"
        if status == FileStatus.INCOMPLETE:
            pct = local_size * 100 // remote_size if remote_size else 0
            return f"  [{pct}% local]"
        return ""

    for fname, size in plan.files.items():
        label = _status_label(fname, size)
        typer.echo(f"    {fname:<{col}}  {_fmt_size(size):>10}{label}")

    if plan.mmproj_files:
        typer.echo("")
        typer.echo("  Multimodal projection (vision):")
        for fname, size in plan.mmproj_files.items():
            label = _status_label(fname, size)
            typer.echo(f"    {fname:<{col}}  {_fmt_size(size):>10}{label}")

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
        str | None,
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

    from vllama.downloader import FileStatus

    statuses = plan.check_local_files(models_dir)
    has_existing = any(s != FileStatus.MISSING for s, _ in statuses.values())
    all_complete = has_existing and all(s == FileStatus.COMPLETE for s, _ in statuses.values())

    if all_complete:
        typer.echo(
            f"Model '{plan.local_name}' is already fully downloaded "
            f"at {plan.destination(models_dir)}."
        )
        raise typer.Exit(0)

    _print_download_plan(plan, models_dir, statuses)

    if not yes:
        if has_existing:
            typer.confirm("Some files already exist. Resume download?", abort=True)
        else:
            typer.confirm("Proceed with download?", abort=True)

    def progress(current: int, total: int, filename: str) -> None:
        typer.echo(f"  [{current}/{total}] {filename}")

    try:
        primary = plan.execute(models_dir, progress_cb=progress)
    except Exception as e:
        typer.echo(f"Download failed: {e}", err=True)
        raise typer.Exit(1)

    # Save download metadata to sidecar TOML
    from vllama.downloader import parse_model_ref

    _, quant = parse_model_ref(model_ref)
    all_files = sorted(plan.file_names) + sorted(plan.mmproj_files.keys())
    save_download_info(
        primary,
        DownloadInfo(
            repo_id=plan.repo_id,
            files=all_files,
            quant=quant,
        ),
    )

    # Auto-configure mmproj if a multimodal projection file was downloaded
    if plan.mmproj_files:
        mmproj_name = sorted(plan.mmproj_files.keys())[0]
        mmproj_path = primary.parent / Path(mmproj_name).name
        model_cfg = load_model_config(primary)
        updated_cfg = model_cfg.model_copy(update={"mmproj": str(mmproj_path)})
        save_model_config(primary, updated_cfg)
        typer.echo(f"  mmproj auto-configured: {mmproj_path.name}")

    size_mb = sum(f.stat().st_size for f in primary.parent.rglob("*") if f.is_file()) / 1_048_576
    typer.echo(f"\nSaved as '{plan.local_name}'  ({size_mb:.1f} MB)")


@models_app.command("config")
def model_config_cmd(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Model name, or config key for global config.",
            autocompletion=_complete_model_name,
        ),
    ] = None,
    key: Annotated[str | None, typer.Argument(help="Config key to get/set.")] = None,
    value: Annotated[str | None, typer.Argument(help="Value to set.")] = None,
    global_config: Annotated[
        bool,
        typer.Option(
            "--global",
            "-g",
            help="Force global config mode.",
        ),
    ] = False,
    unset: Annotated[
        str | None,
        typer.Option("--unset", help="Remove a config key."),
    ] = None,
) -> None:
    """Get or set model configuration parameters.

    If the first argument is a config key (not a model name), operates on
    the global config automatically. Use --global/-g to force global mode.
    Per-model settings override global ones.

    Examples:

        vllama models config mymodel                  # show model params

        vllama models config mymodel ctx_size 8192     # set a model param

        vllama models config ctx_size 16384            # set a global param

        vllama models config ctx_size                  # get a global param

        vllama models config --unset ctx_size          # remove a global param

        vllama models config --global                  # show all global params
    """
    models_dir = _models_dir(ctx)

    # Auto-detect global mode: if first arg is a known config key and not
    # an existing model, treat it as global config operation.
    if not global_config and name is not None:
        is_config_key = name in ModelConfig.model_fields
        is_model = _model_root(models_dir, name) is not None
        if is_config_key and not is_model:
            global_config = True

    # --unset without a name arg → global unset
    if not global_config and name is None and unset is not None:
        global_config = True

    if global_config:
        # Positional args shift: name→key, key→value
        effective_key = name
        effective_value = key
        if value is not None:
            typer.echo("Too many arguments for global config mode.", err=True)
            raise typer.Exit(1)

        if unset:
            cfg = load_global_model_config(models_dir)
            if unset not in ModelConfig.model_fields:
                typer.echo(f"Unknown key '{unset}'.", err=True)
                raise typer.Exit(1)
            updated = cfg.model_copy(update={unset: None})
            save_global_model_config(models_dir, updated)
            typer.echo(f"Unset '{unset}' from global config.")
            return

        if effective_key is None:
            base = _base_model_config(ctx)
            cfg = load_global_model_config(models_dir)
            all_fields = ModelConfig().model_dump()
            typer.echo("Global model config (effective):\n")
            _print_resolved_config(
                all_fields,
                [
                    ("config.toml", base.to_dict()),
                    ("_global.toml", cfg.to_dict()),
                ],
            )
            return

        if effective_value is None:
            cfg = load_global_model_config(models_dir)
            params = cfg.to_dict()
            if effective_key in params:
                typer.echo(str(params[effective_key]))
            else:
                typer.echo("(not set)")
            return

        try:
            set_global_model_config_key(
                models_dir,
                effective_key,
                effective_value,
            )
            typer.echo(
                f"Set '{effective_key}' = {effective_value} in global config.",
            )
        except ValueError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        return

    # Per-model config mode — name is required
    if name is None:
        typer.echo(
            "Missing model name. Use --global for global config.",
            err=True,
        )
        raise typer.Exit(1)

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
        base = _base_model_config(ctx)
        global_cfg = load_global_model_config(models_dir)
        model_cfg = load_model_config(model_path)
        all_fields = ModelConfig().model_dump()
        typer.echo(f"Model '{name}' config (effective):\n")
        _print_resolved_config(
            all_fields,
            [
                ("config.toml", base.to_dict()),
                ("_global.toml", global_cfg.to_dict()),
                (f"{name}.toml", model_cfg.to_dict()),
            ],
        )
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


def _base_model_config(ctx: typer.Context) -> ModelConfig:
    """Build the base ModelConfig from config.toml [llama_server] defaults."""
    config = ctx.obj["config"]
    return ModelConfig(
        n_gpu_layers=config.llama_server.n_gpu_layers,
        threads=config.llama_server.threads,
    )


def _print_resolved_config(
    all_fields: dict[str, object],
    layers: list[tuple[str, dict[str, object]]],
) -> None:
    """Print all ModelConfig fields with resolved values and their source.

    *layers* is an ordered list of ``(label, set_values)`` pairs from lowest
    to highest priority.  The last layer that sets a field wins.
    """
    for field, default in all_fields.items():
        source = None
        value = default
        for label, vals in layers:
            if field in vals:
                value = vals[field]
                source = label
        if source:
            typer.echo(f"  {field} = {value}  ({source})")
        elif value is not None:
            typer.echo(f"  {field} = {value}  (default)")
        else:
            typer.echo(f"  {field} = (not set)")
