"""CLI subcommands for model management."""

from __future__ import annotations

from pathlib import Path
from shlex import quote
from typing import TYPE_CHECKING, Annotated

import typer
from rich.text import Text

from drove.cli.output import (
    console,
    error,
    fields,
    heading,
    hint,
    info,
    section,
    success,
    table,
    unknown,
    warning,
)

if TYPE_CHECKING:
    from drove.downloader import DownloadPlan, FileStatus

from drove.model_config import (
    DownloadInfo,
    ModelConfig,
    config_path_for_model,
    load_download_info,
    load_global_model_config,
    load_model_config,
    save_download_info,
    save_global_model_config,
    save_model_config,
    set_global_model_config_key,
    set_model_config_key,
)
from drove.model_store import ModelStore

models_app = typer.Typer(help="Manage models.", no_args_is_help=True)


def _models_dir(ctx: typer.Context) -> Path:
    models_dir: Path = ctx.obj["config"].models_dir
    return models_dir


def _complete_model_name(ctx: typer.Context, incomplete: str) -> list[str]:
    """Shell completion callback: return model names matching the incomplete string."""
    from drove.config import DEFAULT_MODELS_DIR, load_config

    try:
        models_dir = ctx.obj["config"].models_dir if ctx.obj else load_config().models_dir
    except Exception:
        models_dir = DEFAULT_MODELS_DIR

    return ModelStore(models_dir).complete(incomplete)


def _complete_model_config_key(ctx: typer.Context, incomplete: str) -> list[str]:
    """Complete model configuration keys."""
    from drove.model_config import ModelConfig

    keys = sorted(ModelConfig.model_fields.keys())
    return [k for k in keys if k.startswith(incomplete)]


def _model_root(models_dir: Path, name: str) -> Path | None:
    """Return the model directory (or legacy flat file), or None if absent."""
    return ModelStore(models_dir).find_root(name)


def _find_model(models_dir: Path, name: str) -> Path:
    """Locate the primary model file by name or HuggingFace reference."""
    try:
        return ModelStore(models_dir).resolve(name)
    except FileNotFoundError:
        error(
            unknown("model", name, (entry.name for entry in ModelStore(models_dir).list())),
            next_step="List models: drove models list",
        )
        raise typer.Exit(1)


def _detect_capabilities(primary: Path) -> list[str]:
    """Return a list of capability tags for a model (e.g. 'vision').

    Detection is filesystem-based: looks for sibling mmproj files or an
    ``mmproj`` entry in the sidecar config.  This is reliable because the
    download command auto-saves mmproj metadata and llama.cpp expects the
    file to live next to the primary model.
    """
    if primary.suffix.lower() == ".tts":
        return ["tts"]
    try:
        if load_model_config(primary).backend == "tts":
            return ["tts"]
    except Exception:
        pass
    caps: list[str] = []

    # Vision: any mmproj-*.gguf file in the same directory, or mmproj set
    # in the sidecar config.
    model_dir = primary.parent
    has_mmproj_file = any(
        f.is_file() and f.suffix.lower() == ".gguf" and "mmproj" in f.name.lower()
        for f in model_dir.iterdir()
    )
    has_mmproj_cfg = False
    if config_path_for_model(primary).exists():
        try:
            has_mmproj_cfg = load_model_config(primary).mmproj is not None
        except Exception:
            pass
    if has_mmproj_file or has_mmproj_cfg:
        caps.append("vision")

    # Speech-to-text: ONNX primary (built-in ASR worker) or explicit backend
    if primary.suffix.lower() == ".onnx":
        caps.append("stt")
    else:
        try:
            if load_model_config(primary).backend == "asr":
                caps.append("stt")
        except Exception:
            pass

    return caps


def _iter_models(models_dir: Path) -> list[tuple[str, Path, int]]:
    """Return (name, primary_path, total_bytes) for each model."""
    return [(e.name, e.primary, e.total_bytes) for e in ModelStore(models_dir).list()]


@models_app.command("add-tts")
def add_tts_model(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Local model name, e.g. kokoro.")],
    engine: Annotated[
        str, typer.Option(help="kokoro, qwen, chatterbox, fish, or kitten.")
    ] = "kokoro",
    voice: Annotated[
        str | None, typer.Option(help="Default voice or reference WAV; inherits global settings.")
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="auto, cpu, cuda, or mps; inherits global settings.")
    ] = None,
) -> None:
    """Register a Cathy text-to-speech engine (weights are cached on first use)."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in {".", ".."}:
        raise typer.BadParameter(
            "Use a simple local name containing letters, digits, dots or dashes."
        )
    try:
        cfg = ModelConfig.model_validate(
            {
                "backend": "tts",
                "tts_engine": engine,
                "tts_voice": voice,
                "tts_device": device,
            }
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    models_dir = _models_dir(ctx)
    directory = models_dir / name
    if directory.exists() or (models_dir / f"{name}.gguf").exists():
        raise typer.BadParameter(f"Model '{name}' already exists; use 'drove models config'.")
    primary = directory / "model.tts"
    try:
        directory.mkdir(parents=True)
        save_model_config(primary, cfg)
        primary.touch()
    except OSError as exc:
        error(
            f"Could not finish registering '{name}': {exc}",
            next_step="Check the model directory and permissions.",
        )
        raise typer.Exit(1) from exc
    success(
        f"Registered {name} ({engine}).", next_step=f"POST /v1/audio/speech with model='{name}'."
    )


@models_app.command("list")
def list_models(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-V", help="Show storage, configuration, and download origin details."
        ),
    ] = False,
) -> None:
    """List all downloaded models."""
    models_dir = _models_dir(ctx)
    models = _iter_models(models_dir)

    out = console()
    heading(out, "Models", str(models_dir) if verbose else None)
    if not models:
        out.print("No models found.")
        hint(out, "Download a model: drove models download <org/repo[:QUANT]>")
        return

    listing = table("NAME", "CAPABILITY", "SIZE")
    if verbose:
        listing.add_column("CONFIG", overflow="fold")
    listing.columns[0].style = "cyan"
    listing.columns[2].justify = "right"
    for name, primary, total_bytes in models:
        has_cfg = config_path_for_model(primary).exists()
        caps = ", ".join(_detect_capabilities(primary)) or "text"
        row = [Text(name), Text(caps), Text(_fmt_size(total_bytes))]
        if verbose:
            row.append(Text("custom" if has_cfg else "defaults"))
        listing.add_row(*row)
    out.print(listing)
    hint(out, f"{len(models)} model(s) · {_fmt_size(sum(m[2] for m in models))} on disk")
    if not verbose:
        hint(out, "Inspect a model: drove models info <name>")

    if verbose:
        for name, primary, _ in models:
            dl = load_download_info(primary)
            if dl:
                section(out, name)
                fields(out, [("Origin", dl.repo_id), *(("File", f) for f in dl.files)])


@models_app.command("info")
def model_info(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Model name.", autocompletion=_complete_model_name)],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Show local paths and the downloaded file inventory."),
    ] = False,
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

    out = console()
    heading(out, "Model", name)
    fields(
        out,
        [
            ("Capabilities", ", ".join(_detect_capabilities(primary)) or "text"),
            ("On disk", f"{_fmt_size(total_bytes)} · {sum(f.is_file() for f in all_files)} files"),
        ],
    )
    dl = load_download_info(primary)
    if dl:
        fields(out, [("Repository", dl.repo_id)])
        if dl.quant:
            fields(out, [("Quantization", dl.quant)])
        if verbose:
            section(out, "Downloaded files")
            fields(out, (("File", f) for f in dl.files))

    cfg = load_model_config(primary)
    cfg_path = config_path_for_model(primary)
    if verbose:
        section(out, "Storage")
        fields(
            out,
            [
                ("Primary", primary),
                ("Config", f"{cfg_path} ({'exists' if cfg_path.exists() else 'not set'})"),
            ],
        )
    section(out, "Model overrides")
    params = cfg.to_dict()
    if params:
        fields(out, params.items())
    else:
        hint(out, "No model-specific parameters set (using global defaults).")
    hint(out, "See effective settings and sources: drove models config <name>")


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
        error(
            unknown("model", name, (entry.name for entry in ModelStore(models_dir).list())),
            next_step="List models: drove models list",
        )
        raise typer.Exit(1)

    target_str = f"{root}/"

    if not yes:
        typer.confirm(f"Delete model '{name}' at {target_str}?", abort=True)

    cfg_path = models_dir / f"{name}.toml"
    had_sidecar = cfg_path.exists()
    try:
        if root.is_dir():
            shutil.rmtree(root)
        else:
            root.unlink()
        if had_sidecar:
            cfg_path.unlink()
    except OSError as exc:
        error(
            f"Could not finish deleting '{name}': {exc}",
            next_step="Check remaining files and permissions before retrying.",
        )
        raise typer.Exit(1) from exc
    detail = f" and config {cfg_path}" if had_sidecar else ""
    success(f"Deleted {target_str}{detail}.", next_step="List remaining models: drove models list")


def _fmt_size(b: int) -> str:
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    return f"{b / 1_048_576:.1f} MB"


def _prompt_quant_choice(quants: dict[str, int]) -> str | None:
    """Show a numbered menu of quants and return the chosen tag, or None for 'all'."""
    tags = list(quants.keys())
    typer.echo("Multiple quantizations available:")
    typer.echo("")
    width = max(len(t) for t in tags)
    for i, tag in enumerate(tags, start=1):
        typer.echo(f"  {i:>2}) {tag:<{width}}  {_fmt_size(quants[tag]):>10}")
    total = sum(quants.values())
    typer.echo(f"   a) {'all':<{width}}  {_fmt_size(total):>10}")
    typer.echo("")

    valid = {str(i) for i in range(1, len(tags) + 1)} | {"a"}
    while True:
        raw = typer.prompt(f"Select quantization [1-{len(tags)}/a]").strip().lower()
        if raw in valid:
            break
        warning(
            f"Invalid choice '{raw}'.", next_step="Choose a listed number or a for all variants."
        )

    if raw == "a":
        return None
    return tags[int(raw) - 1]


def _apply_quant_selection(
    plan: DownloadPlan, chosen: str, filtered: dict[str, int], name_override: str | None
) -> str | None:
    """Narrow *plan* to the files of the chosen quant variant, in place.

    Returns *chosen* on success, or None when no files matched (the plan is
    left untouched).
    """
    from drove.downloader import infer_local_name, is_sharded

    if not filtered:
        return None
    plan.files = filtered
    plan.sharded = is_sharded(list(filtered.keys()))
    if name_override is None:
        plan.local_name = infer_local_name(plan.repo_id, list(filtered.keys()), chosen)
    return chosen


def _select_quant_variant(plan: DownloadPlan, name_override: str | None) -> str | None:
    """Offer the quant-variant menu for *plan* and apply the choice in place.

    ONNX (ASR) repos carry variants as filename infixes (model.int8.onnx) and
    resolve to the default variant, so choices come from the full pre-filter
    set kept on ``plan.onnx_files``; GGUF repos carry tags in filenames and
    resolve unfiltered. Returns the chosen quant tag, or None when the user
    keeps the default / all files (or only one variant exists).
    """
    from drove.downloader import (
        available_onnx_quants,
        available_quants,
        filter_by_quant,
        filter_onnx_quant,
    )

    if plan.is_tts:
        return None
    if plan.is_asr:
        variants = available_onnx_quants(plan.onnx_files)
        if len(variants) < 2:
            return None
        chosen = _prompt_quant_choice(variants)
        info(f"Selected: {chosen or 'all variants'}")
        if chosen is None:
            plan.files = dict(plan.onnx_files)
            return None
        if chosen == "default":
            return None
        return _apply_quant_selection(
            plan, chosen, filter_onnx_quant(plan.onnx_files, chosen), name_override
        )

    quants = available_quants(plan.files)
    if len(quants) < 2:
        return None
    chosen = _prompt_quant_choice(quants)
    info(f"Selected: {chosen or 'all quantizations'}")
    if chosen is None:
        return None
    return _apply_quant_selection(plan, chosen, filter_by_quant(plan.files, chosen), name_override)


def _print_download_plan(
    plan: DownloadPlan,
    models_dir: Path,
    statuses: dict[str, tuple[FileStatus, int]] | None = None,
    *,
    verbose: bool = False,
) -> None:
    from drove.downloader import FileStatus

    out = console()
    heading(out, "Download", plan.repo_id)
    total_files = len(plan.files) + len(plan.mmproj_files) + len(plan.extra_files)
    fields(
        out,
        [
            ("Model", plan.local_name),
            ("Destination", plan.destination(models_dir)),
            ("Files", f"{total_files} · {_fmt_size(plan.total_bytes)} total"),
        ],
    )
    if statuses is not None:
        all_files = {**plan.files, **plan.mmproj_files, **plan.extra_files}
        remaining = 0
        complete = 0
        for fname, size in all_files.items():
            status, local_size = statuses.get(fname, (FileStatus.MISSING, 0))
            if status == FileStatus.COMPLETE:
                complete += 1
            else:
                remaining += max(0, size - local_size)
        fields(
            out,
            [("To download", f"~{_fmt_size(remaining)} · {complete}/{total_files} files complete")],
        )
    if not verbose:
        hint(out, "Use --verbose to inspect every file and resume state.")
        out.print()
        return
    for title, files in [
        ("Model files", plan.files),
        ("Multimodal projection (vision)", plan.mmproj_files),
        ("Support files", plan.extra_files),
    ]:
        if not files:
            continue
        section(out, title)
        listing = table("FILE", "SIZE", "STATUS")
        listing.columns[1].justify = "right"
        for fname, size in files.items():
            label = "pending"
            if statuses is not None:
                status, local_size = statuses.get(fname, (FileStatus.MISSING, 0))
                if status == FileStatus.COMPLETE:
                    label = "complete"
                elif status == FileStatus.INCOMPLETE:
                    pct = local_size * 100 // size if size else 0
                    label = f"{pct}% local"
            listing.add_row(Text(fname), _fmt_size(size), label)
        out.print(listing)
    out.print()


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
                "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M  "
                "istupakov/parakeet-tdt-0.6b-v3-onnx:int8  "
                "hexgrad/Kokoro-82M"
            )
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Override local model name."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-V", help="Show every download file and resume state.")
    ] = False,
    tts_engine: Annotated[
        str | None,
        typer.Option(help="TTS engine for a compatible repo (auto-detected for known models)."),
    ] = None,
) -> None:
    """Download a model from HuggingFace Hub.

    Automatically discovers files in the repo. If a quantization tag is
    provided (:Q4_K_M for GGUF, :int8 for ONNX speech-to-text), only
    matching files are downloaded; otherwise a repo with several variants
    prompts for a choice. Sharded models (multiple files) are stored in a
    named subdirectory.

    Examples:

        drove models download unsloth/Qwen3-8B-GGUF

        drove models download unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M

        drove models download unsloth/Qwen3-8B-GGUF:Q8_0 --name qwen3-8b-q8

        drove models download istupakov/parakeet-tdt-0.6b-v3-onnx:int8  (speech-to-text)

        drove models download hexgrad/Kokoro-82M --name kokoro  (text-to-speech)
    """
    from drove.downloader import (
        FileStatus,
        parse_model_ref,
        resolve_download,
    )

    models_dir = _models_dir(ctx)
    models_dir.mkdir(parents=True, exist_ok=True)

    info(f"Resolving {model_ref} ...")
    try:
        if tts_engine is None:
            plan = resolve_download(model_ref, name_override=name)
        else:
            plan = resolve_download(model_ref, name_override=name, tts_engine=tts_engine)
    except (ValueError, OSError) as e:
        error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        error(
            f"Failed to resolve repo: {e}",
            next_step="Check the repository reference and network connection.",
        )
        raise typer.Exit(1)

    _, quant = parse_model_ref(model_ref)
    if quant is None and not yes:
        quant = _select_quant_variant(plan, name)

    statuses = plan.check_local_files(models_dir)
    has_existing = any(s != FileStatus.MISSING for s, _ in statuses.values())
    all_complete = has_existing and all(s == FileStatus.COMPLETE for s, _ in statuses.values())

    if all_complete and not plan.is_tts:
        info(
            f"Model '{plan.local_name}' is already fully downloaded "
            f"at {plan.destination(models_dir)}.",
            next_step="Review the model with drove models info " + quote(plan.local_name),
        )
        raise typer.Exit(0)

    _print_download_plan(plan, models_dir, statuses, verbose=verbose)
    if plan.tts_engine == "kitten":
        warning(
            "KittenTTS 2 weights: Stellon Labs Community License (not Apache 2.0). "
            "Commercial use requires registration; revenue/funding limits and attribution apply.\n"
            "https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md"
        )

    if not yes and not all_complete:
        if has_existing:
            typer.confirm("Some files already exist. Resume download?", abort=True)
        else:
            typer.confirm("Proceed with download?", abort=True)

    def progress(current: int, total: int, filename: str) -> None:
        typer.echo(f"  [{current}/{total}] {filename}")

    try:
        primary = plan.execute(models_dir, progress_cb=progress)
    except Exception as e:
        error(
            f"Download failed: {e}",
            next_step="Check the network connection and retry the download.",
        )
        raise typer.Exit(1)

    # Save download metadata to sidecar TOML
    all_files = (
        sorted(plan.file_names) + sorted(plan.mmproj_files.keys()) + sorted(plan.extra_files.keys())
    )
    save_download_info(
        primary,
        DownloadInfo(
            repo_id=plan.repo_id,
            files=all_files,
            quant=quant,
        ),
    )

    if plan.is_tts:
        info(f"Text-to-speech auto-configured: {plan.tts_engine}")

    # Auto-configure speech-to-text models for the built-in ASR worker
    if plan.is_asr:
        from drove.backend import infer_asr_model_type

        updates: dict[str, str] = {}
        inferred = infer_asr_model_type(plan.repo_id)
        if inferred:
            updates["asr_model"] = inferred
        if quant:
            updates["asr_quantization"] = quant.lower()
        if updates:
            model_cfg = load_model_config(primary)
            save_model_config(primary, model_cfg.model_copy(update=updates))
            for k, v in updates.items():
                info(f"{k} auto-configured: {v}")
        if not inferred:
            warning(
                "Could not infer the ASR model type.",
                next_step=(
                    f"Set it with drove models config {quote(plan.local_name)} asr_model <type>."
                ),
            )

    # Auto-configure mmproj if a multimodal projection file was downloaded
    # Store just the filename — resolved to an absolute path at server startup.
    if plan.mmproj_files:
        mmproj_name = Path(sorted(plan.mmproj_files.keys())[0]).name
        model_cfg = load_model_config(primary)
        updated_cfg = model_cfg.model_copy(update={"mmproj": mmproj_name})
        save_model_config(primary, updated_cfg)
        info(f"mmproj auto-configured: {mmproj_name}")

    size_mb = sum(f.stat().st_size for f in primary.parent.rglob("*") if f.is_file()) / 1_048_576
    success(
        f"Saved as '{plan.local_name}' ({size_mb:.1f} MB).",
        next_step=f"Review the model: drove models info {quote(plan.local_name)}",
    )


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
    key: Annotated[
        str | None,
        typer.Argument(help="Config key to get/set.", autocompletion=_complete_model_config_key),
    ] = None,
    value: Annotated[str | None, typer.Argument(help="Value to set.")] = None,
    show_all: Annotated[
        bool, typer.Option("--all", help="Include unset parameters in the configuration report.")
    ] = False,
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
        typer.Option(
            "--unset", help="Remove a config key.", autocompletion=_complete_model_config_key
        ),
    ] = None,
) -> None:
    """Get or set model configuration parameters.

    If the first argument is a config key (not a model name), operates on
    the global config automatically. Use --global/-g to force global mode.
    Per-model settings override global ones.

    Examples:

        drove models config mymodel                  # show model params

        drove models config mymodel ctx_size 8192     # set a model param

        drove models config ctx_size 16384            # set a global param

        drove models config ctx_size                  # get a global param

        drove models config --unset ctx_size          # remove a global param

        drove models config --global                  # show all global params
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
            error(
                "Too many arguments for global config mode.",
                next_step="See drove models config --help.",
            )
            raise typer.Exit(1)

        if unset:
            cfg = load_global_model_config(models_dir)
            if unset not in ModelConfig.model_fields:
                error(
                    unknown("key", unset, ModelConfig.model_fields),
                    next_step="List valid keys: drove models config --global --all",
                )
                raise typer.Exit(1)
            updated = cfg.model_copy(update={unset: None})
            try:
                save_global_model_config(models_dir, updated)
            except OSError as exc:
                error(f"Could not write global model configuration: {exc}")
                raise typer.Exit(1) from exc
            success(
                f"Unset '{unset}' from global config.",
                next_step="Review effective settings: drove models config --global",
            )
            return

        if effective_key is None:
            base = _base_model_config(ctx)
            cfg = load_global_model_config(models_dir)
            all_fields = ModelConfig().model_dump()
            heading(console(), "Global model configuration", "Effective values and their sources")
            _print_resolved_config(
                all_fields,
                [
                    ("config.toml", base.to_dict()),
                    ("_global.toml", cfg.to_dict()),
                ],
                show_all=show_all,
            )
            return

        if effective_key not in ModelConfig.model_fields:
            error(
                unknown("key", effective_key, ModelConfig.model_fields),
                next_step="List valid keys: drove models config --global --all",
            )
            raise typer.Exit(1)

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
            success(
                f"Set '{effective_key}' = {effective_value} in global config.",
                next_step="Review effective settings: drove models config --global",
            )
        except (ValueError, OSError) as e:
            error(str(e))
            raise typer.Exit(1)
        return

    # Per-model config mode — name is required
    if name is None:
        error(
            "Missing model name.",
            next_step="Use --global for global config or list models with drove models list.",
        )
        raise typer.Exit(1)

    model_path = _find_model(models_dir, name)

    if unset:
        cfg = load_model_config(model_path)
        if unset not in ModelConfig.model_fields:
            error(
                unknown("key", unset, ModelConfig.model_fields),
                next_step=f"List valid keys: drove models config {quote(name)} --all",
            )
            raise typer.Exit(1)
        updated = cfg.model_copy(update={unset: None})
        try:
            save_model_config(model_path, updated)
        except OSError as exc:
            error(f"Could not write model configuration: {exc}")
            raise typer.Exit(1) from exc
        success(
            f"Unset '{unset}' for model '{name}'.",
            next_step=f"Review effective settings: drove models config {quote(name)}",
        )
        return

    if key is None:
        base = _base_model_config(ctx)
        global_cfg = load_global_model_config(models_dir)
        model_cfg = load_model_config(model_path)
        all_fields = ModelConfig().model_dump()
        heading(console(), "Model configuration", name)
        _print_resolved_config(
            all_fields,
            [
                ("config.toml", base.to_dict()),
                ("_global.toml", global_cfg.to_dict()),
                (f"{name}.toml", model_cfg.to_dict()),
            ],
            show_all=show_all,
        )
        return

    if key not in ModelConfig.model_fields:
        error(
            unknown("key", key, ModelConfig.model_fields),
            next_step=f"List valid keys: drove models config {quote(name)} --all",
        )
        raise typer.Exit(1)

    if value is None:
        cfg = load_model_config(model_path)
        params = cfg.to_dict()
        typer.echo(str(params[key]) if key in params else "(not set)")
        return

    try:
        set_model_config_key(model_path, key, value)
        success(
            f"Set '{key}' = {value} for model '{name}'.",
            next_step=f"Review effective settings: drove models config {quote(name)}",
        )
    except (ValueError, OSError) as e:
        error(str(e))
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
    *,
    show_all: bool = False,
) -> None:
    """Print all ModelConfig fields with resolved values and their source.

    *layers* is an ordered list of ``(label, set_values)`` pairs from lowest
    to highest priority.  The last layer that sets a field wins.
    """
    out = console()
    listing = table("PARAMETER", "VALUE", "SOURCE")
    hidden = 0
    for field, default in all_fields.items():
        source = None
        value = default
        for label, vals in layers:
            if field in vals:
                value = vals[field]
                source = label
        if value is None and not show_all:
            hidden += 1
            continue
        listing.add_row(
            Text(field),
            Text(str(value) if value is not None else "(not set)"),
            Text(source or ("default" if value is not None else "—")),
        )
    out.print(listing)
    if hidden:
        hint(out, f"{hidden} unset parameters hidden. Use --all to show them.")
