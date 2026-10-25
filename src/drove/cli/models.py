"""CLI subcommands for model management."""

from __future__ import annotations

from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from stat import S_ISDIR, S_ISLNK
from typing import TYPE_CHECKING, Annotated

import typer
from rich.text import Text

from drove.cli.output import confirm_action, console, fields, heading, hint, section, table

if TYPE_CHECKING:
    from rich.console import Console

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
        typer.echo(f"Model '{name}' not found.", err=True)
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
    directory.mkdir(parents=True)
    primary = directory / "model.tts"
    save_model_config(primary, cfg)
    primary.touch()
    typer.echo(f"Registered {name} ({engine}). POST /v1/audio/speech with model='{name}'.")


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


@dataclass(frozen=True)
class _DeletionPreview:
    root: Path
    external_config: Path | None
    files: tuple[tuple[Path, int, bool], ...]

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size, _ in self.files)


def _deletion_preview(models_dir: Path, root: Path) -> _DeletionPreview:
    """Inventory precisely the targets we remove, without following symlinks."""
    base = models_dir.resolve()
    if ".." in root.parts:
        raise ValueError("Deletion target cannot contain parent traversal.")
    root = Path(abspath(root))
    if root == base or not root.parent.resolve().is_relative_to(base):
        raise ValueError("Deletion target must be a model inside the models directory.")
    entries: list[tuple[Path, int, bool]] = []

    def collect(path: Path) -> None:
        info = path.lstat()
        if S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):
                collect(child)
        else:
            entries.append((path, info.st_size, S_ISLNK(info.st_mode)))

    collect(root)
    candidate = root.with_name(root.name + ".toml") if root.is_dir() else root.with_suffix(".toml")
    external: Path | None = None
    if candidate.exists() or candidate.is_symlink():
        if candidate.is_dir() and not candidate.is_symlink():
            raise ValueError(f"Expected a sidecar file, found a directory: {candidate}")
        collect(candidate)
        external = candidate
    return _DeletionPreview(root, external, tuple(entries))


def _loaded_warning(ctx: typer.Context, models_dir: Path, root: Path) -> str:
    """Best-effort snapshot; malformed or unreachable status is never 'unloaded'."""
    import httpx

    config = ctx.obj["config"]
    host = config.listen_host.strip("[]")
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in host:
        host = f"[{host}]"
    try:
        response = httpx.get(f"http://{host}:{config.listen_port}/status", timeout=2.0)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Invalid status")
        store = ModelStore(models_dir)
        matched: list[str] = []
        resolved_root = root.resolve()
        recursive = root.is_dir() and not root.is_symlink()
        for model in data["models"]:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                raise ValueError("Invalid model status")
            candidate = store.find_root(model["name"])
            if candidate is None:
                continue
            resolved_candidate = candidate.resolve()
            affected = resolved_candidate == resolved_root or (
                recursive and resolved_candidate.is_relative_to(resolved_root)
            )
            if affected:
                active = model.get("active_requests")
                requests = (
                    f" ({active} active requests)"
                    if type(active) is int and active >= 0
                    else " (active requests unknown)"
                )
                matched.append(model["name"] + requests)
        if matched:
            return (
                "Warning: currently loaded: " + ", ".join(matched) + ". Deletion does not stop it."
            )
        return "Not currently loaded (server status snapshot)."
    except httpx.HTTPError, ValueError, OSError:
        return "Warning: loaded status unknown; server status could not be checked."


@models_app.command("delete")
def delete_model(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Model name.", autocompletion=_complete_model_name)],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a model and its config."""
    import shutil

    models_dir = Path(abspath(_models_dir(ctx)))
    root = _model_root(models_dir, name)
    if root is None:
        typer.echo(f"Model '{name}' not found.", err=True)
        raise typer.Exit(1)
    try:
        preview = _deletion_preview(models_dir, root)
    except (OSError, ValueError) as error:
        typer.echo(f"Cannot prepare deletion: {error}", err=True)
        raise typer.Exit(1) from None

    root = preview.root
    out = console()
    heading(out, "Delete model", name)
    fields(
        out,
        [
            ("Target", root),
            ("Files", len(preview.files)),
            (
                "Estimated space",
                f"{_fmt_size(preview.total_bytes)} ({preview.total_bytes:,} bytes)",
            ),
        ],
    )
    listing = table("REMOVE", "BYTES")
    for path, size, is_link in preview.files:
        label = str(path.relative_to(models_dir)) + (" (symlink only)" if is_link else "")
        listing.add_row(Text(label), f"{size:,}")
    out.print(listing)
    configs = [
        str(path.relative_to(models_dir)) for path, _, _ in preview.files if path.suffix == ".toml"
    ]
    fields(out, [("Sidecar config", ", ".join(configs) if configs else "none present")])
    hint(out, "Space is estimated; open files and hard links may delay or reduce reclaimed space.")
    if not yes:
        out.print(Text(_loaded_warning(ctx, models_dir, root), style="yellow"))
        confirm_action(out, f"Delete model '{name}' and the listed files?")

    if root.is_symlink() or not root.is_dir():
        root.unlink()
    else:
        shutil.rmtree(root)
    typer.echo(f"Deleted {root}")
    if preview.external_config is not None:
        preview.external_config.unlink()
        typer.echo(f"Deleted config {preview.external_config}")


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
        typer.echo(f"Invalid choice '{raw}'.")

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
        typer.echo(f"Selected: {chosen or 'all variants'}")
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
    typer.echo(f"Selected: {chosen or 'all quantizations'}")
    if chosen is None:
        return None
    return _apply_quant_selection(plan, chosen, filter_by_quant(plan.files, chosen), name_override)


def _download_remaining(
    plan: DownloadPlan, statuses: dict[str, tuple[FileStatus, int]]
) -> tuple[int, int]:
    from drove.downloader import FileStatus

    remaining = complete = 0
    for fname, size in {**plan.files, **plan.mmproj_files, **plan.extra_files}.items():
        status, local_size = statuses.get(fname, (FileStatus.MISSING, 0))
        if status == FileStatus.COMPLETE:
            complete += 1
        else:
            remaining += max(0, size - local_size)
    return remaining, complete


def _print_download_plan(
    plan: DownloadPlan,
    models_dir: Path,
    statuses: dict[str, tuple[FileStatus, int]] | None = None,
    *,
    verbose: bool = False,
) -> Console:
    from drove.downloader import FileStatus

    out = console()
    heading(out, "Download", plan.repo_id)
    if verbose:
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
    else:
        hint(out, "Use --verbose to inspect every file and resume state.")
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
        remaining, complete = _download_remaining(plan, statuses)
        fields(
            out,
            [("To download", f"~{_fmt_size(remaining)} · {complete}/{total_files} files complete")],
        )
    return out


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

    typer.echo(f"Resolving {model_ref} ...")
    try:
        if tts_engine is None:
            plan = resolve_download(model_ref, name_override=name)
        else:
            plan = resolve_download(model_ref, name_override=name, tts_engine=tts_engine)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Failed to resolve repo: {e}", err=True)
        raise typer.Exit(1)

    _, quant = parse_model_ref(model_ref)
    if quant is None and not yes:
        quant = _select_quant_variant(plan, name)

    statuses = plan.check_local_files(models_dir)
    has_existing = any(s != FileStatus.MISSING for s, _ in statuses.values())
    all_complete = has_existing and all(s == FileStatus.COMPLETE for s, _ in statuses.values())

    if all_complete and not plan.is_tts:
        typer.echo(
            f"Model '{plan.local_name}' is already fully downloaded "
            f"at {plan.destination(models_dir)}."
        )
        raise typer.Exit(0)

    if plan.tts_engine == "kitten":
        typer.echo(
            "KittenTTS 2 weights: Stellon Labs Community License (not Apache 2.0). "
            "Commercial use requires registration; revenue/funding limits and attribution apply.\n"
            "https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md"
        )

    out = _print_download_plan(plan, models_dir, statuses, verbose=verbose)
    if not yes and not all_complete:
        remaining, _ = _download_remaining(plan, statuses)
        action = "Resume download of" if has_existing else "Download"
        confirm_action(out, f"{action} approximately {_fmt_size(remaining)}?")

    def progress(current: int, total: int, filename: str) -> None:
        typer.echo(f"  [{current}/{total}] {filename}")

    try:
        primary = plan.execute(models_dir, progress_cb=progress)
    except Exception as e:
        typer.echo(f"Download failed: {e}", err=True)
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
        typer.echo(f"  Text-to-speech auto-configured: {plan.tts_engine}")

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
                typer.echo(f"  {k} auto-configured: {v}")
        if not inferred:
            typer.echo(
                "  Could not infer the ASR model type. Set it with: "
                f"drove models config '{plan.local_name}' asr_model <type>"
            )

    # Auto-configure mmproj if a multimodal projection file was downloaded
    # Store just the filename — resolved to an absolute path at server startup.
    if plan.mmproj_files:
        mmproj_name = Path(sorted(plan.mmproj_files.keys())[0]).name
        model_cfg = load_model_config(primary)
        updated_cfg = model_cfg.model_copy(update={"mmproj": mmproj_name})
        save_model_config(primary, updated_cfg)
        typer.echo(f"  mmproj auto-configured: {mmproj_name}")

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
