"""Single authority for locating model files on disk."""

from __future__ import annotations

import builtins
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from drove.download_state import UNREADABLE, read_state
from drove.model_config import load_download_info, resolve_model_alias

# Suffixes that mark a directory as a model root rather than a namespace.  Raw
# weights count so an unservable download is not mistaken for a namespace; only
# _PRIMARY_EXTS can become a served primary.
_MODEL_EXTS = frozenset({".gguf", ".onnx", ".safetensors", ".bin", ".pt", ".tts"})
_PRIMARY_EXTS = (".tts", ".gguf", ".onnx")

# Split GGUF shard name: <stem>-00001-of-00003.gguf
_SPLIT_GGUF = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)

# Multimodal projector filenames (``mmproj-*.gguf``): companion files passed via
# ``--mmproj``, never a model's primary file.
MMPROJ_RE = re.compile(r"mmproj", re.IGNORECASE)

# One path segment of a model name: no separators, no leading dot or dash.
_NAME_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


def validate_model_name(name: object) -> str:
    """Return *name* unchanged if it is a safe model name, else raise ValueError.

    A model name is ``name`` or ``org/name``; each segment starts with a letter
    or digit and continues with letters, digits, ``.``, ``_``, ``:`` or ``-``
    (so ``org/repo:Q4_K_M`` is valid).  The check runs on the raw string with
    no normalization, so ``./name``, ``name/`` and ``a//b`` are rejected rather
    than silently aliased to another spelling.
    """
    if not isinstance(name, str):
        raise ValueError("Model name must be a string.")
    segments = name.split("/")
    if len(segments) > 2 or not all(_NAME_SEGMENT.fullmatch(s) for s in segments):
        raise ValueError(
            f"Invalid model name {name!r}: use a relative name like 'model' or "
            "'org/model', made of letters, digits, '.', '_', ':' or '-'."
        )
    return name


def _is_valid_name(name: str) -> bool:
    try:
        validate_model_name(name)
    except ValueError:
        return False
    return True


def is_within(base: Path, candidate: Path) -> bool:
    """True when *candidate* resolves (following symlinks) strictly inside *base*."""
    try:
        root = base.resolve()
        target = candidate.resolve()
    except OSError, RuntimeError:
        return False
    return target != root and target.is_relative_to(root)


class IncompleteModelError(FileNotFoundError):
    """A model whose download did not finish, or whose files went missing since."""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    primary: Path
    total_bytes: int


class ModelStore:
    """Locates and enumerates models under a models directory.

    Supports flat files (legacy), per-model subdirectories, namespaced models
    (org/model), and HuggingFace alias resolution.  All callers share a single
    resolution rule so CLI and server always agree on the primary file.
    """

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    def resolve(self, name: str) -> Path:
        """Return the primary GGUF path for *name*, or raise FileNotFoundError.

        Exact local path is always tried first; HuggingFace alias resolution
        only runs as a fallback so an explicit org/repo directory is never
        silently redirected to a different aliased model.  Raises ValueError
        for an unsafe name before touching the disk; a model whose files
        resolve outside the models directory counts as not found.  Raises
        IncompleteModelError for an unfinished download (never served).
        """
        validate_model_name(name)
        found = self._find_primary(name)
        if found is None and "/" in name:
            alias = self._alias(name)
            if alias:
                found = self._find_primary(alias)
        if found is not None:
            primary, reason = found
            if reason is None:
                return primary
            raise IncompleteModelError(
                f"Model '{name}' is incomplete: {reason}. "
                "Rerun 'drove models download' for it to finish the download."
            )
        raise FileNotFoundError(
            f"Model '{name}' not found in {self._dir}. "
            "Run 'drove models list' to see available models."
        )

    def find_root(self, name: str) -> Path | None:
        """Return the model directory (or legacy .gguf file), or None if absent.

        Exact local path is tried before alias resolution (mirrors resolve()).
        Raises ValueError for an unsafe name, like resolve().
        """
        validate_model_name(name)
        root = self._direct_root(name)
        if root is None and "/" in name:
            alias = self._alias(name)
            if alias:
                root = self._direct_root(alias)
        return root

    def list(self) -> list[ModelEntry]:
        """Return all models with their primary path and total on-disk size.

        Only names that resolve() accepts are listed; files or directories
        whose names break the naming rule stay hidden until renamed, and
        incomplete downloads stay hidden until finished.
        """
        results: list[ModelEntry] = []
        if not self._dir.exists():
            return results
        for p in sorted(self._dir.iterdir()):
            if not self._contains(p):
                continue
            if p.is_dir() and not p.name.startswith("."):
                if self._has_model_files(p):
                    self._add_entry(p, p.name, results)
                else:
                    # Namespace directory — scan one level deeper
                    for sub in sorted(p.iterdir()):
                        if sub.is_dir() and not sub.name.startswith(".") and self._contains(sub):
                            self._add_entry(sub, f"{p.name}/{sub.name}", results)
            elif (
                p.suffix.lower() == ".gguf"
                and not MMPROJ_RE.search(p.name)
                and p.is_file()
                and self._servable(p)
                and self._incomplete_reason(p) is None
            ):
                results.append(ModelEntry(p.stem, p, p.stat().st_size))
        return [entry for entry in results if _is_valid_name(entry.name)]

    def complete(self, prefix: str) -> builtins.list[str]:
        """Return model names that start with *prefix* (for shell completion).

        Derived from list() so completion never offers a name that list() or
        resolve() would reject.
        """
        return [e.name for e in self.list() if e.name.lower().startswith(prefix.lower())]

    # ── private helpers ──────────────────────────────────────────────────────

    def _contains(self, path: Path) -> bool:
        return is_within(self._dir, path)

    def _servable(self, primary: Path) -> bool:
        """True when *primary* and its sidecar TOML (if any) stay inside the store.

        Config reads and writes follow the sidecar, so an outward (or dangling)
        sidecar symlink would let them escape even though the weights are inside.
        """
        if not self._contains(primary):
            return False
        sidecar = primary.with_suffix(".toml")
        return not (sidecar.exists() or sidecar.is_symlink()) or self._contains(sidecar)

    def _alias(self, ref: str) -> str | None:
        """Resolve a HuggingFace alias, dropping any result that is not a safe name."""
        alias = resolve_model_alias(self._dir, ref)
        try:
            return validate_model_name(alias) if alias else None
        except ValueError:
            return None

    def _find_primary(self, name: str) -> tuple[Path, str | None] | None:
        """Find a local name's primary file and why it is incomplete (no alias resolution).

        TTS markers take priority, then GGUF (served by llama-server), then
        ONNX (served by the built-in ASR worker).  Other formats (.safetensors,
        .bin, .pt) are never primary, so callers such as ServerManager never
        receive a path that no backend can open.  The reason is None for a
        complete model.
        """
        subdir = self._dir / name
        if subdir.is_dir() and self._contains(subdir):
            primary = self._select_primary(subdir)
            if primary is not None:
                return primary, self._incomplete_reason(primary)
        candidate = self._dir / f"{name}.gguf"
        if (
            not MMPROJ_RE.search(candidate.name)
            and candidate.is_file()
            and self._servable(candidate)
        ):
            return candidate, self._incomplete_reason(candidate)
        return None

    def _select_primary(self, directory: Path) -> Path | None:
        """Choose a model directory's primary: the first servable file by format priority.

        Format priority is TTS, then GGUF, then ONNX.  Hidden paths (the Hub's
        ``.cache`` and partial files) never count, only a top-level ``.tts``
        marker registers a TTS model, and projector GGUFs (``mmproj``) are
        never primary, whatever their name sorts like.  Among GGUFs, a download
        sidecar whose first non-projector entry names the candidate wins;
        otherwise the lexically first servable one (shard 1 of a sharded
        model).  Returns None when nothing servable remains, which includes a
        directory holding only projectors.
        """
        files = sorted(
            f
            for f in directory.rglob("*")
            if f.suffix.lower() in _PRIMARY_EXTS
            and not any(part.startswith(".") for part in f.relative_to(directory).parts)
            and not (f.suffix.lower() == ".tts" and f.parent != directory)
            and not (f.suffix.lower() == ".gguf" and MMPROJ_RE.search(f.name))
            and f.is_file()
        )
        for suffix in _PRIMARY_EXTS:
            servable = [f for f in files if f.suffix.lower() == suffix and self._servable(f)]
            if not servable:
                continue
            if suffix == ".gguf":
                return next((f for f in servable if _sidecar_names(f)), servable[0])
            return servable[0]
        return None

    def _incomplete_reason(self, primary: Path) -> str | None:
        """Why *primary* cannot be served yet, or None when its model is complete.

        Every download record between the models directory and the primary
        applies, whatever name reached it (``org`` resolving into ``org/asr``
        still honours ``org/asr``'s record), and so does every record above
        the file a symlinked primary points to, so an alias never bypasses its
        target's record.  A record must say complete, list the primary itself
        (a TTS model's generated marker excepted) and every file it lists must
        still exist at its expected size.  Without any (downloads made before
        records existed) the files themselves are trusted.  Either way a split
        GGUF needs every shard.
        """
        try:
            real_dir, real_primary = self._dir.resolve(), primary.resolve()
        except OSError, RuntimeError:
            return "its files cannot be read"
        for top, path in ((self._dir, primary), (real_dir, real_primary)):
            directory = path.parent
            while directory != top and directory.is_relative_to(top):
                reason = _record_reason(directory, path)
                if reason is not None:
                    return reason
                directory = directory.parent
        return _split_reason(primary) or _split_reason(real_primary)

    def _direct_root(self, name: str) -> Path | None:
        """Return model directory or legacy flat file without alias resolution."""
        subdir = self._dir / name
        if subdir.is_dir():
            return subdir if self._contains(subdir) else None
        candidate = self._dir / f"{name}.gguf"
        if candidate.exists() and self._contains(candidate):
            return candidate
        return None

    @staticmethod
    def _has_model_files(directory: Path) -> bool:
        return any(f.suffix.lower() in _MODEL_EXTS for f in directory.iterdir() if f.is_file())

    def _add_entry(self, directory: Path, name: str, results: builtins.list[ModelEntry]) -> None:
        # Same choice as _find_primary, so list() only offers what resolve() serves.
        primary = self._select_primary(directory)
        if primary is None or self._incomplete_reason(primary) is not None:
            return
        total = sum(
            f.stat().st_size for f in directory.rglob("*") if f.is_file() and self._contains(f)
        )
        results.append(ModelEntry(name, primary, total))


def _sidecar_names(candidate: Path) -> bool:
    """True when *candidate*'s download sidecar names it as the first weights file.

    The first non-projector entry of ``[download].files`` is the intended
    primary (the downloader writes weights first, shard 1 leading).  Remote
    subpaths are reduced to the basename, as the downloader flattens them.
    Absent, stale or unreadable metadata is no preference.
    """
    try:
        info = load_download_info(candidate)
    except OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, TypeError:
        return False
    if info is None:
        return False
    first = next((f for f in info.files if not MMPROJ_RE.search(Path(f).name)), None)
    return first is not None and Path(first).name == candidate.name


def _record_reason(root: Path, primary: Path) -> str | None:
    """Why the download record under *root* rules out *primary*, or None (also without one)."""
    state = read_state(root)
    if state is UNREADABLE:
        return "its download record is unreadable"
    if state is None:
        return None
    if not state.complete:
        return "the download did not finish"
    for rel, size in state.files.items():
        path = root / rel
        if not (path.is_file() and is_within(root, path)):
            return f"{rel} is missing"
        if size and path.stat().st_size != size:
            return f"{rel} does not have its expected size"
    # The record vouches only for files it downloaded; the TTS marker is
    # generated after the weights, so it is the one entry never recorded.
    is_marker = primary.suffix.lower() == ".tts" and primary.parent == root
    if not is_marker and primary.relative_to(root).as_posix() not in state.files:
        return f"{primary.relative_to(root).as_posix()} is not part of its download record"
    return None


def _split_reason(primary: Path) -> str | None:
    """Why a split GGUF *primary* cannot be served, or None (also for unsplit files)."""
    match = _SPLIT_GGUF.match(primary.name)
    if match is None:
        return None
    if int(match["index"]) != 1:
        return f"{primary.name} is not the first shard"
    count = match["count"]
    stem, suffix = match["stem"], primary.suffix
    missing = [
        index
        for index in range(2, int(count) + 1)
        if not (primary.parent / f"{stem}-{index:05d}-of-{count}{suffix}").is_file()
    ]
    if missing:
        return f"{len(missing)} of {int(count)} shards are missing"
    return None
