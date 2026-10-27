"""Single authority for locating model files on disk."""

from __future__ import annotations

import builtins
import re
from dataclasses import dataclass
from pathlib import Path

from drove.model_config import resolve_model_alias

_MODEL_EXTS = frozenset({".gguf", ".onnx", ".safetensors", ".bin", ".pt", ".tts"})

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
        resolve outside the models directory counts as not found.
        """
        validate_model_name(name)
        primary = self._find_primary(name)
        if primary is None and "/" in name:
            alias = self._alias(name)
            if alias:
                primary = self._find_primary(alias)
        if primary is not None:
            return primary
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
        whose names break the naming rule stay hidden until renamed.
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
            elif p.suffix.lower() == ".gguf" and p.is_file() and self._servable(p):
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

    def _find_primary(self, name: str) -> Path | None:
        """Find the primary model file for a local name (no alias resolution).

        GGUF takes priority (served by llama-server), then ONNX (served by the
        built-in ASR worker).  Other formats (.safetensors, .bin, .pt) are
        intentionally excluded so callers such as ServerManager never receive
        a path that no backend can open.
        """
        subdir = self._dir / name
        if subdir.is_dir() and self._contains(subdir):
            for found in (subdir.glob("*.tts"), subdir.rglob("*.gguf"), subdir.rglob("*.onnx")):
                contained = sorted(p for p in found if self._servable(p))
                if contained:
                    return contained[0]
        candidate = self._dir / f"{name}.gguf"
        if candidate.exists() and self._servable(candidate):
            return candidate
        return None

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
        files = [f for f in directory.rglob("*") if f.is_file() and self._contains(f)]
        total = sum(f.stat().st_size for f in files)
        ggufs = sorted(f for f in files if f.suffix.lower() == ".gguf")
        onnx = sorted(f for f in files if f.suffix.lower() == ".onnx")
        tts = sorted(f for f in files if f.suffix.lower() == ".tts")
        if tts or ggufs or onnx:
            # Same choice as _find_primary: the first servable file, by format priority.
            primary = next((f for f in (*tts, *ggufs, *onnx) if self._servable(f)), None)
        else:
            primary = min(files, default=None)
            if primary is not None and not self._servable(primary):
                primary = None
        if primary is not None:
            results.append(ModelEntry(name, primary, total))
