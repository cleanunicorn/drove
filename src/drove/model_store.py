"""Single authority for locating model files on disk."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from drove.model_config import resolve_model_alias

_MODEL_EXTS = frozenset({".gguf", ".safetensors", ".bin", ".pt"})
_PRIMARY_PRIORITY = (".gguf", ".safetensors", ".bin", ".pt")


class ModelBackend(StrEnum):
    LLAMA_CPP = "llama.cpp"
    MLX = "mlx"


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
        """Return the primary model path for *name*, or raise FileNotFoundError.

        Exact local path is always tried first; HuggingFace alias resolution
        only runs as a fallback so an explicit org/repo directory is never
        silently redirected to a different aliased model.
        """
        primary = self._find_primary(name)
        if primary is None and "/" in name:
            alias = resolve_model_alias(self._dir, name)
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
        """
        root = self._direct_root(name)
        if root is None and "/" in name:
            alias = resolve_model_alias(self._dir, name)
            if alias:
                root = self._direct_root(alias)
        return root

    def list(self) -> list[ModelEntry]:
        """Return all models with their primary path and total on-disk size."""
        results: list[ModelEntry] = []
        if not self._dir.exists():
            return results
        for p in sorted(self._dir.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                if self._has_model_files(p):
                    self._add_entry(p, p.name, results)
                else:
                    # Namespace directory — scan one level deeper
                    for sub in sorted(p.iterdir()):
                        if sub.is_dir() and not sub.name.startswith("."):
                            self._add_entry(sub, f"{p.name}/{sub.name}", results)
            elif p.suffix.lower() == ".gguf" and p.is_file():
                results.append(ModelEntry(p.stem, p, p.stat().st_size))
        return results

    def complete(self, prefix: str) -> list[str]:
        """Return model names that start with *prefix* (for shell completion)."""
        names: list[str] = []
        if not self._dir.exists():
            return names
        for p in sorted(self._dir.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                if self._has_model_files(p):
                    names.append(p.name)
                else:
                    for sub in sorted(p.iterdir()):
                        if sub.is_dir() and not sub.name.startswith("."):
                            names.append(f"{p.name}/{sub.name}")
            elif p.suffix.lower() == ".gguf" and p.is_file():
                names.append(p.stem)
        return [n for n in names if n.lower().startswith(prefix.lower())]

    # ── private helpers ──────────────────────────────────────────────────────

    def resolve_backend(self, name: str) -> ModelBackend:
        """Return backend type for *name*, or raise FileNotFoundError."""
        primary = self.resolve(name)
        return self.backend_for_path(primary)

    @staticmethod
    def backend_for_path(model_path: Path) -> ModelBackend:
        """Return backend inferred from model file extension."""
        if model_path.suffix.lower() == ".gguf":
            return ModelBackend.LLAMA_CPP
        return ModelBackend.MLX

    def _find_primary(self, name: str) -> Path | None:
        """Find the primary model file for a local name (no alias resolution)."""
        subdir = self._dir / name
        if subdir.is_dir():
            for ext in _PRIMARY_PRIORITY:
                matches = sorted(subdir.rglob(f"*{ext}"))
                if matches:
                    return matches[0]
        for ext in _PRIMARY_PRIORITY:
            candidate = self._dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _direct_root(self, name: str) -> Path | None:
        """Return model directory or legacy flat file without alias resolution."""
        subdir = self._dir / name
        if subdir.is_dir():
            return subdir
        for ext in _PRIMARY_PRIORITY:
            candidate = self._dir / f"{name}{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _has_model_files(directory: Path) -> bool:
        return any(f.suffix.lower() in _MODEL_EXTS for f in directory.iterdir() if f.is_file())

    def _add_entry(self, directory: Path, name: str, results: list[ModelEntry]) -> None:
        files = [f for f in directory.rglob("*") if f.is_file()]
        total = sum(f.stat().st_size for f in files)
        primary = None
        for ext in _PRIMARY_PRIORITY:
            matches = sorted(f for f in files if f.suffix.lower() == ext)
            if matches:
                primary = matches[0]
                break
        if primary is None:
            primary = sorted(files)[0] if files else None
        if primary:
            results.append(ModelEntry(name, primary, total))
