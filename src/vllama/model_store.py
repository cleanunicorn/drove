"""Single authority for locating model files on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vllama.model_config import resolve_model_alias

_MODEL_EXTS = frozenset({".gguf", ".safetensors", ".bin", ".pt"})


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
        """Return the primary GGUF path for *name*, or raise FileNotFoundError."""
        primary = self._find_primary(self._canonical(name))
        if primary is not None:
            return primary
        raise FileNotFoundError(
            f"Model '{name}' not found in {self._dir}. "
            "Run 'vllama models list' to see available models."
        )

    def find_root(self, name: str) -> Path | None:
        """Return the model directory (or legacy .gguf file), or None if absent."""
        cname = self._canonical(name)
        subdir = self._dir / cname
        if subdir.is_dir():
            return subdir
        candidate = self._dir / f"{cname}.gguf"
        if candidate.exists():
            return candidate
        return None

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

    def _canonical(self, name: str) -> str:
        """Resolve a HuggingFace alias (org/repo[:quant]) to a local name."""
        if "/" in name:
            local = resolve_model_alias(self._dir, name)
            if local:
                return local
        return name

    def _find_primary(self, name: str) -> Path | None:
        """Find the primary model file for a local (already-resolved) name."""
        subdir = self._dir / name
        if subdir.is_dir():
            ggufs = sorted(subdir.rglob("*.gguf"))
            if ggufs:
                return ggufs[0]
            others = sorted(p for p in subdir.rglob("*") if p.suffix.lower() in _MODEL_EXTS)
            if others:
                return others[0]
        candidate = self._dir / f"{name}.gguf"
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _has_model_files(directory: Path) -> bool:
        return any(f.suffix.lower() in _MODEL_EXTS for f in directory.iterdir() if f.is_file())

    def _add_entry(self, directory: Path, name: str, results: list[ModelEntry]) -> None:
        files = [f for f in directory.rglob("*") if f.is_file()]
        total = sum(f.stat().st_size for f in files)
        ggufs = sorted(f for f in files if f.suffix.lower() == ".gguf")
        primary = ggufs[0] if ggufs else (sorted(files)[0] if files else None)
        if primary:
            results.append(ModelEntry(name, primary, total))
