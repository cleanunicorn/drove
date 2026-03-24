"""HuggingFace model discovery and download logic."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

# Matches shard suffixes like -00001-of-00012
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)

# Model file extensions we care about
_MODEL_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """Split 'org/repo:QUANT' into (repo_id, quant).

    >>> parse_model_ref("unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M")
    ('unsloth/Qwen3.5-35B-A3B-GGUF', 'Q4_K_M')
    >>> parse_model_ref("unsloth/Qwen3.5-35B-A3B-GGUF")
    ('unsloth/Qwen3.5-35B-A3B-GGUF', None)
    """
    if ":" in ref:
        repo_id, quant = ref.rsplit(":", 1)
        return repo_id, quant.strip() or None
    return ref, None


def _fetch_files_with_sizes(repo_id: str) -> dict[str, int]:
    """Return {filename: size_bytes} for all model files in the repo."""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    result: dict[str, int] = {}
    for sibling in info.siblings or []:
        path = sibling.rfilename
        if PurePosixPath(path).suffix.lower() in _MODEL_EXTS:
            result[path] = sibling.size or 0
    return result


def filter_by_quant(files: dict[str, int], quant: str) -> dict[str, int]:
    """Filter files by case-insensitive quantization tag match."""
    q = quant.lower()
    return {f: s for f, s in files.items() if q in Path(f).name.lower()}


def is_sharded(files: list[str]) -> bool:
    """True when files contain shard suffixes (-00001-of-00012)."""
    return any(_SHARD_RE.search(Path(f).name) for f in files)


def first_shard(files: list[str]) -> str:
    """Return the first shard from a sorted list of shard filenames."""
    return sorted(files, key=lambda f: Path(f).name)[0]


def infer_local_name(repo_id: str, files: list[str], quant: str | None) -> str:
    """Derive a clean local model name.

    For a single file: use the filename stem (sans shard suffix).
    For multiple files: use repo basename + quant tag (if any).
    """
    if len(files) == 1:
        stem = Path(files[0]).stem
        return _SHARD_RE.sub("", stem).rstrip("-")

    repo_name = repo_id.split("/")[-1]
    if quant:
        return f"{repo_name}-{quant}"
    return repo_name


class DownloadPlan:
    """Resolved download plan — all metadata known before any I/O."""

    def __init__(
        self,
        repo_id: str,
        files: dict[str, int],  # filename → size in bytes
        local_name: str,
        sharded: bool,
    ) -> None:
        self.repo_id = repo_id
        self.files = files          # preserves insertion order
        self.local_name = local_name
        self.sharded = sharded

    @property
    def file_names(self) -> list[str]:
        return list(self.files.keys())

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values())

    @property
    def dest_is_dir(self) -> bool:
        return self.sharded or len(self.files) > 1

    def destination(self, models_dir: Path) -> Path:
        if self.dest_is_dir:
            return models_dir / self.local_name
        return models_dir / f"{self.local_name}.gguf"

    def already_exists(self, models_dir: Path) -> bool:
        return self.destination(models_dir).exists()

    def execute(
        self,
        models_dir: Path,
        progress_cb: "ProgressCallback | None" = None,
    ) -> Path:
        """Download all files and return the path to the model entry point."""
        dest = self.destination(models_dir)

        if self.dest_is_dir:
            dest.mkdir(parents=True, exist_ok=True)
            for i, repo_file in enumerate(sorted(self.file_names)):
                if progress_cb:
                    progress_cb(i + 1, len(self.files), repo_file)
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_file,
                    local_dir=str(dest),
                )
                # Flatten any repo subdirectory structure
                downloaded_path = Path(downloaded)
                flat_path = dest / downloaded_path.name
                if downloaded_path != flat_path and downloaded_path.exists():
                    downloaded_path.rename(flat_path)
            return dest / Path(first_shard(self.file_names)).name
        else:
            if progress_cb:
                progress_cb(1, 1, self.file_names[0])
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.file_names[0],
                local_dir=str(models_dir),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.name != dest.name:
                downloaded_path.rename(dest)
            return dest


ProgressCallback = "Callable[[int, int, str], None]"


def resolve_download(
    model_ref: str,
    name_override: str | None = None,
) -> DownloadPlan:
    """Resolve a model reference to a DownloadPlan without downloading.

    Fetches repo metadata (including file sizes) from HuggingFace.
    Raises ValueError if no matching files are found.
    """
    repo_id, quant = parse_model_ref(model_ref)
    files = _fetch_files_with_sizes(repo_id)

    if not files:
        raise ValueError(f"No model files found in repo '{repo_id}'.")

    if quant:
        matched = filter_by_quant(files, quant)
        if not matched:
            available = _summarise_quants(list(files.keys()))
            raise ValueError(
                f"No files matching quantization '{quant}' in '{repo_id}'.\n"
                f"Available: {available}"
            )
        files = matched

    sharded = is_sharded(list(files.keys()))
    local_name = name_override or infer_local_name(repo_id, list(files.keys()), quant)

    return DownloadPlan(
        repo_id=repo_id,
        files=files,
        local_name=local_name,
        sharded=sharded,
    )


def _summarise_quants(files: list[str]) -> str:
    """Extract unique quantization tags from filenames for display."""
    quant_re = re.compile(r"(IQ\w+|Q\d+_\w+|F\d+|BF\d+)", re.IGNORECASE)
    seen: dict[str, str] = {}
    for f in files:
        for m in quant_re.finditer(Path(f).name):
            key = m.group(0).upper()
            seen[key] = m.group(0)
    if not seen:
        return ", ".join(sorted({Path(f).suffix for f in files}))
    return ", ".join(sorted(seen.values()))
