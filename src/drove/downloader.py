"""HuggingFace model discovery and download logic."""

from __future__ import annotations

import enum
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download


class FileStatus(enum.Enum):
    """Status of a local file relative to its remote counterpart."""

    MISSING = "missing"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


# Matches shard suffixes like -00001-of-00012
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)

# Model file extensions we care about
_MODEL_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}

# Pattern matching mmproj (multimodal projection) filenames
_MMPROJ_RE = re.compile(r"mmproj", re.IGNORECASE)


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


def _fetch_files_with_sizes(
    repo_id: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (model_files, mmproj_files) as {filename: size_bytes} dicts."""
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    model_files: dict[str, int] = {}
    mmproj_files: dict[str, int] = {}
    for sibling in info.siblings or []:
        path = sibling.rfilename
        if PurePosixPath(path).suffix.lower() not in _MODEL_EXTS:
            continue
        if _MMPROJ_RE.search(Path(path).name):
            mmproj_files[path] = sibling.size or 0
        else:
            model_files[path] = sibling.size or 0
    return model_files, mmproj_files


def filter_by_quant(files: dict[str, int], quant: str) -> dict[str, int]:
    """Filter files by exact quantization tag match (case-insensitive)."""
    q = quant.upper()
    return {f: s for f, s in files.items() if quant_tag(f) == q}


_QUANT_RE = re.compile(r"(IQ\w+|Q\d+_\w+|Q\d+|BF\d+|F\d+)", re.IGNORECASE)


def quant_tag(filename: str) -> str | None:
    """Extract the quantization tag from a filename, or None if absent."""
    m = _QUANT_RE.search(Path(filename).name)
    return m.group(0).upper() if m else None


def available_quants(files: dict[str, int]) -> dict[str, int]:
    """Return {QUANT_TAG: total_bytes} for each detectable quant in *files*.

    Files without a recognisable quant tag are skipped. Insertion order is
    the order in which each quant is first encountered.
    """
    out: dict[str, int] = {}
    for fname, size in files.items():
        tag = quant_tag(fname)
        if tag is None:
            continue
        out[tag] = out.get(tag, 0) + size
    return out


# Preferred mmproj variants, best first.
_MMPROJ_PREF = ["f16", "bf16", "f32"]


def pick_mmproj(files: dict[str, int]) -> dict[str, int]:
    """Select a single mmproj file from multiple variants.

    Preference order: F16 > BF16 > F32 > smallest file.
    """
    if len(files) <= 1:
        return files
    names = list(files.keys())
    for tag in _MMPROJ_PREF:
        for name in names:
            if tag in Path(name).stem.lower():
                return {name: files[name]}
    # Fallback: pick smallest
    smallest = min(names, key=lambda n: files[n])
    return {smallest: files[smallest]}


def is_sharded(files: list[str]) -> bool:
    """True when files contain shard suffixes (-00001-of-00012)."""
    return any(_SHARD_RE.search(Path(f).name) for f in files)


def first_shard(files: list[str]) -> str:
    """Return the first shard from a sorted list of shard filenames."""
    return sorted(files, key=lambda f: Path(f).name)[0]


def infer_local_name(repo_id: str, files: list[str], quant: str | None) -> str:
    """Derive a clean local model name using the repo/name:quant format.

    Examples:
        infer_local_name("unsloth/Qwen3-8B-GGUF", [...], "Q8_0")
        → "unsloth/Qwen3-8B-GGUF:Q8_0"

        infer_local_name("unsloth/Qwen3-8B-GGUF", [...], None)
        → "unsloth/Qwen3-8B-GGUF"
    """
    if quant:
        return f"{repo_id}:{quant}"
    return repo_id


class DownloadPlan:
    """Resolved download plan — all metadata known before any I/O."""

    def __init__(
        self,
        repo_id: str,
        files: dict[str, int],  # filename → size in bytes
        local_name: str,
        sharded: bool,
        mmproj_files: dict[str, int] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.files = files  # preserves insertion order
        self.local_name = local_name
        self.sharded = sharded
        self.mmproj_files = mmproj_files or {}

    @property
    def file_names(self) -> list[str]:
        return list(self.files.keys())

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values()) + sum(self.mmproj_files.values())

    def destination(self, models_dir: Path) -> Path:
        """Return the model directory (always a directory)."""
        return models_dir / self.local_name

    def _local_path(self, repo_file: str, models_dir: Path) -> Path:
        """Return the expected local path for a repo file after flattening."""
        return self.destination(models_dir) / Path(repo_file).name

    def _all_remote_files(self) -> dict[str, int]:
        """Return all files (model + mmproj) with their remote sizes."""
        all_files: dict[str, int] = {}
        all_files.update(self.files)
        all_files.update(self.mmproj_files)
        return all_files

    def check_local_files(self, models_dir: Path) -> dict[str, tuple[FileStatus, int]]:
        """Check local status of each file.

        Returns {repo_filename: (status, local_size_bytes)}.
        """
        result: dict[str, tuple[FileStatus, int]] = {}
        for repo_file, remote_size in self._all_remote_files().items():
            local = self._local_path(repo_file, models_dir)
            if not local.exists():
                result[repo_file] = (FileStatus.MISSING, 0)
            elif local.stat().st_size >= remote_size:
                result[repo_file] = (FileStatus.COMPLETE, local.stat().st_size)
            else:
                result[repo_file] = (
                    FileStatus.INCOMPLETE,
                    local.stat().st_size,
                )
        return result

    def execute(
        self,
        models_dir: Path,
        progress_cb: ProgressCallback | None = None,
    ) -> Path:
        """Download all files and return the path to the model entry point."""
        dest = self.destination(models_dir)
        dest.mkdir(parents=True, exist_ok=True)
        all_files = sorted(self.file_names) + sorted(self.mmproj_files.keys())
        total_count = len(all_files)
        statuses = self.check_local_files(models_dir)

        for i, repo_file in enumerate(all_files):
            status, _ = statuses.get(repo_file, (FileStatus.MISSING, 0))
            if status == FileStatus.COMPLETE:
                if progress_cb:
                    progress_cb(i + 1, total_count, f"{repo_file} (skipped)")
                continue
            if progress_cb:
                progress_cb(i + 1, total_count, repo_file)
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
    files, mmproj_files = _fetch_files_with_sizes(repo_id)

    if not files:
        raise ValueError(f"No model files found in repo '{repo_id}'.")

    if quant:
        matched = filter_by_quant(files, quant)
        if not matched:
            available = _summarise_quants(list(files.keys()))
            raise ValueError(
                f"No files matching quantization '{quant}' in '{repo_id}'.\nAvailable: {available}"
            )
        files = matched

    mmproj_files = pick_mmproj(mmproj_files)
    sharded = is_sharded(list(files.keys()))
    local_name = name_override or infer_local_name(repo_id, list(files.keys()), quant)

    return DownloadPlan(
        repo_id=repo_id,
        files=files,
        local_name=local_name,
        sharded=sharded,
        mmproj_files=mmproj_files,
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
