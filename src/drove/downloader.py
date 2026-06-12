"""HuggingFace model discovery and download logic."""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

type ProgressCallback = Callable[[int, int, str], None]


class FileStatus(enum.Enum):
    """Status of a local file relative to its remote counterpart."""

    MISSING = "missing"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


# Matches shard suffixes like -00001-of-00012
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}", re.IGNORECASE)

# Model file extensions we care about
_MODEL_EXTS = {".gguf", ".safetensors", ".bin", ".pt"}

# ONNX model weights (".data" covers external-data files like model.onnx.data)
_ONNX_EXTS = {".onnx", ".onnx_data", ".data"}

# Small support files an ONNX ASR model needs alongside its weights
_ONNX_EXTRA_EXTS = {".txt", ".json", ".model"}

# Quantization variant infix in ONNX filenames (e.g. encoder-model.int8.onnx)
_ONNX_QUANT_RE = re.compile(r"\.(int8|uint8|fp16|int8_fp16)\.", re.IGNORECASE)

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
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return (model_files, mmproj_files, extra_files) as {filename: size_bytes} dicts.

    For ONNX repos (e.g. ASR models like Parakeet), model files are the ONNX
    weights and extra_files are the small support files (vocab, config) the
    runtime needs next to them.
    """
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    siblings = [(s.rfilename, s.size or 0) for s in info.siblings or []]

    onnx_mode = any(PurePosixPath(p).suffix.lower() == ".onnx" for p, _ in siblings)
    model_files: dict[str, int] = {}
    mmproj_files: dict[str, int] = {}
    extra_files: dict[str, int] = {}

    for path, size in siblings:
        suffix = PurePosixPath(path).suffix.lower()
        if onnx_mode:
            if suffix in _ONNX_EXTS:
                model_files[path] = size
            elif suffix in _ONNX_EXTRA_EXTS:
                extra_files[path] = size
        elif suffix in _MODEL_EXTS:
            if _MMPROJ_RE.search(Path(path).name):
                mmproj_files[path] = size
            else:
                model_files[path] = size
    return model_files, mmproj_files, extra_files


def is_onnx_files(files: dict[str, int]) -> bool:
    """True when the model files are ONNX weights (ASR backend)."""
    return any(PurePosixPath(f).suffix.lower() == ".onnx" for f in files)


def available_onnx_quants(files: dict[str, int]) -> dict[str, int]:
    """Return {variant: total_bytes} for each ONNX quant variant in *files*.

    Unquantized files are grouped under "default". Insertion order is the
    order in which each variant is first encountered.
    """
    out: dict[str, int] = {}
    for fname, size in files.items():
        m = _ONNX_QUANT_RE.search(Path(fname).name)
        tag = m.group(1).lower() if m else "default"
        out[tag] = out.get(tag, 0) + size
    return out


def filter_onnx_quant(files: dict[str, int], quant: str | None) -> dict[str, int]:
    """Filter ONNX files by quantization variant.

    With *quant* set (e.g. "int8"), keep only files carrying that infix.
    Without it, keep only the unquantized variants.
    """
    if quant is not None:
        needle = f".{quant.lower()}."
        return {f: s for f, s in files.items() if needle in Path(f).name.lower()}
    # No quant requested: drop quantized variants, keeping the full-precision files.
    return {f: s for f, s in files.items() if not _ONNX_QUANT_RE.search(Path(f).name)}


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
        extra_files: dict[str, int] | None = None,
        onnx_files: dict[str, int] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.files = files  # preserves insertion order
        self.local_name = local_name
        self.sharded = sharded
        self.mmproj_files = mmproj_files or {}
        self.extra_files = extra_files or {}
        # Full pre-quant-filter set of ONNX weights, so the CLI can offer
        # the other variants (e.g. int8) after a default resolution.
        self.onnx_files = onnx_files or {}

    @property
    def file_names(self) -> list[str]:
        return list(self.files.keys())

    @property
    def is_asr(self) -> bool:
        """True when this plan downloads an ONNX (ASR backend) model."""
        return is_onnx_files(self.files)

    @property
    def total_bytes(self) -> int:
        return (
            sum(self.files.values())
            + sum(self.mmproj_files.values())
            + sum(self.extra_files.values())
        )

    def destination(self, models_dir: Path) -> Path:
        """Return the model directory (always a directory)."""
        return models_dir / self.local_name

    def _local_path(self, repo_file: str, models_dir: Path) -> Path:
        """Return the expected local path for a repo file after flattening."""
        return self.destination(models_dir) / Path(repo_file).name

    def _all_remote_files(self) -> dict[str, int]:
        """Return all files (model + mmproj + extras) with their remote sizes."""
        all_files: dict[str, int] = {}
        all_files.update(self.files)
        all_files.update(self.mmproj_files)
        all_files.update(self.extra_files)
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
        all_files = (
            sorted(self.file_names)
            + sorted(self.mmproj_files.keys())
            + sorted(self.extra_files.keys())
        )
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


def resolve_download(
    model_ref: str,
    name_override: str | None = None,
) -> DownloadPlan:
    """Resolve a model reference to a DownloadPlan without downloading.

    Fetches repo metadata (including file sizes) from HuggingFace.
    Raises ValueError if no matching files are found.
    """
    repo_id, quant = parse_model_ref(model_ref)
    files, mmproj_files, extra_files = _fetch_files_with_sizes(repo_id)

    if not files:
        raise ValueError(f"No model files found in repo '{repo_id}'.")

    onnx_files: dict[str, int] = {}
    if is_onnx_files(files):
        # ONNX repos carry quant variants as filename infixes (model.int8.onnx)
        onnx_files = files
        matched = filter_onnx_quant(files, quant)
        if not matched:
            raise ValueError(
                f"No files matching quantization '{quant}' in '{repo_id}'.\n"
                "Available variants: " + _summarise_onnx_quants(list(files.keys()))
            )
        files = matched
    elif quant:
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
        extra_files=extra_files,
        onnx_files=onnx_files,
    )


def _summarise_onnx_quants(files: list[str]) -> str:
    """List the quant variants present in ONNX filenames (plus 'default')."""
    return ", ".join(sorted(available_onnx_quants(dict.fromkeys(files, 0))))


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
