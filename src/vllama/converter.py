"""Conversion from HuggingFace format (safetensors/bin) to GGUF."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"

# Non-GGUF model file extensions that need conversion
SOURCE_EXTS = {".safetensors", ".bin", ".pt"}

# Common installation locations for the llama.cpp Python scripts
_SEARCH_DIRS = [
    Path("/usr/share/llama.cpp"),
    Path("/usr/local/share/llama.cpp"),
    Path.home() / ".local" / "share" / "llama.cpp",
    Path("/opt/llama.cpp"),
]


def needs_conversion(path: Path) -> bool:
    """True if path (file or dir) contains non-GGUF model files."""
    if path.is_file():
        return path.suffix.lower() in SOURCE_EXTS
    if path.is_dir():
        return any(
            f.suffix.lower() in SOURCE_EXTS for f in path.rglob("*") if f.is_file()
        )
    return False


def find_convert_script(llama_server_bin: str = "llama-server") -> Path | None:
    """Auto-detect convert_hf_to_gguf.py.

    Search order:
      1. Same directory as the llama-server binary
      2. Common system installation dirs
      3. PATH
    """
    server_bin = shutil.which(llama_server_bin)
    if server_bin:
        candidate = Path(server_bin).parent / _CONVERT_SCRIPT_NAME
        if candidate.exists():
            return candidate

    for d in _SEARCH_DIRS:
        candidate = d / _CONVERT_SCRIPT_NAME
        if candidate.exists():
            return candidate

    found = shutil.which(_CONVERT_SCRIPT_NAME)
    if found:
        return Path(found)

    return None


def convert_to_gguf(
    model_dir: Path,
    output_path: Path,
    script: Path,
    output_type: str = "f16",
) -> None:
    """Run convert_hf_to_gguf.py on model_dir, writing GGUF to output_path.

    Uses the same Python interpreter that is running vllama. If dependencies
    like ``transformers`` or ``gguf`` are missing you will see an ImportError
    from the script — install them with:

        pip install transformers gguf sentencepiece
    """
    cmd = [
        sys.executable,
        str(script),
        str(model_dir),
        "--outtype",
        output_type,
        "--outfile",
        str(output_path),
    ]
    logger.info("Converting: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Conversion failed (exit {e.returncode}). "
            "Make sure 'transformers', 'gguf', and 'sentencepiece' are installed:\n"
            "  pip install transformers gguf sentencepiece"
        ) from e


def remove_source_files(model_dir: Path) -> list[Path]:
    """Delete non-GGUF model files from model_dir. Returns list of deleted paths."""
    deleted: list[Path] = []
    for f in sorted(model_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in SOURCE_EXTS:
            f.unlink()
            deleted.append(f)
    return deleted
