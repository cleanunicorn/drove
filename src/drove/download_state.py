"""Per-model download record: which files a download expects and whether it finished.

The downloader writes ``<model root>/.drove-download.json`` as ``incomplete``
before its first transfer and as ``complete`` after its last, so the model
store can tell an interrupted download from a finished one.  A model root
without the file predates this record and is judged by its files alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

MANIFEST_NAME = ".drove-download.json"
_VERSION = 1

type State = Literal["incomplete", "complete"]


@dataclass(frozen=True)
class DownloadState:
    """A parsed record; ``files`` maps root-relative POSIX paths to expected bytes (0 = unknown)."""

    state: State
    files: dict[str, int]

    @property
    def complete(self) -> bool:
        return self.state == "complete"


# Returned for a record that exists but cannot be trusted: never complete.
UNREADABLE = DownloadState("incomplete", {})


def read_state(root: Path) -> DownloadState | None:
    """Return the record under *root*, None when there is none, UNREADABLE when invalid."""
    path = root / MANIFEST_NAME
    if not (path.exists() or path.is_symlink()):
        return None
    try:
        if path.is_symlink() or not path.is_file():
            return UNREADABLE
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, ValueError:
        return UNREADABLE
    if not isinstance(data, dict) or data.get("version") != _VERSION:
        return UNREADABLE
    state, files = data.get("state"), data.get("files")
    if state not in ("incomplete", "complete") or not isinstance(files, dict):
        return UNREADABLE
    parsed: dict[str, int] = {}
    for rel, size in files.items():
        if not _is_safe_relative(rel) or type(size) is not int or size < 0:
            return UNREADABLE
        parsed[rel] = size
    return DownloadState(state, parsed)


def write_state(root: Path, state: State, files: dict[str, int]) -> None:
    """Atomically replace the record under *root* (temporary sibling, then rename)."""
    target = root / MANIFEST_NAME
    temp = root / f"{MANIFEST_NAME}.tmp"
    payload = {"version": _VERSION, "state": state, "files": dict(sorted(files.items()))}
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)


def _is_safe_relative(rel: object) -> bool:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        return False
    path = PurePosixPath(rel)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)
