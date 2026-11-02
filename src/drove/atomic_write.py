"""Atomic file writes: a private temporary sibling, then a rename into place.

Readers see the old file or the new one, never a mix. The temporary file is
created exclusively under a fresh random name, so a file or symlink planted at
a predictable name can neither redirect the write nor be reused, and it is
removed again whenever the write fails. The final rename replaces a symlink at
the target itself rather than writing through it.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_NAME_ATTEMPTS = 100


def stage_text(target: Path, text: str, *, mode: int = 0o600, keep_mode: bool = False) -> Path:
    """Write *text* (UTF-8) to a new temporary file beside *target* and return its path.

    *mode* is the creation mode, filtered by the umask as for ``open(2)``:
    ``0o600`` keeps the file private (as ``tempfile.mkstemp`` does), ``0o666``
    gives the umask default. With *keep_mode*, an existing *target*'s
    permission bits are copied instead, so a private file stays private.
    The caller renames the result into place or unlinks it; it is unlinked
    here if anything fails first. The target's directory must exist.
    """
    fd, tmp = _create_temp(target, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if keep_mode and target.exists():
            os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def write_text_atomic(
    target: Path, text: str, *, mode: int = 0o600, keep_mode: bool = False
) -> None:
    """Replace *target* with *text* atomically; see :func:`stage_text` for *mode*."""
    tmp = stage_text(target, text, mode=mode, keep_mode=keep_mode)
    try:
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _create_temp(target: Path, mode: int) -> tuple[int, Path]:
    """Exclusively create ``.<target name>.<random>.tmp`` beside *target*."""
    for _ in range(_NAME_ATTEMPTS):
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
        try:
            return os.open(tmp, _CREATE_FLAGS, mode), tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"No free temporary name beside {target}")
