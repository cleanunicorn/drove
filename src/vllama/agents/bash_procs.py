"""BgProcs: async manager for long-running shell commands.

Each started command runs as a subprocess under asyncio. A reader task drains
stdout+stderr (merged) into a ring buffer. Callers poll output by offset,
terminate via kill, or enumerate via list().
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BgProcess:
    shell_id: str
    command: str
    cwd: Path
    proc: asyncio.subprocess.Process
    buffer: bytearray = field(default_factory=bytearray)
    dropped_bytes: int = 0
    reader_task: asyncio.Task[None] | None = None
    exit_code: int | None = None
    exited_at: float | None = None

    @property
    def pid(self) -> int:
        return self.proc.pid


def _kill_pgrp(pid: int, sig: signal.Signals) -> None:
    """Send sig to the process group of pid, ignoring ProcessLookupError."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, OSError):
        pass


class BgProcs:
    """Singleton-per-ChatApp manager for background shell procs."""

    def __init__(self, buffer_bytes: int = 65_536, gc_seconds: float = 600.0) -> None:
        self._buffer_bytes = buffer_bytes
        self._gc_seconds = gc_seconds
        self._procs: dict[str, BgProcess] = {}

    async def start(self, command: str, cwd: Path) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        shell_id = uuid.uuid4().hex[:8]
        bp = BgProcess(shell_id=shell_id, command=command, cwd=cwd, proc=proc)
        bp.reader_task = asyncio.create_task(self._reader(bp))
        self._procs[shell_id] = bp
        return shell_id

    async def _reader(self, bp: BgProcess) -> None:
        assert bp.proc.stdout is not None
        try:
            while True:
                chunk = await bp.proc.stdout.read(4096)
                if not chunk:
                    break
                self._append(bp, chunk)
        finally:
            bp.exit_code = await bp.proc.wait()
            bp.exited_at = time.monotonic()

    def _append(self, bp: BgProcess, chunk: bytes) -> None:
        cap = self._buffer_bytes
        bp.buffer.extend(chunk)
        if len(bp.buffer) > cap:
            overflow = len(bp.buffer) - cap
            del bp.buffer[:overflow]
            bp.dropped_bytes += overflow

    def get(self, shell_id: str) -> BgProcess | None:
        return self._procs.get(shell_id)

    async def output(self, shell_id: str, offset: int = 0) -> tuple[bytes, int | None]:
        bp = self._procs.get(shell_id)
        if bp is None:
            return b"", None
        if offset < 0:
            offset = 0
        tail = bytes(bp.buffer[offset:]) if offset < len(bp.buffer) else b""
        return tail, bp.exit_code

    async def kill(self, shell_id: str, term_wait: float = 2.0) -> bool:
        bp = self._procs.get(shell_id)
        if bp is None or bp.exit_code is not None:
            return False
        # Send SIGTERM to the entire process group so children also receive it.
        _kill_pgrp(bp.proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(bp.proc.wait(), timeout=term_wait)
        except TimeoutError:
            # Escalate: SIGKILL the process group to guarantee termination and
            # close the stdout pipe (so the reader task gets EOF).
            _kill_pgrp(bp.proc.pid, signal.SIGKILL)
            try:
                await asyncio.wait_for(bp.proc.wait(), timeout=term_wait)
            except TimeoutError:
                pass
        if bp.reader_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(bp.reader_task), timeout=term_wait)
            except (TimeoutError, asyncio.CancelledError):
                pass
        return True

    async def wait(self, shell_id: str, timeout: float) -> bool:
        """Await the reader task (which awaits proc.wait) up to timeout.

        Returns True if the process has exited before timeout, False otherwise.
        """
        bp = self._procs.get(shell_id)
        if bp is None or bp.reader_task is None:
            return False
        try:
            await asyncio.wait_for(asyncio.shield(bp.reader_task), timeout=timeout)
            return bp.exit_code is not None
        except TimeoutError:
            return False

    def list(self) -> list[BgProcess]:
        return list(self._procs.values())

    def gc_once(self) -> None:
        """Drop exited procs older than gc_seconds."""
        now = time.monotonic()
        dead: list[str] = []
        for shell_id, bp in self._procs.items():
            if bp.exited_at is not None and (now - bp.exited_at) >= self._gc_seconds:
                dead.append(shell_id)
        for shell_id in dead:
            self._procs.pop(shell_id, None)

    async def shutdown(self) -> None:
        """SIGTERM process groups, escalate to SIGKILL, await cleanup."""
        for shell_id in list(self._procs.keys()):
            bp = self._procs[shell_id]
            if bp.exit_code is None:
                _kill_pgrp(bp.proc.pid, signal.SIGTERM)
        for shell_id in list(self._procs.keys()):
            bp = self._procs[shell_id]
            if bp.reader_task is None or bp.reader_task.done():
                continue
            # Wait for reader to finish (reader sets exit_code on completion).
            # Use shield so a timeout here does not cancel the reader task itself.
            try:
                await asyncio.wait_for(asyncio.shield(bp.reader_task), timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                # SIGTERM didn't work — kill the whole process group to close the pipe.
                _kill_pgrp(bp.proc.pid, signal.SIGKILL)
                try:
                    await asyncio.wait_for(bp.proc.wait(), timeout=2.0)
                except TimeoutError:
                    pass
                # Now the pipe is closed; the reader can drain EOF and set exit_code.
                try:
                    await asyncio.wait_for(asyncio.shield(bp.reader_task), timeout=2.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass
