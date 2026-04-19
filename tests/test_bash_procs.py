"""Tests for BgProcs background-process manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.bash_procs import BgProcs


@pytest.fixture
def procs() -> BgProcs:
    return BgProcs(buffer_bytes=1024, gc_seconds=600)


async def test_start_returns_shell_id_and_pid(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("echo hello", cwd=tmp_path)
    assert isinstance(shell_id, str) and shell_id
    proc = procs.get(shell_id)
    assert proc is not None
    assert proc.pid > 0


async def test_output_captures_stdout(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("echo hello && echo world", cwd=tmp_path)
    await procs.wait(shell_id, timeout=2.0)
    data, exit_code = await procs.output(shell_id, offset=0)
    text = data.decode("utf-8", errors="replace")
    assert "hello" in text
    assert "world" in text
    assert exit_code == 0


async def test_output_merges_stderr(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("echo out; echo err 1>&2", cwd=tmp_path)
    await procs.wait(shell_id, timeout=2.0)
    data, _ = await procs.output(shell_id, offset=0)
    text = data.decode("utf-8", errors="replace")
    assert "out" in text
    assert "err" in text


async def test_output_offset_returns_tail(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("printf AAAA; printf BBBB", cwd=tmp_path)
    await procs.wait(shell_id, timeout=2.0)
    full, _ = await procs.output(shell_id, offset=0)
    tail, _ = await procs.output(shell_id, offset=4)
    assert full[:4] == b"AAAA"
    assert tail == full[4:]


async def test_buffer_capped(procs: BgProcs, tmp_path: Path) -> None:
    cmd = "for i in $(seq 1 200); do printf '0123456789\\n'; done"
    shell_id = await procs.start(cmd, cwd=tmp_path)
    await procs.wait(shell_id, timeout=5.0)
    data, _ = await procs.output(shell_id, offset=0)
    assert len(data) <= 1024


async def test_kill_terminates_process(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("sleep 30", cwd=tmp_path)
    ok = await procs.kill(shell_id)
    assert ok is True
    proc = procs.get(shell_id)
    assert proc is not None
    assert proc.exit_code is not None


async def test_kill_already_dead_returns_false(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("true", cwd=tmp_path)
    await procs.wait(shell_id, timeout=2.0)
    ok = await procs.kill(shell_id)
    assert ok is False


async def test_sigkill_escalation(procs: BgProcs, tmp_path: Path) -> None:
    cmd = "trap '' TERM; sleep 30"
    shell_id = await procs.start(cmd, cwd=tmp_path)
    ok = await procs.kill(shell_id, term_wait=0.5)
    assert ok is True


async def test_list_returns_snapshot(procs: BgProcs, tmp_path: Path) -> None:
    a = await procs.start("echo a", cwd=tmp_path)
    b = await procs.start("echo b", cwd=tmp_path)
    snap = procs.list()
    ids = {p.shell_id for p in snap}
    assert a in ids and b in ids


async def test_shutdown_terminates_all(procs: BgProcs, tmp_path: Path) -> None:
    a = await procs.start("sleep 30", cwd=tmp_path)
    b = await procs.start("sleep 30", cwd=tmp_path)
    await procs.shutdown()
    assert procs.get(a).exit_code is not None  # type: ignore[union-attr]
    assert procs.get(b).exit_code is not None  # type: ignore[union-attr]


async def test_gc_removes_stale_exited_procs(tmp_path: Path) -> None:
    procs = BgProcs(buffer_bytes=1024, gc_seconds=0)
    shell_id = await procs.start("echo bye", cwd=tmp_path)
    await procs.wait(shell_id, timeout=2.0)
    procs.gc_once()
    assert procs.get(shell_id) is None


async def test_unknown_shell_id_returns_none(procs: BgProcs) -> None:
    assert procs.get("nope") is None
    data, code = await procs.output("nope", offset=0)
    assert data == b""
    assert code is None


async def test_wait_timeout_returns_false(procs: BgProcs, tmp_path: Path) -> None:
    shell_id = await procs.start("sleep 5", cwd=tmp_path)
    finished = await procs.wait(shell_id, timeout=0.1)
    assert finished is False
    await procs.kill(shell_id)
