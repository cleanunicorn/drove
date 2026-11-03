"""Tests for the per-model download record (read_state fails closed, write_state round-trips)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from drove.download_state import (
    MANIFEST_NAME,
    UNREADABLE,
    DownloadState,
    read_state,
    write_state,
)


def _write(root: Path, data: object) -> None:
    (root / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")


def _record(**overrides: object) -> dict[str, object]:
    return {"version": 1, "state": "complete", "files": {"m.gguf": 4}, **overrides}


def test_no_record_is_none(tmp_path: Path) -> None:
    assert read_state(tmp_path) is None


def test_write_then_read_round_trips_with_sorted_files(tmp_path: Path) -> None:
    write_state(tmp_path, "incomplete", {"b.gguf": 2, "a/x.gguf": 0})

    assert read_state(tmp_path) == DownloadState("incomplete", {"a/x.gguf": 0, "b.gguf": 2})
    assert list(json.loads((tmp_path / MANIFEST_NAME).read_text())["files"]) == [
        "a/x.gguf",
        "b.gguf",
    ]
    assert [p.name for p in tmp_path.iterdir()] == [MANIFEST_NAME]


def test_rewrite_replaces_the_previous_record(tmp_path: Path) -> None:
    write_state(tmp_path, "incomplete", {"m.gguf": 4})
    write_state(tmp_path, "complete", {"m.gguf": 4})
    state = read_state(tmp_path)
    assert state is not None and state.complete


def test_symlinked_record_is_unreadable(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps(_record()))
    (tmp_path / MANIFEST_NAME).symlink_to(real)
    assert read_state(tmp_path) is UNREADABLE


def test_dangling_symlink_record_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).symlink_to(tmp_path / "missing.json")
    assert read_state(tmp_path) is UNREADABLE


def test_directory_in_place_of_record_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).mkdir()
    assert read_state(tmp_path) is UNREADABLE


def test_undecodable_record_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_bytes(b"\xff\xfe{}")
    assert read_state(tmp_path) is UNREADABLE


@pytest.mark.parametrize(
    "data",
    [
        [],
        _record(version=2),
        _record(state="done"),
        _record(files=["m.gguf"]),
        _record(files={"/abs.gguf": 1}),
        _record(files={"..\\up.gguf": 1}),
        _record(files={"a/../b.gguf": 1}),
        _record(files={"": 1}),
        _record(files={"m.gguf": -1}),
        _record(files={"m.gguf": True}),
        _record(files={"m.gguf": "4"}),
    ],
    ids=[
        "not-object",
        "version",
        "state",
        "files-list",
        "absolute",
        "backslash",
        "dotdot",
        "empty",
        "negative",
        "bool",
        "string",
    ],
)
def test_invalid_record_fails_closed(tmp_path: Path, data: object) -> None:
    _write(tmp_path, data)
    state = read_state(tmp_path)
    assert state is UNREADABLE
    assert not state.complete


# --- write_state goes through the shared atomic writer (#156) -----------------


def test_record_bytes_are_unchanged(tmp_path: Path) -> None:
    write_state(tmp_path, "complete", {"b.gguf": 2, "a.gguf": 1})
    assert (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8") == (
        '{\n  "version": 1,\n  "state": "complete",\n'
        '  "files": {\n    "a.gguf": 1,\n    "b.gguf": 2\n  }\n}\n'
    )


def test_record_keeps_the_umask_default_mode(tmp_path: Path) -> None:
    previous = os.umask(0o027)
    try:
        write_state(tmp_path, "incomplete", {"m.gguf": 4})
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / MANIFEST_NAME).stat().st_mode) == 0o640


def test_failed_record_write_leaves_no_temp_and_keeps_the_old_record(tmp_path: Path) -> None:
    write_state(tmp_path, "incomplete", {"m.gguf": 4})
    before = (tmp_path / MANIFEST_NAME).read_bytes()
    with patch("drove.atomic_write.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            write_state(tmp_path, "complete", {"m.gguf": 4})
    assert (tmp_path / MANIFEST_NAME).read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [MANIFEST_NAME]


def test_planted_link_at_the_old_temp_name_is_not_written_through(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / f"{MANIFEST_NAME}.tmp").symlink_to(victim)
    write_state(tmp_path, "complete", {"m.gguf": 4})
    assert victim.read_text() == "keep me"
    state = read_state(tmp_path)
    assert state is not None and state.complete


def test_planted_link_at_the_record_is_replaced_not_written_through(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / MANIFEST_NAME).symlink_to(victim)
    write_state(tmp_path, "complete", {"m.gguf": 4})
    assert victim.read_text() == "keep me"
    assert not (tmp_path / MANIFEST_NAME).is_symlink()
    state = read_state(tmp_path)
    assert state is not None and state.complete
