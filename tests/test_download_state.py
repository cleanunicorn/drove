"""Tests for the per-model download record (read_state fails closed, write_state round-trips)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drove.download_state import (
    MANIFEST_NAME,
    MANIFEST_TEMP_NAME,
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
    assert not (tmp_path / MANIFEST_TEMP_NAME).exists()


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
