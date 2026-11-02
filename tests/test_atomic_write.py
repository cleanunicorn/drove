"""Tests for the shared atomic writer: exclusive temp file, rename, cleanup, modes."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from drove.atomic_write import stage_text, write_text_atomic


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def umask_022() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def _tokens(*values: str) -> Iterator[str]:
    yield from values


def test_writes_text_and_leaves_only_the_target(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    write_text_atomic(target, "héllo\n")
    assert target.read_bytes() == "héllo\n".encode()
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    write_text_atomic(target, "new")
    assert target.read_text() == "new"


def test_default_mode_is_private(tmp_path: Path, umask_022: None) -> None:
    target = tmp_path / "f.json"
    write_text_atomic(target, "x")
    assert _mode(target) == 0o600


def test_explicit_mode_is_filtered_by_umask(tmp_path: Path, umask_022: None) -> None:
    target = tmp_path / "f.json"
    write_text_atomic(target, "x", mode=0o666)
    assert _mode(target) == 0o644


def test_keep_mode_copies_the_existing_target_mode(tmp_path: Path, umask_022: None) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    target.chmod(0o640)
    write_text_atomic(target, "new", keep_mode=True)
    assert _mode(target) == 0o640


def test_keep_mode_without_a_target_stays_private(tmp_path: Path, umask_022: None) -> None:
    target = tmp_path / "f.json"
    write_text_atomic(target, "x", keep_mode=True)
    assert _mode(target) == 0o600


def test_stage_names_a_hidden_sibling_and_leaves_the_target(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    tmp = stage_text(target, "new")
    assert tmp.parent == tmp_path
    assert tmp.name.startswith(".f.json.") and tmp.name.endswith(".tmp")
    assert tmp.read_text() == "new"
    assert target.read_text() == "old"


def test_failed_write_leaves_no_temp_and_keeps_the_target(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    with patch("drove.atomic_write.os.fdopen", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            write_text_atomic(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_failed_chmod_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    with patch("drove.atomic_write.os.chmod", side_effect=PermissionError("no")):
        with pytest.raises(PermissionError):
            stage_text(target, "new", keep_mode=True)
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_failed_replace_leaves_no_temp_and_keeps_the_target(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_text("old")
    with patch("drove.atomic_write.os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            write_text_atomic(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_interrupt_during_write_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    with patch("drove.atomic_write.os.fdopen", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            write_text_atomic(target, "new")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dangling", [False, True])
def test_planted_link_at_the_temp_name_is_skipped_not_followed(
    tmp_path: Path, dangling: bool
) -> None:
    victim = tmp_path / "victim.txt"
    if not dangling:
        victim.write_text("keep me")
    planted = tmp_path / ".f.json.aaaaaaaaaaaa.tmp"
    planted.symlink_to(victim)
    target = tmp_path / "f.json"
    tokens = _tokens("aaaaaaaaaaaa", "bbbbbbbbbbbb")
    with patch("drove.atomic_write.secrets.token_hex", lambda _n: next(tokens)):
        write_text_atomic(target, "new")
    assert next(tokens, None) is None  # the planted name was tried first
    assert target.read_text() == "new"
    assert planted.is_symlink()
    if dangling:
        assert not victim.exists()
    else:
        assert victim.read_text() == "keep me"


def test_gives_up_when_every_temp_name_is_taken(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / ".f.json.aaaaaaaaaaaa.tmp").symlink_to(victim)
    with patch("drove.atomic_write.secrets.token_hex", lambda _n: "aaaaaaaaaaaa"):
        with pytest.raises(FileExistsError):
            write_text_atomic(tmp_path / "f.json", "new")
    assert victim.read_text() == "keep me"
    assert not (tmp_path / "f.json").exists()


def test_symlink_at_the_target_is_replaced_not_written_through(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    target = tmp_path / "f.json"
    target.symlink_to(victim)
    write_text_atomic(target, "new")
    assert not target.is_symlink()
    assert target.read_text() == "new"
    assert victim.read_text() == "keep me"


def test_missing_directory_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_text_atomic(tmp_path / "missing" / "f.json", "x")
    assert not (tmp_path / "missing").exists()
