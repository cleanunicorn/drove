"""Tests for chat session persistence."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from drove.sessions import (
    Session,
    latest_session,
    list_sessions,
    load_session,
    new_session,
    save_session,
)


class _Clock:
    """Stand-in for ``datetime`` in drove.sessions returning scripted instants."""

    times: list[datetime] = []

    @classmethod
    def now(cls) -> datetime:
        return cls.times.pop(0) if len(cls.times) > 1 else cls.times[0]


def frozen(*times: datetime):
    _Clock.times = list(times)
    return patch("drove.sessions.datetime", _Clock)


def test_valid_model_keeps_its_session_layout(tmp_path: Path) -> None:
    session = new_session("org/repo:Q4_K_M")
    save_session(tmp_path, session)
    assert (tmp_path / "org" / "repo:Q4_K_M" / f"{session.id}.json").is_file()
    assert latest_session(tmp_path, "org/repo:Q4_K_M").id == session.id


@pytest.mark.parametrize(
    "model", ["../outside", "../../x", "/abs/dir", "a/b/c", "remote model", "", "..", "a//b"]
)
def test_unsafe_model_labels_stay_inside_sessions_dir(tmp_path: Path, model: str) -> None:
    """Remote endpoints may name models anything; sessions must not leave sessions_dir."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session = new_session(model)
    save_session(sessions, session)
    written = [p for p in tmp_path.rglob("*.json")]
    assert len(written) == 1 and written[0].is_relative_to(sessions)
    assert [s.id for s in list_sessions(sessions, model)] == [session.id]
    assert latest_session(sessions, model).model == model


@pytest.mark.parametrize("model", ["/tmp/x", "../x", "..\\x", "org/repo:Q4"])
def test_issue_119_model_labels_stay_inside_sessions_dir(tmp_path: Path, model: str) -> None:
    """Containment comes from 48b8aee (PR #142); these are the inputs issue #119 names."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    save_session(sessions, new_session(model))
    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1 and written[0].resolve().is_relative_to(sessions.resolve())
    assert latest_session(sessions, model).model == model


def test_distinct_unsafe_labels_do_not_share_a_directory(tmp_path: Path) -> None:
    save_session(tmp_path, new_session("a/b/c"))
    assert list_sessions(tmp_path, "a/b c") == []


def test_outward_symlink_is_not_followed(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (sessions / "linked").symlink_to(outside, target_is_directory=True)
    save_session(sessions, new_session("linked"))
    assert list(outside.iterdir()) == []
    assert len(list_sessions(sessions, "linked")) == 1


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    session = new_session("m", "be brief ✓")
    session.messages += [
        {"role": "user", "content": "héllo\nwörld"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    save_session(tmp_path, session)
    loaded = load_session(tmp_path / "m" / f"{session.id}.json")
    assert loaded.to_dict() == session.to_dict()
    assert loaded.title == "héllo wörld" and loaded.message_count == 1
    session.messages.append({"role": "user", "content": "again"})
    save_session(tmp_path, session)
    assert latest_session(tmp_path, "m").messages == session.messages


# --- atomic saves -----------------------------------------------------------


def _saved(tmp_path: Path) -> tuple[Session, Path, bytes]:
    session = new_session("m")
    session.messages.append({"role": "user", "content": "keep me"})
    save_session(tmp_path, session)
    path = tmp_path / "m" / f"{session.id}.json"
    return session, path, path.read_bytes()


def _torn_fdopen(real=os.fdopen):
    """``os.fdopen`` whose file writes a prefix, then fails like a full disk."""

    class Torn:
        def __init__(self, *args, **kwargs) -> None:
            self.file = real(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            self.file.close()
            return False

        def write(self, text: str) -> int:
            self.file.write(text[:10])
            self.file.flush()
            raise OSError("disk full")

    return Torn


def test_failed_write_keeps_previous_file_intact(tmp_path: Path) -> None:
    session, path, before = _saved(tmp_path)
    session.messages.append({"role": "user", "content": "lost"})
    with patch("os.fdopen", _torn_fdopen()), pytest.raises(OSError):
        save_session(tmp_path, session)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_failed_replace_keeps_previous_file_intact(tmp_path: Path) -> None:
    session, path, before = _saved(tmp_path)
    session.messages.append({"role": "user", "content": "lost"})
    with patch("os.replace", side_effect=OSError("boom")), pytest.raises(OSError):
        save_session(tmp_path, session)
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_failed_first_save_leaves_no_file(tmp_path: Path) -> None:
    with patch("os.fdopen", _torn_fdopen()), pytest.raises(OSError):
        save_session(tmp_path, new_session("m"))
    assert list((tmp_path / "m").iterdir()) == []


# --- unreadable files are skipped --------------------------------------------

BAD_FILES: dict[str, bytes] = {
    "invalid-json": b'{"id": "x", "model": "m", "messages": [',
    "empty": b"",
    "not-utf8": b"\xff\xfe\x00",
    "not-object": b"[]",
    "missing-messages": json.dumps({"id": "x", "model": "m"}).encode(),
    "missing-id": json.dumps({"model": "m", "messages": []}).encode(),
    "messages-not-list": json.dumps({"id": "x", "model": "m", "messages": "hi"}).encode(),
    "message-not-object": json.dumps({"id": "x", "model": "m", "messages": ["hi"]}).encode(),
    "message-no-role": json.dumps(
        {"id": "x", "model": "m", "messages": [{"content": "hi"}]}
    ).encode(),
    "user-content-not-text": json.dumps(
        {"id": "x", "model": "m", "messages": [{"role": "user", "content": 5}]}
    ).encode(),
    "updated-at-not-text": json.dumps(
        {"id": "x", "model": "m", "messages": [], "updated_at": 5}
    ).encode(),
    "traversal-id": json.dumps({"id": "../../evil", "model": "m", "messages": []}).encode(),
}
# Message shapes chat replay reads (drove.tui.ChatApp._replay_history).
BAD_MESSAGES: dict[str, dict[str, object]] = {
    "tool-without-call-id": {"role": "tool", "content": "result"},
    "tool-content-not-text": {"role": "tool", "tool_call_id": "1", "content": 5},
    "assistant-content-not-text": {"role": "assistant", "content": ["x"]},
    "thinking-not-text": {"role": "assistant", "content": "", "thinking": 5},
    "tool-calls-not-list": {"role": "assistant", "content": "", "tool_calls": {"id": "1"}},
    "tool-call-not-object": {"role": "assistant", "content": "", "tool_calls": ["x"]},
    "tool-call-id-not-text": {"role": "assistant", "content": "", "tool_calls": [{"id": 1}]},
    "tool-call-function-not-object": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": "f"}],
    },
    "tool-call-arguments-null": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": None}}],
    },
    "tool-call-name-null": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": None, "arguments": "{}"}}],
    },
    "tool-call-name-not-text": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": 1, "arguments": "{}"}}],
    },
}
for _kind, _message in BAD_MESSAGES.items():
    BAD_FILES[_kind] = json.dumps(
        {"id": "x", "model": "m", "messages": [{"role": "user", "content": "hi"}, _message]}
    ).encode()


def _write_bad(directory: Path, name: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("kind", sorted(BAD_FILES))
@pytest.mark.parametrize("position", ["newest", "middle", "oldest"])
def test_bad_session_file_is_skipped(tmp_path: Path, kind: str, position: str) -> None:
    with frozen(datetime(2026, 1, 1, 10)):
        older = new_session("m")
    with frozen(datetime(2026, 1, 2, 10)):
        newer = new_session("m")
    save_session(tmp_path, older)
    save_session(tmp_path, newer)
    name = {"newest": "20990101-000000", "middle": "20260101-120000", "oldest": "19990101"}
    bad = _write_bad(tmp_path / "m", f"{name[position]}.json", BAD_FILES[kind])

    warnings: list[str] = []
    assert [s.id for s in list_sessions(tmp_path, "m", warn=warnings.append)] == [
        newer.id,
        older.id,
    ]
    assert len(warnings) == 1 and str(bad) in warnings[0]
    assert latest_session(tmp_path, "m", warn=warnings.append).id == newer.id


def test_unreadable_entry_is_skipped(tmp_path: Path) -> None:
    session = new_session("m")
    save_session(tmp_path, session)
    (tmp_path / "m" / "20990101-000000.json").mkdir()  # reading it raises OSError
    warnings: list[str] = []
    assert latest_session(tmp_path, "m", warn=warnings.append).id == session.id
    assert len(warnings) == 1


def test_latest_session_stops_at_first_readable_file(tmp_path: Path) -> None:
    session = new_session("m")
    save_session(tmp_path, session)
    _write_bad(tmp_path / "m", "19990101.json", b"{")
    warnings: list[str] = []
    assert latest_session(tmp_path, "m", warn=warnings.append).id == session.id
    assert warnings == []


def test_all_bad_or_missing_directory_yields_nothing(tmp_path: Path) -> None:
    assert list_sessions(tmp_path, "m") == [] and latest_session(tmp_path, "m") is None
    _write_bad(tmp_path / "m", "a.json", b"{")
    _write_bad(tmp_path / "m", "b.json", b"[]")
    warnings: list[str] = []
    assert list_sessions(tmp_path, "m", warn=warnings.append) == []
    assert latest_session(tmp_path, "m", warn=warnings.append) is None
    assert len(warnings) == 4


def test_skipped_file_warns_through_module_logger_by_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = _write_bad(tmp_path / "m", "a.json", b"{")
    with caplog.at_level(logging.WARNING, logger="drove.sessions"):
        assert list_sessions(tmp_path, "m") == []
    assert [r.name for r in caplog.records] == ["drove.sessions"]
    assert str(bad) in caplog.records[0].getMessage()


def test_direct_load_of_bad_file_raises(tmp_path: Path) -> None:
    bad = _write_bad(tmp_path, "a.json", BAD_FILES["missing-messages"])
    with pytest.raises(ValueError):
        load_session(bad)


def test_replayable_message_shapes_still_load(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "thinking": "hm", "stopped": True},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "2"},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
        {"role": "tool", "tool_call_id": "2"},
    ]
    path = _write_bad(
        tmp_path, "a.json", json.dumps({"id": "a", "model": "m", "messages": messages}).encode()
    )
    assert load_session(path).messages == messages


def test_legacy_session_ids_still_load(tmp_path: Path) -> None:
    for session_id in ("20260101-120000", "20260101-120001-1767265201000000000-" + "a" * 32):
        _write_bad(
            tmp_path / "m",
            f"{session_id}.json",
            json.dumps({"id": session_id, "model": "m", "messages": []}).encode(),
        )
    assert len(list_sessions(tmp_path, "m", warn=pytest.fail)) == 2


# --- session ids --------------------------------------------------------------


def test_new_sessions_in_the_same_instant_get_distinct_files(tmp_path: Path) -> None:
    """Two windows starting a chat at once must not share a file."""
    with frozen(datetime(2026, 1, 1, 12, 0, 0, 5)):
        first, second = new_session("m"), new_session("m")
    assert first.id != second.id
    save_session(tmp_path, first)
    save_session(tmp_path, second)
    assert {s.id for s in list_sessions(tmp_path, "m")} == {first.id, second.id}


def test_session_ids_sort_chronologically_within_a_second(tmp_path: Path) -> None:
    with frozen(datetime(2026, 1, 1, 12, 0, 0, 999), datetime(2026, 1, 1, 12, 0, 0, 1000)):
        first = new_session("m")
        second = new_session("m")
    save_session(tmp_path, second)
    save_session(tmp_path, first)
    assert first.id < second.id
    assert latest_session(tmp_path, "m").id == second.id


@pytest.mark.parametrize("session_id", ["../../evil", "a/b", "..\\evil", "..", "", ".hidden"])
def test_unsafe_session_id_is_never_written(tmp_path: Path, session_id: str) -> None:
    sessions = tmp_path / "sessions"
    with pytest.raises(ValueError):
        save_session(sessions, Session("m", session_id, []))
    assert list(tmp_path.rglob("*.json")) == []


def test_outward_session_file_symlink_is_not_written_through(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session = new_session("m")
    (sessions / "m").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("untouched")
    (sessions / "m" / f"{session.id}.json").symlink_to(outside)
    with pytest.raises(ValueError):
        save_session(sessions, session)
    assert outside.read_text() == "untouched"
