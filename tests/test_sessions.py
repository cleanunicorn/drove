"""Tests for chat session persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from drove.sessions import latest_session, list_sessions, new_session, save_session


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


def test_saved_tool_results_are_capped_and_other_messages_kept(tmp_path: Path) -> None:
    import json

    from drove.tool_policy import TOOL_OUTPUT_LIMIT, TRUNCATION_NOTICE

    huge = "€" * TOOL_OUTPUT_LIMIT
    session = new_session("test-model")
    session.messages = [
        {"role": "user", "content": huge},
        {"role": "tool", "tool_call_id": "c1", "content": huge},
    ]

    save_session(tmp_path, session)

    [path] = (tmp_path / "test-model").glob("*.json")
    user, tool = json.loads(path.read_text())["messages"]
    assert user["content"] == huge
    assert len(tool["content"].encode("utf-8")) <= TOOL_OUTPUT_LIMIT
    assert tool["content"].endswith(TRUNCATION_NOTICE)
    assert session.messages[1]["content"] == huge  # the live history is not rewritten
