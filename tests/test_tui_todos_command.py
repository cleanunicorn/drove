"""Tests for /todos slash command helper."""

from __future__ import annotations

from vllama.tui import render_todos_summary


def test_render_todos_summary_empty() -> None:
    text = render_todos_summary([])
    assert "no todos" in text.lower()


def test_render_todos_summary_mixed_statuses() -> None:
    todos = [
        {"id": "1", "content": "write code", "status": "completed"},
        {"id": "2", "content": "write tests", "status": "in_progress"},
        {"id": "3", "content": "ship", "status": "pending"},
    ]
    text = render_todos_summary(todos)
    assert "[x]" in text
    assert "[~]" in text
    assert "[ ]" in text
    assert "write code" in text
    assert "write tests" in text
    assert "ship" in text


def test_render_todos_summary_unknown_status() -> None:
    text = render_todos_summary([{"id": "1", "content": "x", "status": "mystery"}])
    assert "[?]" in text
