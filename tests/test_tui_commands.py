"""Slash command validation keeps drafts and supports literal slash-prefixed prompts."""

import asyncio
from unittest.mock import Mock

import pytest

from drove.tui import ChatApp, ChatInput, Message


@pytest.mark.parametrize(
    "draft",
    [
        "/tmp/example",
        "/new implementation plan",
        "/clear extra",
        "/save extra",
        "/sessions extra",
        "/model other",
        "/help extra",
        "/theme two names",
        "/theme does-not-exist",
        "/tips invalid",
    ],
)
async def test_invalid_command_keeps_draft_and_current_conversation(tmp_path, monkeypatch, draft):
    app = ChatApp("http://localhost:8080", "m", tmp_path, None)
    previous = app._session
    history = [{"role": "user", "content": "keep this conversation"}]
    app._history = list(history)
    start = Mock()
    monkeypatch.setattr(app, "_start_turn", start)
    async with app.run_test() as pilot:
        field = app.query_one(ChatInput)
        field.load_text(draft)
        await pilot.press("enter")
        await pilot.pause()
        assert field.text == draft
        assert app._session is previous
        assert app._history == history
        assert app.query(Message).last()._role == "system-note"
        start.assert_not_called()


@pytest.mark.parametrize(
    "draft,expected",
    [
        ("//tmp/example", "/tmp/example"),
        ("//new plan", "/new plan"),
        ("///tmp", "//tmp"),
        ("//", "/"),
        ("//path\nsecond line", "/path\nsecond line"),
    ],
)
async def test_escaped_slash_is_sent_as_an_ordinary_message(tmp_path, monkeypatch, draft, expected):
    app = ChatApp("http://localhost:8080", "m", tmp_path, None)
    requests = []

    async def stream(messages):
        requests.append([dict(message) for message in messages])
        yield {"kind": "content", "chunk": "reply"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        original = app._session
        field = app.query_one(ChatInput)
        field.load_text(draft)
        await pilot.press("enter")
        async with asyncio.timeout(3):
            while not requests or app.is_generating:
                await pilot.pause()
        assert field.text == ""
        assert app._session is original
        assert requests[0][-1] == {"role": "user", "content": expected}
        assert app._history[-1]["content"] == "reply"


async def test_escaped_slash_joins_the_followup_queue(tmp_path):
    app = ChatApp("http://localhost:8080", "m", tmp_path, None)
    async with app.run_test() as pilot:
        app.is_generating = True
        app.query_one(ChatInput).load_text("//new plan")
        await pilot.press("enter")
        await pilot.pause()
        assert [prompt.text for prompt in app._queue] == ["/new plan"]
        app.is_generating = False


async def test_direct_command_dispatch_also_rejects_unexpected_arguments(tmp_path):
    app = ChatApp("http://localhost:8080", "m", tmp_path, None)
    async with app.run_test():
        original = app._session
        await app._dispatch_command("/new plan")
        assert app._session is original
        assert "takes no arguments" in app.query(Message).last()._content
