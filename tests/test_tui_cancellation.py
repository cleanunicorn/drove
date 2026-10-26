import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from textual.widgets import Button, Label

from drove.sessions import load_session, new_session
from drove.tui import (
    ChatApp,
    ChatInput,
    Message,
    QueuedMessage,
    SessionPicker,
    ToolConfirmationModal,
)


def chat_app(tmp_path: Path) -> ChatApp:
    return ChatApp("http://localhost:8080", "test-model", tmp_path, None)


async def test_escape_stops_stream_and_saves_partial(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    streamed = asyncio.Event()
    closed = asyncio.Event()

    async def stream(messages):
        try:
            yield {"kind": "content", "chunk": "partial reply"}
            streamed.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        app.query_one(ChatInput).insert("hello")
        await pilot.press("enter")
        await asyncio.wait_for(streamed.wait(), 2)
        await pilot.press("escape")
        await asyncio.wait_for(closed.wait(), 2)
        await pilot.pause()
        assert app.is_running
        assert not app.is_generating
        assert app._history[-1] == {
            "role": "assistant",
            "content": "partial reply",
            "stopped": True,
        }
        saved = json.loads(next(tmp_path.rglob("*.json")).read_text())
        assert saved["messages"] == app._history
        assert isinstance(app.focused, ChatInput)


async def test_idle_escape_does_not_quit(tmp_path):
    app = chat_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        assert app.is_running


async def wait_idle(app, pilot):
    async with asyncio.timeout(3):
        while app.is_generating:
            await pilot.pause()


async def submit(app, pilot, text):
    app.query_one(ChatInput).focus()
    app.query_one(ChatInput).insert(text)
    await pilot.press("enter")


@pytest.mark.parametrize("kind", ["empty", "thinking", "tool_call"])
async def test_stop_retains_draft_and_discards_incomplete_tools(tmp_path, monkeypatch, kind):
    app = chat_app(tmp_path)
    ready = asyncio.Event()
    closed = asyncio.Event()
    executed = []

    async def stream(messages):
        try:
            if kind == "thinking":
                yield {"kind": "thinking", "chunk": "partial thought"}
            elif kind == "tool_call":
                yield {
                    "kind": "tool_call",
                    "index": 0,
                    "id": "c1",
                    "name": "read_file",
                    "arguments": "{",
                }
            ready.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: executed.append(args))
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.press("escape", "escape")
        await wait_idle(app, pilot)
        assert closed.is_set()
        record = app._history[-1]
        assert record["stopped"] is True
        assert "tool_calls" not in record
        assert record["content"] == ""
        if kind == "thinking":
            assert record["thinking"] == "partial thought"
        assert not executed
        bubble = app.query(Message).last()
        assert bubble._stopped
        assert "Stopped" in str(bubble.query_one(".message-header", Label).render())


async def test_stopped_reply_replays_and_next_reply_succeeds(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()
    requests = []

    async def stream(messages):
        requests.append(list(messages))
        yield {"kind": "content", "chunk": "first" if len(requests) == 1 else "second"}
        if len(requests) == 1:
            ready.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "one")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.press("escape")
        await wait_idle(app, pilot)
        await submit(app, pilot, "two")
        await wait_idle(app, pilot)
        assert [m["content"] for m in app._history] == ["one", "first", "two", "second"]
        saved = load_session(next(tmp_path.rglob("*.json")))
    resumed = ChatApp("http://localhost", "test-model", tmp_path, None, resume_session=saved)
    async with resumed.run_test() as pilot:
        await pilot.pause()
        assistants = [m for m in resumed.query(Message) if m._role == "assistant"]
        assert [m._stopped for m in assistants] == [True, False]
        assert [m._content for m in assistants] == ["first", "second"]


async def test_real_http_stream_closes_and_omits_local_metadata(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    app._history.append({"role": "assistant", "content": "old", "stopped": True})
    ready, closed = asyncio.Event(), asyncio.Event()
    payloads = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            ready.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    def response(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, stream=Body())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "drove.tui.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(response), **kwargs),
    )
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert closed.is_set()
        assert payloads[0]["messages"][0] == {"role": "assistant", "content": "old"}
        assert all("stopped" not in msg for msg in payloads[0]["messages"])


@pytest.mark.parametrize("key", ["ctrl+c", "ctrl+q"])
async def test_explicit_quit_keys(tmp_path, key):
    app = chat_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(key)
        assert not app.is_running


async def test_escape_autocomplete_and_session_picker(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()

    async def stream(messages):
        ready.set()
        await asyncio.Event().wait()
        yield {}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        app.query_one(ChatInput).insert("/th")
        await pilot.pause()
        await pilot.press("escape")
        assert app.is_running
        assert not app.query_one("#autocomplete").has_class("visible")
        app.query_one(ChatInput).clear()
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        app.query_one(ChatInput).insert("/th")
        await pilot.pause()
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert app._history[-1]["stopped"]
        app.push_screen(SessionPicker([]))
        await pilot.pause()
        await pilot.press("escape")
        assert app.is_running
        assert not isinstance(app.screen, SessionPicker)


@pytest.mark.parametrize(
    "key,choice",
    [
        ("y", "allow-once"),
        ("n", "deny-once"),
        ("escape", "deny-once"),
        ("s", "allow-session"),
        ("a", "allow-always"),
    ],
)
async def test_modal_keyboard_choices(tmp_path, key, choice):
    app = chat_app(tmp_path)
    choices = []
    async with app.run_test() as pilot:
        app.push_screen(
            ToolConfirmationModal("shell_execute", "plain arguments"), callback=choices.append
        )
        await pilot.pause()
        assert len(app.screen.query(Button)) == 4
        await pilot.press(key)
        assert choices == [choice]
        assert app.is_running


async def test_dialog_escape_denies_without_stopping_turn(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    requests, executed = [], []
    monkeypatch.setattr("drove.config.load_config", lambda _: SimpleNamespace(allowed_tools=[]))
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: executed.append(args))

    async def stream(messages):
        requests.append(list(messages))
        if len(requests) == 1:
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "c1",
                "name": "shell_execute",
                "arguments": "{}",
            }
        else:
            yield {"kind": "content", "chunk": "denied"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "tool")
        async with asyncio.timeout(2):
            while not isinstance(app.screen, ToolConfirmationModal):
                await pilot.pause()
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert not executed
        assert len(requests) == 2
        assert requests[1][-1]["content"] == "Error: Tool execution denied by user."
        assert all(not msg.get("stopped") for msg in app._history)


async def test_cancelled_modal_wait_dismisses_owned_screen(tmp_path):
    app = chat_app(tmp_path)
    async with app.run_test() as pilot:
        waiter = asyncio.create_task(app.push_screen_wait(ToolConfirmationModal("tool", "args")))
        await pilot.pause()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await pilot.pause()
        assert not isinstance(app.screen, ToolConfirmationModal)
        assert app.is_running


async def test_stop_waits_for_running_tool_and_pairs_remaining_calls(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    executed, requests = [], []

    def execute(name, args, on_output=None):
        executed.append(name)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return "real tool result"

    async def stream(messages):
        requests.append(list(messages))
        for idx in range(2):
            yield {
                "kind": "tool_call",
                "index": idx,
                "id": f"c{idx}",
                "name": "read_file",
                "arguments": "{}",
            }

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    try:
        async with app.run_test() as pilot:
            await submit(app, pilot, "tool")
            await asyncio.wait_for(started.wait(), 2)
            await pilot.press("escape", "escape")
            assert app.is_generating
            assert "waiting for running tool" in str(app.query_one("#status-label", Label).render())
            await submit(app, pilot, "queued")
            assert len(app._queue) == 1
            release.set()
            await wait_idle(app, pilot)
            assert executed == ["read_file"]
            assert len(requests) == 1
            assert app._history[1]["stopped"]
            assert [m["tool_call_id"] for m in app._history if m["role"] == "tool"] == ["c0", "c1"]
            assert app._history[2]["content"] == "real tool result"
            assert "not executed" in app._history[3]["content"]
            assert app._queue_paused
            assert app._turn is None
            saved = load_session(next(tmp_path.rglob("*.json")))
            assert saved.messages == app._history
    finally:
        release.set()


async def test_queue_is_visible_removable_and_resumes_fifo(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()
    requests = []

    async def stream(messages):
        requests.append(messages[-1]["content"])
        yield {"kind": "content", "chunk": "reply"}
        if len(requests) == 1:
            ready.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "first")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "duplicate")
        await submit(app, pilot, "duplicate")
        await submit(app, pilot, "last")
        assert [q.text for q in app.query(QueuedMessage)] == ["duplicate", "duplicate", "last"]
        assert len(app._history) == 1
        app.query_one("#cancel-queued-2", Button).focus()
        await pilot.press("enter")
        assert [q.id for q in app._queue] == [1, 3]
        app.query_one(ChatInput).focus()
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert requests == ["first"]
        await submit(app, pilot, "after stop")
        assert requests == ["first"]
        app.query_one("#resume-queue", Button).focus()
        await pilot.press("enter")
        await wait_idle(app, pilot)
        await pilot.pause()
        assert requests == ["first", "duplicate", "last", "after stop"]
        assert not app._queue
        assert not app.query(QueuedMessage)
        assert [m["content"] for m in app._history if m["role"] == "user"] == requests


async def test_queue_drains_after_success_and_busy_session_changes_are_rejected(
    tmp_path, monkeypatch
):
    app = chat_app(tmp_path)
    ready, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def stream(messages):
        requests.append(messages[-1]["content"])
        if len(requests) == 1:
            ready.set()
            await release.wait()
        yield {"kind": "content", "chunk": "reply"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "first")
        await asyncio.wait_for(ready.wait(), 2)
        original = app._session
        await submit(app, pilot, "/new")
        await submit(app, pilot, "/clear")
        await submit(app, pilot, "/sessions")
        assert app._session is original
        # Two synchronous admissions must not cancel or overwrite one another.
        app.query_one(ChatInput).insert("second")
        app.action_submit()
        app.query_one(ChatInput).insert("third")
        app.action_submit()
        release.set()
        await wait_idle(app, pilot)
        await pilot.pause()
        assert requests == ["first", "second", "third"]
        assert not app.query(QueuedMessage)


async def test_save_snapshots_draft_without_duplicate_assistant(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready, release = asyncio.Event(), asyncio.Event()

    async def stream(messages):
        yield {"kind": "content", "chunk": "partial"}
        ready.set()
        await release.wait()
        yield {"kind": "content", "chunk": " complete"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "/save")
        saved = load_session(next(tmp_path.rglob("*.json")))
        assert saved.messages[-1]["content"] == "partial"
        release.set()
        await wait_idle(app, pilot)
        assert len(app._history) == 2
        assert (
            load_session(next(tmp_path.rglob("*.json"))).messages[-1]["content"]
            == "partial complete"
        )


async def test_save_error_does_not_leave_busy_or_drain_queue(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()

    async def stream(messages):
        yield {"kind": "content", "chunk": "partial"}
        ready.set()
        await asyncio.Event().wait()

    def fail_save(*args):
        raise OSError("disk full")

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.save_session", fail_save)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "queued")
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert app._history[-1]["content"] == "partial"
        assert app._queue_paused and len(app._queue) == 1
        assert any(
            m._role == "error" and "Could not save" in m._content for m in app.query(Message)
        )
        assert isinstance(app.focused, ChatInput)


async def test_session_reset_clears_paused_queue_widgets(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()

    async def stream(messages):
        ready.set()
        await asyncio.Event().wait()
        yield {}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "queued")
        await pilot.press("escape")
        await wait_idle(app, pilot)
        await app._load_session(new_session("test-model"))
        await pilot.pause()
        assert not app._queue
        assert not app.query(QueuedMessage)
        assert not app._queue_paused


async def test_stop_before_worker_starts_finalizes_once(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    called = []

    async def stream(messages):
        called.append(True)
        yield {}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        app._start_turn("hello")
        app.action_stop_generation()
        app.action_stop_generation()
        await wait_idle(app, pilot)
        assert not called
        assert len(app._history) == 2
        assert app._history[-1] == {"role": "assistant", "content": "", "stopped": True}
        assert any(m._stopped for m in app.query(Message))
        assert load_session(next(tmp_path.rglob("*.json"))).messages == app._history


async def test_escape_after_completion_does_not_append_stopped_draft(tmp_path, monkeypatch):
    app = chat_app(tmp_path)

    async def stream(messages):
        yield {"kind": "content", "chunk": "done"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await wait_idle(app, pilot)
        await pilot.press("escape", "escape")
        assert len(app._history) == 2
        assert "stopped" not in app._history[-1]
        assert app.is_running


async def test_stream_error_keeps_partial_and_pauses_queue(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready, fail = asyncio.Event(), asyncio.Event()

    async def stream(messages):
        yield {"kind": "content", "chunk": "partial"}
        ready.set()
        await fail.wait()
        raise RuntimeError("network error")

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "queued")
        fail.set()
        await wait_idle(app, pilot)
        assert app._history[-1]["content"] == "partial"
        assert len(app._queue) == 1 and app._queue_paused
        assert any(m._role == "error" for m in app.query(Message))
        assert load_session(next(tmp_path.rglob("*.json"))).messages == app._history


async def test_stopping_owned_modal_pairs_tool_history(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    executed = []
    monkeypatch.setattr("drove.config.load_config", lambda _: SimpleNamespace(allowed_tools=[]))
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: executed.append(args))

    async def stream(messages):
        yield {
            "kind": "tool_call",
            "index": 0,
            "id": "c1",
            "name": "shell_execute",
            "arguments": "{}",
        }

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        async with asyncio.timeout(2):
            while not isinstance(app.screen, ToolConfirmationModal):
                await pilot.pause()
        app.action_stop_generation()
        await wait_idle(app, pilot)
        assert not isinstance(app.screen, ToolConfirmationModal)
        assert not executed
        assert app._history[1]["stopped"]
        assert app._history[2]["tool_call_id"] == "c1"
        assert "not executed" in app._history[2]["content"]


async def test_explicit_save_error_during_generation_is_reported(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready = asyncio.Event()

    async def stream(messages):
        yield {"kind": "content", "chunk": "partial"}
        ready.set()
        await asyncio.Event().wait()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.save_session", fail)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await submit(app, pilot, "/save")
        assert app.is_running and app.is_generating
        assert any("Could not save" in m._content for m in app.query(Message))
        assert not any("Session saved" in m._content for m in app.query(Message))
        await pilot.press("escape")
        await wait_idle(app, pilot)


async def test_quit_settles_running_tool_before_exiting(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def execute(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3)
        return "completed before quit"

    async def stream(messages):
        yield {"kind": "tool_call", "index": 0, "id": "c1", "name": "read_file", "arguments": "{}"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    try:
        async with app.run_test() as pilot:
            await submit(app, pilot, "tool")
            await asyncio.wait_for(started.wait(), 2)
            await pilot.press("ctrl+q")
            assert app.is_running and app.is_generating
            release.set()
            await wait_idle(app, pilot)
            await pilot.pause()
            assert not app.is_running
            saved = load_session(next(tmp_path.rglob("*.json")))
            assert saved.messages[-1]["content"] == "completed before quit"
    finally:
        release.set()


async def test_quit_closes_stream_and_saves_partial(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    ready, closed = asyncio.Event(), asyncio.Event()

    async def stream(messages):
        try:
            yield {"kind": "content", "chunk": "partial"}
            ready.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.press("ctrl+c")
        await wait_idle(app, pilot)
        await pilot.pause()
        assert not app.is_running
        assert closed.is_set()
        assert load_session(next(tmp_path.rglob("*.json"))).messages[-1]["content"] == "partial"


@pytest.mark.parametrize("reused_id", ["c1", "c2"])
async def test_stopped_tool_batch_pairs_reused_ids_in_saved_and_next_history(
    tmp_path, monkeypatch, reused_id
):
    app = chat_app(tmp_path)
    app._history.extend(
        [
            {"role": "user", "content": "earlier"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": reused_id,
                        "type": "function",
                        "function": {"name": "shell_execute", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": reused_id, "content": "earlier result"},
            {"role": "assistant", "content": "earlier answer"},
        ]
    )
    app._session.messages = app._history
    requests = []
    monkeypatch.setattr("drove.config.load_config", lambda _: SimpleNamespace(allowed_tools=[]))
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: "current result")

    async def stream(messages):
        requests.append(list(messages))
        if len(requests) == 1:
            if reused_id == "c2":
                yield {
                    "kind": "tool_call",
                    "index": 0,
                    "id": "c1",
                    "name": "read_file",
                    "arguments": "{}",
                }
            yield {
                "kind": "tool_call",
                "index": 1,
                "id": reused_id,
                "name": "shell_execute",
                "arguments": "{}",
            }
        else:
            yield {"kind": "content", "chunk": "next answer"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(app, pilot, "current")
        async with asyncio.timeout(2):
            while not isinstance(app.screen, ToolConfirmationModal):
                await pilot.pause()
        app.action_stop_generation()
        await wait_idle(app, pilot)
        saved = load_session(next(tmp_path.rglob("*.json"))).messages
        current_assistant = next(i for i, msg in enumerate(saved) if msg.get("stopped"))
        results = saved[current_assistant + 1 :]
        assert results[-1]["tool_call_id"] == reused_id
        assert "not executed because the reply was stopped" in results[-1]["content"]
        assert len(results) == (2 if reused_id == "c2" else 1)
        if reused_id == "c2":
            assert results[0]["content"] == "current result"
        assert saved[2]["content"] == "earlier result"
        await submit(app, pilot, "next")
        await wait_idle(app, pilot)
        assert requests[1][current_assistant + 1 : -1] == results
