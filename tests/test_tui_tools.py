"""Live tool feedback: running entries, activity label, end states and streamed output."""

import asyncio
import threading
from pathlib import Path

import pytest
from textual.widgets import Collapsible, Label

from drove.sessions import new_session
from drove.tools import ToolFailure
from drove.tui import (
    TOOL_DENIED,
    TOOL_OUTPUT_LINES,
    ChatApp,
    ToolCallEntry,
    ToolOutputTail,
)

LONG_ARGUMENTS = '{"path": "' + "x" * 100 + '"}'


def chat_app(tmp_path: Path) -> ChatApp:
    return ChatApp("http://localhost:8080", "test-model", tmp_path, None)


def title(entry: ToolCallEntry) -> str:
    return entry.query_one(Collapsible)._title.label.plain


def activity(app: ChatApp) -> str:
    return app.query_one("#activity-label", Label).render().plain


async def until(pilot, condition) -> None:
    async with asyncio.timeout(3):
        while not condition():
            await pilot.pause()


def tool_then_reply(requests: list, name: str = "read_file", arguments: str = LONG_ARGUMENTS):
    async def stream(messages):
        requests.append([dict(m) for m in messages])
        if len(requests) == 1:
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "c1",
                "name": name,
                "arguments": arguments,
            }
        else:
            yield {"kind": "content", "chunk": "Done"}

    return stream


async def test_running_entry_names_tool_then_finishes_in_place(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    started, release = threading.Event(), threading.Event()
    requests: list = []

    def execute(name, arguments, on_output=None):
        started.set()
        assert release.wait(3)
        return "the result"

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests))
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    try:
        async with app.run_test() as pilot:
            app._start_turn("go")
            await until(pilot, started.is_set)
            await pilot.pause()
            [entry] = app.query(ToolCallEntry)
            assert entry.state == "running"
            assert entry.has_class("-running")
            running_title = title(entry)
            assert running_title.startswith("⚙ read_file  ")
            assert '{"path": "xxx' in running_title and "…" in running_title
            assert "x" * 100 not in running_title
            assert running_title.endswith("· running 0s")
            assert activity(app) == "Running read_file… 0s"

            # Elapsed time ticks in both the activity label and the entry title.
            assert entry.started is not None
            entry.started -= 5
            await until(pilot, lambda: activity(app) == "Running read_file… 5s")
            assert title(entry).endswith("· running 5s")

            release.set()
            await until(pilot, lambda: not app.is_generating)
            assert app.query(ToolCallEntry).nodes == [entry]
            assert entry.state == "done"
            assert title(entry) == "✓ read_file → the result"
            assert "Result:\nthe result" in str(
                entry.query_one(Collapsible).query_one(Label).render()
            )
            assert requests[1][-1] == {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "the result",
            }
            assert activity(app) == ""
    finally:
        release.set()


@pytest.mark.parametrize(
    ("result", "state", "mark"),
    [("fine", "done", "✓"), (ToolFailure("Error: file not found: x"), "failed", "✗")],
)
async def test_done_and_failed_are_distinct(tmp_path, monkeypatch, result, state, mark):
    app = chat_app(tmp_path)
    requests: list = []
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, arguments="{}"))
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: result)
    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: not app.is_generating)
        [entry] = app.query(ToolCallEntry)
        assert entry.state == state
        assert entry.has_class(f"-{state}")
        assert title(entry).startswith(f"{mark} read_file")
        assert ("failed" in title(entry)) == (state == "failed")
        # The model still receives the exact text, as a plain string.
        content = requests[1][-1]["content"]
        assert content == str(result) and type(content) is str


async def test_tool_that_raises_after_starting_is_failed_not_skipped(tmp_path, monkeypatch):
    app = chat_app(tmp_path)

    def crash(*args, **kwargs):
        raise TypeError("expected str, not NoneType")

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply([], arguments='{"path": null}'))
    monkeypatch.setattr("drove.tui.execute_tool", crash)
    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: not app.is_generating)
        [entry] = app.query(ToolCallEntry)
        assert entry.state == "failed" and entry.has_class("-failed")
        assert title(entry).startswith("✗ read_file failed → Error executing tool:")
        body = str(entry.query_one(Collapsible).query_one(Label).render())
        assert "expected str, not NoneType" in body
        # The turn still ends in the existing error bubble with a paired tool history.
        assert app.query("Message.error")
        assert app._history[-1]["role"] == "tool"


async def test_denied_tool_is_distinct_and_never_runs(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    requests: list = []
    executed: list = []

    async def deny(name, arguments):
        return False

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "shell_execute", "{}"))
    monkeypatch.setattr(app, "_tool_allowed", deny)
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: executed.append(args))
    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: not app.is_generating)
        [entry] = app.query(ToolCallEntry)
        assert executed == []
        assert entry.state == "denied" and entry.has_class("-denied")
        assert title(entry).startswith("⊘ shell_execute denied")
        assert requests[1][-1]["content"] == TOOL_DENIED


async def test_pending_approval_is_not_shown_as_running_and_stop_marks_not_run(
    tmp_path, monkeypatch
):
    app = chat_app(tmp_path)
    asked = asyncio.Event()

    async def wait_forever(name, arguments):
        asked.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply([], "shell_execute", "{}"))
    monkeypatch.setattr(app, "_tool_allowed", wait_forever)
    async with app.run_test() as pilot:
        app._start_turn("go")
        await asyncio.wait_for(asked.wait(), 2)
        await pilot.pause()
        [entry] = app.query(ToolCallEntry)
        assert entry.state == "pending"
        assert title(entry).endswith("· pending")
        assert activity(app) == "Replying…"
        app.action_stop_generation()
        await until(pilot, lambda: not app.is_generating)
        assert entry.state == "stopped"
        assert title(entry).startswith("⊘ shell_execute not run")
        assert "not executed" in app._history[-1]["content"]


async def test_shell_output_streams_into_the_running_entry_bounded(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    streamed, flood, release = threading.Event(), threading.Event(), threading.Event()
    requests: list = []
    full = "".join(f"line {i}\n" for i in range(5000))

    def execute(name, arguments, on_output=None):
        assert on_output is not None
        on_output("partial without newline")
        streamed.set()
        assert flood.wait(3)
        for i in range(0, len(full), 1000):
            on_output(full[i : i + 1000])
        assert release.wait(3)
        return full

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "shell_execute", "{}"))
    monkeypatch.setattr(app, "_tool_allowed", lambda *args: asyncio.sleep(0, True))
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    try:
        async with app.run_test() as pilot:
            app._start_turn("go")
            await until(pilot, streamed.is_set)
            [entry] = app.query(ToolCallEntry)
            preview = entry.query_one(".tool-output", Label)
            await until(pilot, lambda: entry.has_class("-streaming"))
            assert preview.render().plain == "partial without newline"
            assert entry.state == "running"

            # A flood keeps only a short tail on screen.
            flood.set()
            await until(pilot, lambda: "line 4999" in preview.render().plain)
            shown = preview.render().plain.split("\n")
            assert shown[0] == "… earlier output omitted"
            assert len(shown) == TOOL_OUTPUT_LINES + 1
            assert shown[-1] == "line 4999"

            release.set()
            await until(pilot, lambda: not app.is_generating)
            assert entry.state == "done"
            assert not entry.has_class("-streaming")
            # The preview is presentation only: the model receives everything.
            assert requests[1][-1]["content"] == full
    finally:
        flood.set()
        release.set()


async def test_stop_while_streaming_keeps_the_result_and_skips_follow_up(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    streamed, release = threading.Event(), threading.Event()
    requests: list = []

    def execute(name, arguments, on_output=None):
        on_output("working")
        streamed.set()
        assert release.wait(3)
        return "STDOUT:\nworking\nReturn code: 0"

    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "shell_execute", "{}"))
    monkeypatch.setattr(app, "_tool_allowed", lambda *args: asyncio.sleep(0, True))
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    try:
        async with app.run_test() as pilot:
            app._start_turn("go")
            await until(pilot, streamed.is_set)
            [entry] = app.query(ToolCallEntry)
            await until(pilot, lambda: entry.has_class("-streaming"))
            await pilot.press("escape")
            await pilot.pause()
            assert app.is_generating
            assert activity(app) == "Stopping…"
            release.set()
            await until(pilot, lambda: not app.is_generating)
            assert entry.state == "done"
            assert len(requests) == 1
            assert app._history[-1]["content"] == "STDOUT:\nworking\nReturn code: 0"
            assert app._turn is None
    finally:
        release.set()


async def test_replayed_tool_calls_render_finished_without_guessing(tmp_path):
    session = new_session("test-model")
    session.messages += [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "shell_execute", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "Error: looks like a failure"},
        {"role": "tool", "tool_call_id": "b", "content": TOOL_DENIED},
    ]
    app = ChatApp("http://localhost:8080", "test-model", tmp_path, None, resume_session=session)
    async with app.run_test() as pilot:
        await until(pilot, lambda: len(app.query(ToolCallEntry)) == 2)
        replayed, denied = app.query(ToolCallEntry)
        assert replayed.state == "finished"
        assert title(replayed) == "⚙ read_file → Error: looks like a failure"
        assert denied.state == "denied"


def test_output_tail_is_bounded_thread_safe_and_ignores_late_output():
    tail = ToolOutputTail(limit=100)
    assert tail.take() is None
    threads = [
        threading.Thread(target=lambda: [tail.append("y" * 10) for _ in range(100)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    shown = tail.take()
    assert shown == "… earlier output omitted\n" + "y" * 100
    assert tail.take() is None
    tail.close()
    tail.append("late")
    assert tail.take() is None


async def test_allow_always_persists_only_the_chosen_tool(tmp_path, monkeypatch):
    import tomllib

    config_path = tmp_path / "config.toml"
    config_path.write_text('listen_port = 7000\nallowed_tools = ["fetch_url"]\n')
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    monkeypatch.setenv("DROVE_ALLOWED_TOOLS", '["write_file"]')
    app = ChatApp("http://localhost:8080", "test-model", tmp_path, config_path)

    async def always(screen):
        return "allow-always"

    monkeypatch.setattr(app, "push_screen_wait", always)

    assert await app._tool_allowed("shell_execute", "{}")
    assert tomllib.loads(config_path.read_text()) == {
        "listen_port": 7000,
        "allowed_tools": ["fetch_url", "shell_execute"],
    }
