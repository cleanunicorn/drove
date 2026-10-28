"""Live tool feedback: running entries, activity label, end states and streamed output."""

import asyncio
import json
import threading
import time
import tomllib
from pathlib import Path

import httpx
import pytest
from textual.widgets import Button, Collapsible, Label

from drove import tools
from drove.sessions import Session, new_session
from drove.tool_policy import TOOL_OUTPUT_LIMIT, TRUNCATION_NOTICE, Grant
from drove.tools import ToolFailure
from drove.tui import (
    TOOL_DENIED,
    TOOL_OUTPUT_LINES,
    ChatApp,
    ChatInput,
    ToolCallEntry,
    ToolConfirmationModal,
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

    def execute(name, arguments, on_output=None, **kwargs):
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

    def execute(name, arguments, on_output=None, **kwargs):
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

    def execute(name, arguments, on_output=None, **kwargs):
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


# ── Approval policy in the chat (AC2, AC3) ──────────────────────────────────────


def policy_app(tmp_path: Path, monkeypatch, endpoint: str = "http://localhost:8080"):
    """A chat started in *work*, with its own config file."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    config = tmp_path / "config.toml"
    config.write_text("listen_port = 7000\n")
    return ChatApp(endpoint, "test-model", tmp_path / "sessions", config), work, config


def answering(app: ChatApp, monkeypatch, *choices: str) -> list[ToolConfirmationModal]:
    """Answer each confirmation with the next of *choices*; return the modals shown."""
    shown: list[ToolConfirmationModal] = []
    answers = iter(choices)

    async def push_screen_wait(screen):
        shown.append(screen)
        return next(answers)

    monkeypatch.setattr(app, "push_screen_wait", push_screen_wait)
    return shown


def outside_dir(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "secret.txt").write_text("secret")
    return outside


async def test_reads_ask_only_outside_the_working_directory(tmp_path, monkeypatch):
    app, work, _ = policy_app(tmp_path, monkeypatch)
    outside = outside_dir(tmp_path)
    (work / "link").symlink_to(outside)
    shown = answering(app, monkeypatch, "deny-once", "deny-once", "deny-once", "deny-once")

    assert await app._tool_allowed("read_file", json.dumps({"path": "notes.txt"}))
    assert await app._tool_allowed("list_files", "{}")
    assert await app._tool_allowed("search_files", json.dumps({"path": "."}))
    assert shown == []

    assert not await app._tool_allowed("read_file", json.dumps({"path": str(outside / "x")}))
    assert not await app._tool_allowed("list_files", json.dumps({"path": "link"}))
    assert not await app._tool_allowed("search_files", json.dumps({"path": "link/deep"}))
    assert not await app._tool_allowed("read_file", json.dumps({"path": "../outside/x"}))
    assert len(shown) == 4
    assert all("outside the working directory" in modal._reason for modal in shown)


async def test_every_read_asks_with_a_remote_endpoint(tmp_path, monkeypatch):
    app, _, config = policy_app(tmp_path, monkeypatch, "https://api.example.com")
    config.write_text(
        f"allowed_tools = [{json.dumps(Grant('read_file', directory='/').token())}]\n"
    )
    shown = answering(app, monkeypatch, "allow-once", "allow-once")

    assert await app._tool_allowed("read_file", json.dumps({"path": "notes.txt"}))
    assert await app._tool_allowed("read_file", json.dumps({"path": "notes.txt"}))
    assert len(shown) == 2
    assert all(modal._per_call and "not local" in modal._reason for modal in shown)


async def test_allow_always_saves_a_directory_scope_and_nothing_else(tmp_path, monkeypatch):
    app, _, config = policy_app(tmp_path, monkeypatch)
    monkeypatch.setenv("DROVE_LISTEN_PORT", "9000")
    outside = outside_dir(tmp_path)
    (tmp_path / "sibling").mkdir()
    shown = answering(app, monkeypatch, "allow-always", "deny-once")

    assert await app._tool_allowed("read_file", json.dumps({"path": str(outside / "a.txt")}))
    assert await app._tool_allowed("read_file", json.dumps({"path": str(outside / "deep/b")}))
    assert not await app._tool_allowed(
        "read_file", json.dumps({"path": str(tmp_path / "sibling/c")})
    )
    assert len(shown) == 2
    assert shown[0]._scope == f"read_file in {outside.resolve()}"
    token = Grant("read_file", directory=str(outside.resolve())).token()
    assert tomllib.loads(config.read_text()) == {"listen_port": 7000, "allowed_tools": [token]}

    # A new chat honours the saved scope, and only for read_file.
    fresh = ChatApp("http://localhost:8080", "test-model", tmp_path / "sessions", config)
    shown = answering(fresh, monkeypatch, "deny-once")
    assert await fresh._tool_allowed("read_file", json.dumps({"path": str(outside / "a.txt")}))
    assert not await fresh._tool_allowed("list_files", json.dumps({"path": str(outside)}))
    assert len(shown) == 1


@pytest.mark.parametrize("name", ["shell_execute", "write_file"])
async def test_shell_and_write_ask_every_call_and_ignore_bare_grants(tmp_path, monkeypatch, name):
    app, _, config = policy_app(tmp_path, monkeypatch)
    config.write_text(f'allowed_tools = ["{name}", "fetch_url"]\n')
    shown = answering(app, monkeypatch, "allow-always", "allow-session", "allow-once")
    arguments = json.dumps({"command": "ls", "path": "x.txt", "content": ""})

    for _ in range(3):
        assert await app._tool_allowed(name, arguments)

    assert len(shown) == 3
    assert all(modal._per_call for modal in shown)
    assert tomllib.loads(config.read_text()) == {"allowed_tools": [name, "fetch_url"]}
    assert app._session_allowed_tools == set()


async def test_fetch_grants_are_per_origin_and_wider_grants_are_ignored(tmp_path, monkeypatch):
    app, _, config = policy_app(tmp_path, monkeypatch)
    host_only = json.dumps({"host": "example.com", "tool": "fetch_url", "v": 1})
    config.write_text(f'allowed_tools = ["fetch_url", {json.dumps(host_only)}]\n')
    shown = answering(app, monkeypatch, "allow-session", "deny-once", "deny-once", "deny-once")

    def fetch(url: str):
        return app._tool_allowed("fetch_url", json.dumps({"url": url}))

    assert await fetch("https://example.com/a")
    assert await fetch("https://example.com:443/b")
    assert not await fetch("http://example.com/")
    assert not await fetch("https://example.com:8443/")
    assert not await fetch("https://other.org/")
    assert len(shown) == 4
    assert shown[0]._scope == "fetch_url from https://example.com:443"
    assert app._session_allowed_tools == {
        Grant("fetch_url", origin="https://example.com:443").token()
    }


async def test_per_call_dialog_offers_only_allow_once_or_deny(tmp_path):
    app = chat_app(tmp_path)
    choices: list[str] = []
    async with app.run_test() as pilot:
        modal = ToolConfirmationModal("shell_execute", "{}", per_call=True)
        app.push_screen(modal, callback=choices.append)
        await pilot.pause()
        assert {button.id for button in app.screen.query(Button)} == {"allow-once", "deny-once"}
        await pilot.press("a", "s")
        await pilot.pause()
        assert choices == []
        await pilot.press("y")
        await pilot.pause()
        assert choices == ["allow-once"]


async def test_an_approved_outside_read_runs_within_its_scope(tmp_path, monkeypatch):
    app, _, _ = policy_app(tmp_path, monkeypatch)
    outside = outside_dir(tmp_path)
    (outside / "deep" / "inner.txt").write_text("inner")
    requests: list = []
    answering(app, monkeypatch, "allow-once")
    call = json.dumps({"path": str(outside / "deep" / "inner.txt")})
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "read_file", call))

    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: len(requests) == 2)
    assert requests[1][-1]["content"] == "inner"


async def test_a_search_inside_cwd_skips_symlinks_that_lead_out(tmp_path, monkeypatch):
    app, work, _ = policy_app(tmp_path, monkeypatch)
    outside = outside_dir(tmp_path)
    (work / "leak.txt").symlink_to(outside / "secret.txt")
    requests: list = []
    answering(app, monkeypatch)  # no prompt expected
    search = json.dumps({"path": ".", "content_query": "secret"})
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "search_files", search))

    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: len(requests) == 2)
    content = requests[1][-1]["content"]
    assert content.startswith("No matches found.")
    assert "outside the approved directories" in content


async def test_fetch_asks_again_mid_fetch_for_an_internal_address(tmp_path, monkeypatch):
    app, _, _ = policy_app(tmp_path, monkeypatch)
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"intranet page")

    monkeypatch.setattr(tools, "_resolve", lambda host, port: ["10.0.0.7"])
    monkeypatch.setattr(tools, "_fetch_transport", httpx.MockTransport(handle))
    shown = answering(app, monkeypatch, "allow-once", "allow-once")
    requests: list = []
    call = json.dumps({"url": "http://intranet.example/"})
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "fetch_url", call))

    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: len(requests) == 2)

    assert requests[1][-1]["content"] == "intranet page"
    assert [modal._reason for modal in shown] == [
        "Fetch from http://intranet.example:80.",
        "http://intranet.example:80 is or resolves to an internal (private, loopback or"
        " link-local) address.",
    ]
    assert [request.url.host for request in sent] == ["10.0.0.7"]


# ── Responsiveness and saved output (AC4, AC5) ──────────────────────────────────


async def test_input_stays_responsive_during_a_five_second_shell_tool(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    requests: list = []
    threads: list[int] = []
    real_execute = tools.execute_tool

    def execute(*args, **kwargs):
        threads.append(threading.get_ident())
        return real_execute(*args, **kwargs)

    command = json.dumps({"command": "sleep 5", "timeout": 10})
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests, "shell_execute", command))
    monkeypatch.setattr(app, "_tool_allowed", lambda *args: asyncio.sleep(0, True))
    monkeypatch.setattr("drove.tui.execute_tool", execute)

    async with app.run_test() as pilot:
        started = time.monotonic()
        app._start_turn("go")
        await until(pilot, lambda: bool(threads))
        chat_input = app.query_one(ChatInput)
        chat_input.focus()
        await pilot.press(*"still typing")
        await pilot.pause()
        typed_after = time.monotonic() - started

        assert chat_input.text == "still typing"
        assert typed_after < 4  # typed while the 5 s command was still running
        assert app.is_generating
        async with asyncio.timeout(15):
            while app.is_generating:
                await pilot.pause(0.1)

    assert threads[0] != threading.get_ident()
    assert requests[1][-1]["content"] == "STDOUT:\n\nReturn code: 0"


async def test_saved_session_never_holds_a_tool_result_over_the_cap(tmp_path, monkeypatch):
    app = chat_app(tmp_path)
    requests: list = []
    huge = "ü" * TOOL_OUTPUT_LIMIT  # 128 KiB of UTF-8
    monkeypatch.setattr(app, "_stream_chat", tool_then_reply(requests))
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: huge)

    async with app.run_test() as pilot:
        app._start_turn("go")
        await until(pilot, lambda: not app.is_generating and len(requests) == 2)

    [saved] = (tmp_path / "test-model").glob("*.json")
    messages = json.loads(saved.read_text())["messages"]
    [tool_message] = [m for m in messages if m["role"] == "tool"]
    assert len(tool_message["content"].encode("utf-8")) <= TOOL_OUTPUT_LIMIT
    assert tool_message["content"].endswith(TRUNCATION_NOTICE)
    assert len(requests[1][-1]["content"].encode("utf-8")) <= TOOL_OUTPUT_LIMIT


def test_resumed_sessions_cap_old_oversized_tool_results(tmp_path):
    huge = "x" * (TOOL_OUTPUT_LIMIT * 3)
    session = Session(
        model="test-model",
        session_id="old",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": huge},
        ],
    )

    app = ChatApp("http://localhost:8080", "test-model", tmp_path, None, resume_session=session)

    assert len(app._history[1]["content"].encode()) <= TOOL_OUTPUT_LIMIT
    assert app._history[0] == {"role": "user", "content": "hi"}


async def test_sessions_loaded_from_the_picker_cap_old_oversized_tool_results(
    tmp_path, monkeypatch
):
    app = chat_app(tmp_path)
    huge = "x" * (TOOL_OUTPUT_LIMIT * 3)
    old = Session(
        model="test-model",
        session_id="old",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": huge},
        ],
    )
    requests: list = []

    async def stream(messages):
        requests.append([dict(m) for m in messages])
        yield {"kind": "content", "chunk": "ok"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await app._load_session(old)
        assert len(app._history[2]["content"].encode()) <= TOOL_OUTPUT_LIMIT
        app._start_turn("next")
        await until(pilot, lambda: bool(requests) and not app.is_generating)

    [tool_message] = [m for m in requests[0] if m["role"] == "tool"]
    assert len(tool_message["content"].encode("utf-8")) <= TOOL_OUTPUT_LIMIT
    assert tool_message["content"].endswith(TRUNCATION_NOTICE)
