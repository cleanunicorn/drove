"""Model switching preserves conversations and works through real modal keys."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from textual.widgets import OptionList

from drove.chat_models import CatalogError, ChatModel
from drove.model_picker import ModelPicker
from drove.sessions import list_sessions, new_session, save_session
from drove.tui import ChatApp, ChatInput, SessionPicker


def chat(tmp_path: Path) -> ChatApp:
    session = new_session("a", "remember this prompt")
    session.messages.append({"role": "user", "content": "old conversation"})
    return ChatApp(
        "https://example.test", "a", tmp_path, None, resume_session=session, api_key="secret"
    )


def ticking_clock(start: datetime):
    """``datetime`` stand-in whose ``now`` advances one microsecond per call."""

    class Clock:
        current = start

        @classmethod
        def now(cls) -> datetime:
            cls.current += timedelta(microseconds=1)
            return cls.current

    return Clock


async def open_picker(app: ChatApp, pilot):
    worker = app.run_worker(app._dispatch_command("/model"))
    await pilot.pause()
    assert isinstance(app.screen, ModelPicker)
    return worker


async def test_switch_saves_old_session_resets_state_and_changes_next_payload(
    tmp_path: Path,
) -> None:
    app = chat(tmp_path)
    old_session = app._session
    app._session_allowed_tools.add("shell")
    app._last_speed = 20
    app._last_ttft = 1
    with patch(
        "drove.tui.fetch_models_async", return_value=[ChatModel("a"), ChatModel("b")]
    ) as fetch:
        async with app.run_test() as pilot:
            worker = await open_picker(app, pilot)
            assert app.screen.query_one(OptionList).highlighted == 0
            await pilot.press("down", "enter")
            await worker.wait()
            assert app._model == "b"
            assert app._session.model == "b"
            assert app._session.id != old_session.id
            assert app._history == [{"role": "system", "content": "remember this prompt"}]
            assert not app._session_allowed_tools
            assert app._last_speed is None and app._last_ttft is None
            assert "Model: b\n" in app._details_text()
            assert isinstance(app.focused, ChatInput)
            assert list_sessions(tmp_path, "a")[0].messages == old_session.messages
            assert "/model" in app._COMMANDS
            seen = []

            def reply(request: httpx.Request) -> httpx.Response:
                seen.append(json.loads(request.content))
                assert request.headers["authorization"] == "Bearer secret"
                return httpx.Response(200, text="data: [DONE]\n\n")

            client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
            with patch("drove.tui.httpx.AsyncClient", return_value=client):
                assert [event async for event in app._stream_chat(app._history)] == []
            assert seen[0]["model"] == "b"
            assert seen[0]["messages"] == app._history
    fetch.assert_awaited_once_with("https://example.test", "secret")


@pytest.mark.parametrize("key", ["escape", "ctrl+c", "enter"])
async def test_cancel_or_current_selection_preserves_chat(tmp_path: Path, key: str) -> None:
    app = chat(tmp_path)
    old = app._session
    with patch("drove.tui.fetch_models_async", return_value=[ChatModel("a"), ChatModel("b")]):
        async with app.run_test() as pilot:
            worker = await open_picker(app, pilot)
            await pilot.press(key)
            await worker.wait()
            assert not isinstance(app.screen, ModelPicker)
            assert app.is_running
            assert app._session is old
            assert app._model == "a"
            assert not app._model_picker_open
    assert list_sessions(tmp_path, "a") == []


@pytest.mark.parametrize("busy", ["generating", "queued"])
async def test_busy_chat_does_not_fetch_or_switch(tmp_path: Path, busy: str) -> None:
    app = chat(tmp_path)
    with patch("drove.tui.fetch_models_async") as fetch:
        async with app.run_test() as pilot:
            if busy == "generating":
                app.is_generating = True
            else:
                app._queue.append("pending")
            await app._dispatch_command("/model")
            await pilot.pause()
            assert app._model == "a"
            fetch.assert_not_called()
            app._queue.clear()
            app.is_generating = False


async def test_fetch_failure_is_noop(tmp_path: Path) -> None:
    app = chat(tmp_path)
    old = app._session
    with patch("drove.tui.fetch_models_async", side_effect=CatalogError("offline")):
        async with app.run_test():
            await app._dispatch_command("/model")
            assert app._session is old
            assert not app._model_picker_open


async def test_save_failure_is_noop(tmp_path: Path) -> None:
    app = chat(tmp_path)
    old = app._session
    with (
        patch("drove.tui.fetch_models_async", return_value=[ChatModel("b")]),
        patch("drove.tui.save_session", side_effect=OSError("read only")),
    ):
        async with app.run_test() as pilot:
            worker = await open_picker(app, pilot)
            await pilot.press("enter")
            await worker.wait()
            assert app._session is old and app._model == "a"
            assert not app._model_picker_open


async def test_fetch_in_flight_is_single_and_rechecks_generation(tmp_path: Path) -> None:
    app = chat(tmp_path)
    ready = asyncio.Event()

    async def fetch(*args):
        await ready.wait()
        return [ChatModel("b")]

    with patch("drove.tui.fetch_models_async", side_effect=fetch) as mock:
        async with app.run_test() as pilot:
            worker = app.run_worker(app._dispatch_command("/model"))
            await pilot.pause()
            await app._dispatch_command("/model")
            app.is_generating = True
            ready.set()
            await worker.wait()
            mock.assert_awaited_once()
            assert app._model == "a" and not isinstance(app.screen, ModelPicker)
            app.is_generating = False


async def test_generation_after_picker_open_cannot_switch(tmp_path: Path) -> None:
    app = chat(tmp_path)
    with patch("drove.tui.fetch_models_async", return_value=[ChatModel("b")]):
        async with app.run_test() as pilot:
            worker = await open_picker(app, pilot)
            app.is_generating = True
            await pilot.press("enter")
            await worker.wait()
            assert app._model == "a"
            app.is_generating = False


async def test_rapid_switch_back_never_overwrites_previous_conversation(tmp_path: Path) -> None:
    clock = ticking_clock(datetime(2026, 1, 1, 12))
    with patch("drove.sessions.datetime", clock):
        app = chat(tmp_path)
    old = app._session

    with (
        patch("drove.tui.fetch_models_async", return_value=[ChatModel("a"), ChatModel("b")]),
        patch("drove.sessions.datetime", clock),
    ):
        async with app.run_test() as pilot:
            worker = await open_picker(app, pilot)
            await pilot.press("down", "enter")
            await worker.wait()
            worker = await open_picker(app, pilot)
            await pilot.press("up", "enter")
            await worker.wait()
            assert app._model == "a" and app._session.id != old.id
            assert app._session.id[:15] == old.id[:15]  # all within one second
            first_return_id = app._session.id
            await app._dispatch_command("/save")
            # Two returns within the same timestamp stay ordered as well as
            # distinct, so --resume chooses the newest switched session.
            worker = await open_picker(app, pilot)
            await pilot.press("down", "enter")
            await worker.wait()
            worker = await open_picker(app, pilot)
            await pilot.press("up", "enter")
            await worker.wait()
            assert app._session.id > first_return_id
            await app._dispatch_command("/save")
    records = list_sessions(tmp_path, "a")
    assert len(records) == 3
    assert next(s for s in records if s.id == old.id).messages == old.messages


async def test_pending_picker_keeps_unsent_input_and_does_not_start_request(tmp_path: Path) -> None:
    app = chat(tmp_path)
    async with app.run_test() as pilot:
        app._model_picker_open = True
        field = app.query_one("#user-input", ChatInput)
        field.load_text("keep this draft")
        with patch.object(app, "_send_message") as send:
            app.action_submit()
            await pilot.pause()
            send.assert_not_called()
        assert field.text == "keep this draft"
        app._model_picker_open = False


@pytest.mark.parametrize("command", ["/new", "/clear"])
async def test_new_conversation_in_the_same_second_gets_its_own_file(
    tmp_path: Path, command: str
) -> None:
    clock = ticking_clock(datetime(2026, 1, 1, 12))
    with patch("drove.sessions.datetime", clock):
        app = chat(tmp_path)
        async with app.run_test():
            first = app._session
            await app._dispatch_command("/save")
            await app._dispatch_command(command)
            assert app._session.id > first.id and app._session.id[:15] == first.id[:15]
            app._session.messages.append({"role": "user", "content": "second"})
            await app._dispatch_command("/save")
    assert [s.id for s in list_sessions(tmp_path, "a")] == [app._session.id, first.id]


async def test_sessions_picker_skips_corrupt_file_without_printing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = chat(tmp_path)
    saved = new_session("a")
    saved.messages.append({"role": "user", "content": "keep me"})
    save_session(tmp_path, saved)
    (tmp_path / "a" / "20990101-000000.json").write_text("{")
    with caplog.at_level(logging.WARNING):
        async with app.run_test() as pilot:
            worker = app.run_worker(app._dispatch_command("/sessions"))
            await pilot.pause()
            assert isinstance(app.screen, SessionPicker)
            assert [s.id for s in app.screen._sessions] == [saved.id]
            await pilot.press("enter")
            await worker.wait()
            assert app._session.id == saved.id
    # Root logging writes to the terminal Textual owns; diagnostics go to app.log.
    assert [r for r in caplog.records if r.name == "drove.sessions"] == []
