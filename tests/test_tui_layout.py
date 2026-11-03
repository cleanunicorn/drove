"""Chat navigation, composer controls, and layout at common terminal sizes."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from textual.events import MouseScrollDown
from textual.widgets import Button, DataTable, Label, OptionList

from drove.chat_models import ChatModel
from drove.model_picker import ModelPicker
from drove.sessions import new_session, save_session
from drove.tui import ChatApp, ChatInput, CodePicker, Message, SessionPicker, Transcript


def chat(tmp_path: Path, theme: str = "textual-dark") -> ChatApp:
    return ChatApp(
        "http://localhost:8080", "org/a-long-model-name:Q4_K_M", tmp_path, None, theme=theme
    )


@pytest.mark.parametrize("size", [(44, 20), (60, 24), (80, 24), (120, 40)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_controls_fit_and_welcome_yields_to_conversation(tmp_path, size, theme):
    app = chat(tmp_path, theme)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert isinstance(app.focused, ChatInput)
        assert app.query_one("#conversation").has_class("empty")
        for selector in (
            "#chat-details",
            "#new-chat",
            "#welcome-help",
            "#user-input",
            "#send-message",
            "#guidance-label",
        ):
            control = app.query_one(selector)
            assert control.region.area > 0
            assert app.screen.region.contains_region(control.region), selector
        field = app.query_one(ChatInput)
        send = app.query_one("#send-message", Button)
        assert field.region.right <= send.region.x
        assert send.content_region.width >= len(send.label.plain) + 2
        assert app.query_one("#welcome-help").region.bottom <= field.region.y
        assert await pilot.click("#welcome-help")
        await pilot.pause()
        assert not app.query_one("#conversation").has_class("empty")
        assert "Available commands" in app.query_one(Message)._content
        assert await pilot.click("#new-chat")
        await pilot.pause()
        assert app.query_one("#conversation").has_class("empty")
        assert not app.query(Message)
        assert isinstance(app.focused, ChatInput)


async def test_send_and_stop_buttons_preserve_partial_reply(tmp_path, monkeypatch):
    app = chat(tmp_path)
    ready = asyncio.Event()

    async def stream(messages):
        yield {"kind": "content", "chunk": "A partial reply"}
        ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        app.query_one(ChatInput).insert("Hello")
        assert await pilot.click("#send-message")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.pause()
        assert app.query_one("#composer").has_class("generating")
        assert app.query_one("#activity-label", Label).render().plain == "Replying…"
        assert "Queue" in app.query_one(ChatInput).placeholder
        worker = app._generation_worker
        assert worker is not None
        assert await pilot.click("#stop-reply")
        await worker.wait()
        await pilot.pause()
        assert not app.is_generating
        assert not app.query_one("#composer").has_class("generating")
        assert app._history[-1]["stopped"] is True
        assert app._history[-1]["content"] == "A partial reply"
        assert not app.query_one("#status-bar").has_class("visible")
        assert isinstance(app.focused, ChatInput)


async def test_header_actions_open_model_and_session_pickers(tmp_path):
    app = chat(tmp_path)
    session = new_session(app._model, endpoint=app._endpoint)
    session.messages = [{"role": "user", "content": "Saved conversation"}]
    save_session(tmp_path, session)
    with patch("drove.tui.fetch_models_async", return_value=[ChatModel(app._model)]):
        async with app.run_test() as pilot:
            assert await pilot.click("#change-model")
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)
            await pilot.press("escape")
            await pilot.pause()
            assert await pilot.click("#open-sessions")
            await pilot.pause()
            assert isinstance(app.screen, SessionPicker)
            await pilot.press("enter")
            await pilot.pause()
            assert not app.query_one("#conversation").has_class("empty")
            assert app.query_one(Message)._content == "Saved conversation"


async def test_learned_tips_reclaim_rows_and_survive_restart(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    app = ChatApp("http://localhost:8080", "model", tmp_path, config_path)

    async def stream(messages):
        yield {"kind": "content", "chunk": "Done"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert "Enter sends" in app.query_one("#guidance-label", Label).render().plain
        field = app.query_one(ChatInput)
        field.insert("hello")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "send" in app._completed_hints
        assert "Shift+Enter" in app.query_one("#guidance-label", Label).render().plain
        await pilot.press("shift+enter")
        await pilot.pause()
        assert field.text == "\n"
        assert "newline" in app._completed_hints
        assert "Type /" in app.query_one("#guidance-label", Label).render().plain
        field.clear()
        field.insert("/help")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Info" in app.query_one("#guidance-label", Label).render().plain
        assert await pilot.click("#chat-details")
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query_one("#guidance").display
        assert not app.query_one("#status-bar").has_class("visible")
        # Toolbar (1 row) + separator (1) + input (1), with everything else for content.
        assert app.query_one("#conversation").size.height == 21
        assert app.query_one("#messages").content_size.width >= 77
    restarted = ChatApp("http://localhost:8080", "other-model", tmp_path, config_path)
    async with restarted.run_test() as pilot:
        await pilot.pause()
        assert not restarted.query_one("#guidance").display
        await restarted._dispatch_command("/tips")
        assert restarted.query_one("#guidance").display
        assert not restarted._completed_hints
        await restarted._dispatch_command("/tips off")
        assert not restarted.query_one("#guidance").display


async def test_dismiss_advances_single_tip_without_sending_or_clearing_draft(tmp_path):
    app = chat(tmp_path)
    async with app.run_test() as pilot:
        field = app.query_one(ChatInput)
        field.insert("keep my draft")
        assert await pilot.click("#dismiss-tip")
        await pilot.pause()
        assert app._completed_hints == {"send"}
        assert "Shift+Enter" in app.query_one("#guidance-label", Label).render().plain
        assert field.text == "keep my draft"
        assert not app._history
        assert isinstance(app.focused, ChatInput)


@pytest.mark.parametrize("size", [(44, 16), (80, 24)])
async def test_info_scrolls_without_leaking_diagnostics_into_conversation(tmp_path, size):
    app = chat(tmp_path)
    app._base_url = "https://example.test/" + "long-path/" * 70
    app._last_speed = 12.5
    app._last_ttft = 0.2
    app._last_usage = {"prompt_tokens": 123, "completion_tokens": 45}
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert not app.query_one("#status-bar").has_class("visible")
        assert "chunks/s" not in app.query_one("#status-label", Label).render().plain
        assert await pilot.click("#chat-details")
        await pilot.pause()
        modal = app.screen
        details = modal.query_one("#details-content", Label).render().plain
        assert app._model in details and app._base_url in details
        assert "12.5 chunks/s (received)" in details
        assert "0.20s" in details
        assert "Last prompt tokens: 123" in details
        assert "Last completion tokens: 45" in details
        assert modal.region.contains_region(modal.query_one("#close-details").region)
        viewport = modal.query_one("#details-scroll")
        assert viewport.max_scroll_y > 0
        viewport.scroll_end(animate=False)
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert await pilot.click("#close-details")
        await pilot.pause()
        assert not app.query(Message)


async def test_stream_usage_feeds_info_and_the_context_meter(tmp_path, monkeypatch):
    app = chat(tmp_path)
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
                "invalid": "secret",
            },
        },
    ]
    response = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    response += "data: [DONE]\n\n"
    real_client = httpx.AsyncClient
    payloads = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, text=response)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        "drove.tui.httpx.AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    async with app.run_test() as pilot:
        app._start_turn("hi")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert payloads[0]["stream_options"] == {"include_usage": True}
        assert app._last_usage == {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
        }
        assert app.query("Message.assistant").first()._content == "Hello"
        assert not app.query_one("#status-bar").has_class("visible")
        # No context size is known here, so the meter shows the raw count.
        assert app.query_one("#ctx-meter", Label).render().plain == "ctx 168"
        await app._dispatch_command("/info")
        await pilot.pause()
        assert (
            "Last prompt tokens: 123"
            in app.screen.query_one("#details-content", Label).render().plain
        )
        await pilot.press("escape")
        assert isinstance(app.focused, ChatInput)
        await app._dispatch_command("/new")
        assert app._last_usage == {}
        assert "Last prompt tokens: Not reported" in app._details_text()
        assert not app.query_one("#ctx-meter").has_class("visible")


# ── Mouse focus (#140): clicks outside controls keep the composer focused ──────

TRANSCRIPT = [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "Answer\n\n```python\nprint('copied')\n```\n"},
]


def resumed_chat(tmp_path: Path, messages: list[dict[str, str]] = TRANSCRIPT) -> ChatApp:
    session = new_session("a")
    session.messages = list(messages)
    save_session(tmp_path, session)  # so the session picker has a row to click
    return ChatApp("http://localhost:8080", "a", tmp_path, None, resume_session=session)


async def wait_for_messages(pilot, app: ChatApp, count: int) -> list[Message]:
    for _ in range(50):
        await pilot.pause()
        bubbles = list(app.query(Message))
        if len(bubbles) == count:
            return bubbles
    raise AssertionError("transcript was not replayed")


async def click_inside(pilot, message: Message) -> None:
    """Click the message body; the hit lands on a child, so pilot.click reports False."""
    x, y = message.region.offset + (2, 1)
    hit, _ = pilot.app.screen.get_widget_at(x, y)
    assert message in hit.ancestors_with_self
    await pilot.click(offset=(x, y))


async def test_clicking_messages_empty_transcript_and_header_keeps_input_focused(tmp_path):
    app = resumed_chat(tmp_path)
    async with app.run_test(size=(100, 40)) as pilot:
        user, assistant = await wait_for_messages(pilot, app, 2)
        await click_inside(pilot, user)
        assert isinstance(app.focused, ChatInput)
        await click_inside(pilot, assistant)
        assert isinstance(app.focused, ChatInput)

        transcript = app.query_one("#messages", Transcript)
        empty = (2, 0)  # messages are bottom-aligned, so the top row is empty
        x, y = transcript.region.offset + empty
        assert app.screen.get_widget_at(x, y)[0] is transcript  # really empty space
        assert await pilot.click(transcript, offset=empty)
        assert isinstance(app.focused, ChatInput)


async def test_click_returns_focus_from_a_keyboard_focused_message(tmp_path, monkeypatch):
    copied = Mock()
    app = resumed_chat(tmp_path)
    async with app.run_test(size=(100, 40)) as pilot:
        user, assistant = await wait_for_messages(pilot, app, 2)
        monkeypatch.setattr(app, "copy_to_clipboard", copied)
        for _ in range(10):  # Shift+Tab still walks back into the transcript
            await pilot.press("shift+tab")
            if isinstance(app.focused, Message):
                break
        assert app.focused is assistant
        await pilot.press("ctrl+y")
        copied.assert_called_once_with(TRANSCRIPT[1]["content"])

        assert await pilot.click("#brand")
        assert isinstance(app.focused, ChatInput)
        app.set_focus(user)
        await click_inside(pilot, assistant)
        assert isinstance(app.focused, ChatInput)


async def test_tab_reaches_messages_and_code_copy_still_works(tmp_path, monkeypatch):
    copied = Mock()
    app = resumed_chat(tmp_path)
    async with app.run_test(size=(100, 40)) as pilot:
        _, assistant = await wait_for_messages(pilot, app, 2)
        monkeypatch.setattr(app, "copy_to_clipboard", copied)
        for _ in range(20):
            await pilot.press("tab")
            if app.focused is assistant:
                break
        assert app.focused is assistant
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert isinstance(app.screen, CodePicker)
        await pilot.press("enter")
        await pilot.pause()
        copied.assert_called_once_with("print('copied')\n")
        assert app.focused is assistant


async def test_clicks_on_controls_keep_their_focus(tmp_path):
    app = resumed_chat(tmp_path)
    async with app.run_test(size=(100, 40)) as pilot:
        await wait_for_messages(pilot, app, 2)
        await pilot.press("/", "n")
        await pilot.pause()
        autocomplete = app.query_one("#autocomplete", OptionList)
        assert autocomplete.display and autocomplete.option_count
        assert await pilot.click(autocomplete, offset=(2, 1))  # row 0 is the border
        await pilot.pause()
        assert app.query_one(ChatInput).text.startswith("/n")
        assert isinstance(app.focused, ChatInput)

        app.query_one(ChatInput).clear()
        assert await pilot.click("#open-sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)
        table = app.screen.query_one("#session-table", DataTable)
        assert await pilot.click(table)
        assert app.focused is table  # modal tables keep their own focus


async def test_wheel_scrolls_the_transcript_without_moving_focus(tmp_path):
    long = [{"role": "user", "content": f"line {i}\n\nmore text"} for i in range(30)]
    app = resumed_chat(tmp_path, long)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_messages(pilot, app, 30)
        transcript = app.query_one("#messages", Transcript)
        transcript.scroll_home(animate=False)
        await pilot.pause()
        x, y = transcript.region.offset + (4, 2)
        for _ in range(3):
            app.post_message(MouseScrollDown(None, x, y, 0, 1, 0, False, False, False, x, y))
            await pilot.pause()
        assert transcript.scroll_y > 0
        assert isinstance(app.focused, ChatInput)


async def test_transcript_drag_selection_is_copied_with_input_focused(tmp_path, monkeypatch):
    copied = Mock()
    app = resumed_chat(tmp_path)
    async with app.run_test(size=(100, 40)) as pilot:
        _, assistant = await wait_for_messages(pilot, app, 2)
        monkeypatch.setattr(app, "copy_to_clipboard", copied)
        composer = app.query_one(ChatInput)
        composer.insert("my draft")
        composer.select_all()

        await pilot.mouse_down(assistant, offset=(2, 1))
        await pilot.hover(assistant, offset=(6, 1))
        await pilot.mouse_up(assistant, offset=(6, 1))
        await pilot.pause()
        selected = app.screen.get_selected_text()
        assert selected
        assert isinstance(app.focused, ChatInput)
        # Starting the drag dropped the draft selection, so the composer's copy binding
        # skips and the screen copies the transcript selection.
        assert not composer.selected_text
        await pilot.press("super+c")
        copied.assert_called_once_with(selected)

        composer.select_all()  # a newer draft selection wins over the transcript one
        await pilot.press("super+c")
        copied.assert_called_with("my draft")
