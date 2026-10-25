"""Chat text is literal at every untrusted rendering boundary."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.content import Content
from textual.widgets import Button, Collapsible, Label

from drove.sessions import Session
from drove.tui import CSS, ChatApp, Message, ToolConfirmationModal


class MessageApp(App[None]):
    CSS = CSS

    def __init__(self, message: Message) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield self.message


def assert_literal(label: Label, expected: str) -> None:
    rendered = label.render()
    assert isinstance(rendered, Content)
    assert rendered.plain == expected
    assert not rendered.spans


@pytest.mark.parametrize("role", ["user", "assistant", "error"])
@pytest.mark.parametrize("text", ["a[i]", "[bold]x[/bold]", "[/x]"])
async def test_message_body_is_literal(role: str, text: str) -> None:
    message = Message(role, text)
    app = MessageApp(message)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert_literal(message.query_one(".message-body", Label), text)


@pytest.mark.parametrize("text", ["a[i]", "[bold]x[/bold]", "[/x]", ""])
async def test_streamed_body_and_thinking_are_literal(text: str) -> None:
    message = Message("assistant")
    async with MessageApp(message).run_test() as pilot:
        for chunk in [text[:2], text[2:]]:
            message.append_text(chunk)
            await message.append_thinking(chunk)
        message.finish_thinking()
        thinking = message.query_one(Collapsible)
        thinking.collapsed = False
        await pilot.pause()
        assert_literal(message.query_one(".message-body", Label), text)
        assert_literal(thinking.query_one(Label), text)


@pytest.mark.parametrize(
    "text", ["[on black][black]literal [/x]", r"literal \[bold] [bold]x[/bold]"]
)
async def test_tool_result_and_preview_title_remain_literal(text: str) -> None:
    message = Message("assistant")
    async with MessageApp(message).run_test() as pilot:
        await message.append_tool_call(text, text, text)
        collapsible = message.query_one(Collapsible)
        collapsible.collapsed = False
        await pilot.pause()
        assert_literal(collapsible.query_one(Label), f"Arguments: {text}\n\nResult:\n{text}")
        assert collapsible._title.label.plain == f"⚙ {text} → {text}"
        assert not collapsible._title.label.spans


@pytest.mark.parametrize("button", ["allow-once", "deny-once", "allow-session", "allow-always"])
async def test_confirmation_preserves_text_and_button_choices(button: str) -> None:
    name = "shell_[bold]execute[/bold]"
    arguments = '  {"command": "[on black][black]rm -rf ~[/]", "escape": "\\n"}\n[/x]  '
    app = MessageApp(Message("assistant"))
    results: list[str | None] = []
    async with app.run_test(size=(60, 24)) as pilot:
        modal = ToolConfirmationModal(name, arguments)
        await app.push_screen(modal, results.append)
        await pilot.pause()
        assert_literal(
            modal.query_one(".message", Label), f"The model wants to execute the tool: {name}"
        )
        assert_literal(modal.query_one("#tool-arguments", Label), arguments)
        choice = modal.query_one(f"#{button}", Button)
        assert modal.region.contains_region(choice.region)
        assert await pilot.click(choice)
        await pilot.pause()
        assert results == [button]


@pytest.mark.parametrize("arguments", ["line [/x]\n" * 100 + "END", "x" * 6000 + "END"])
@pytest.mark.parametrize("size", [(60, 24), (100, 40)])
async def test_long_arguments_scroll_without_hiding_controls(
    arguments: str, size: tuple[int, int]
) -> None:
    app = MessageApp(Message("assistant"))
    async with app.run_test(size=size) as pilot:
        modal = ToolConfirmationModal("long-tool-name-" * 50, arguments)
        await app.push_screen(modal)
        await pilot.pause()
        viewport = modal.query_one("#tool-arguments-scroll", ScrollableContainer)
        label = modal.query_one("#tool-arguments", Label)
        assert_literal(label, arguments)
        assert viewport.max_scroll_y > 0
        assert viewport.max_scroll_x == 0
        viewport.scroll_end(animate=False)
        await pilot.pause()
        assert viewport.scroll_y == viewport.max_scroll_y
        assert label.region.bottom <= viewport.content_region.bottom
        assert label.region.bottom > viewport.content_region.y
        for button in modal.query(Button):
            assert modal.region.contains_region(button.region)
            assert button.region.y >= viewport.region.bottom
        assert await pilot.click("#deny-once")


def chat_app(tmp_path: Path, **kwargs: Any) -> ChatApp:
    return ChatApp("http://127.0.0.1:1", "model", tmp_path, tmp_path / "config.toml", **kwargs)


async def test_replayed_messages_and_thinking_are_literal(tmp_path: Path) -> None:
    text = "a[i] [bold]x[/bold] [/x]"
    session = Session(
        "model",
        "session",
        [
            {"role": "user", "content": text},
            {"role": "assistant", "content": text, "thinking": text},
        ],
    )
    app = chat_app(tmp_path, resume_session=session)
    async with app.run_test() as pilot:
        await pilot.pause()
        messages = list(app.query(Message))
        assert len(messages) == 2
        for message in messages:
            assert_literal(message.query_one(".message-body", Label), text)
        thinking = messages[1].query_one(Collapsible)
        thinking.collapsed = False
        await pilot.pause()
        assert_literal(thinking.query_one(Label), text)


async def test_live_messages_and_stream_error_are_literal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "a[i] [bold]x[/bold] [/x]"

    async def stream(
        self: ChatApp, messages: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, str]]:
        yield {"kind": "thinking", "chunk": text}
        yield {"kind": "content", "chunk": text[:4]}
        yield {"kind": "content", "chunk": text[4:]}
        raise RuntimeError(text)

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    app = chat_app(tmp_path)
    async with app.run_test() as pilot:
        await app._send_message(text).wait()
        await pilot.pause()
        for role in ["user", "assistant", "error"]:
            assert_literal(app.query_one(f"Message.{role} .message-body", Label), text)
        assert_literal(app.query_one(".thinking-section Label", Label), text)


async def test_status_values_are_literal(tmp_path: Path) -> None:
    app = chat_app(tmp_path)
    app._model = "model[bold]x[/bold]"
    app._base_url = "http://[::1]:8080/[/x]"
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one("#status-label", Label)
        assert_literal(
            status, f"  model: {app._model}  •  {app._base_url}  •  session: {app._session.id}"
        )


async def test_approved_arguments_match_executed_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = ' {"command": "[on black][black]echo literal[/]", "text": "\\n"}\n'
    requests = 0
    executed: list[tuple[str, str]] = []

    async def stream(
        self: ChatApp, messages: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "call",
                "name": "shell_execute",
                "arguments": arguments,
            }
        else:
            yield {"kind": "content", "chunk": "Done"}

    def execute(name: str, args: str) -> str:
        executed.append((name, args))
        return "test result"

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", execute)
    app = chat_app(tmp_path)
    async with app.run_test() as pilot:
        worker = app._send_message("test")
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ToolConfirmationModal)
        assert_literal(modal.query_one("#tool-arguments", Label), arguments)
        assert executed == []
        assert await pilot.click("#allow-once")
        await worker.wait()
        assert executed == [("shell_execute", arguments)]
