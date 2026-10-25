"""Chat presentation regressions using real headless Textual widgets."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from textual.widgets import Label, Markdown
from textual.widgets.markdown import MarkdownFence
from textual.worker import Worker

from drove.config import Config
from drove.sessions import Session, load_session
from drove.tui import ChatApp, ChatInput, CodePicker, Message, _http_error


@pytest.fixture
def app(tmp_path: Path) -> ChatApp:
    return ChatApp("http://127.0.0.1:8080", "model", tmp_path / "sessions", None)


def start_turn(app: ChatApp, text: str) -> Worker[None]:
    app._start_turn(text)
    assert app._generation_worker is not None
    return app._generation_worker


def plain(label: Label) -> str:
    return label.render().plain


def assistant(app: ChatApp) -> Message:
    return list(app.query("Message.assistant"))[-1]


async def test_stream_formats_before_eof_and_matches_replay(app, monkeypatch):
    waiting = asyncio.Event()
    release = asyncio.Event()
    source = "# Heading\n\n- first\n- second\n\n```python\na[i] = '[red]literal[/red]'  \n```\n"

    async def stream(self, history):
        yield {"kind": "thinking", "chunk": "a[i] [/x]"}
        for chunk in (
            "# Heading\n\n- first\n- second\n\n`",
            "``python\na[i] = '",
            "[red]literal[/red]'  \n```\n",
        ):
            yield {"kind": "content", "chunk": chunk}
        waiting.set()
        await release.wait()

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test() as pilot:
        worker = start_turn(app, "explain a[i]")
        await asyncio.wait_for(waiting.wait(), 5)
        await pilot.pause()
        bubble = assistant(app)
        body = bubble.query_one(Markdown)
        assert body.source == source
        assert len(body.query("MarkdownH1")) == 1
        assert len(body.query("MarkdownBullet")) == 2
        fence = body.query_one(MarkdownFence)
        assert fence.lexer == "python"
        assert "[red]literal[/red]" in fence._highlighted_code.plain
        assert plain(bubble.query_one(".thinking-section Label", Label)) == "a[i] [/x]"
        release.set()
        await worker.wait()
        assert bubble._markdown_stream is None
        saved = load_session(next(app._sessions_dir.rglob("*.json")))
        assert saved.messages[-1]["content"] == source
        await app._load_session(saved)
        await pilot.pause()
        replay = assistant(app).query_one(Markdown)
        assert replay.source == source
        assert len(replay.query("MarkdownH1")) == 1
        assert len(replay.query("MarkdownBullet")) == 2
        assert replay.query_one(MarkdownFence).lexer == "python"


@pytest.mark.parametrize("role", ["user", "error", "system-note"])
async def test_nonassistant_text_is_literal(app, role):
    text = "# heading **bold** a[i] [red]x[/red] [/x] ```python"
    async with app.run_test():
        bubble = Message(role, text)
        await app.query_one("#messages").mount(bubble)
        label = bubble.query_one(".message-body", Label)
        assert plain(label) == text
        assert not bubble.query(Markdown)
        await bubble.append_text(" [bold]tail[/bold]")
        assert plain(label) == text + " [bold]tail[/bold]"


async def test_burst_flush_preserves_final_fragment_and_unknown_fence(app):
    source = "# Burst\n\n```unknown-language\n" + "a[i] = 1\n" * 200 + "```\nlast"
    async with app.run_test() as pilot:
        bubble = Message("assistant")
        await app.query_one("#messages").mount(bubble)
        for index in range(0, len(source), 7):
            await bubble.append_text(source[index : index + 7])
        await bubble.finish_text()
        await pilot.pause()
        assert bubble.query_one(Markdown).source == source
        assert bubble._markdown_stream is None
        assert "a[i] = 1" in bubble.query_one(MarkdownFence)._content.plain


@pytest.mark.parametrize("finish", ["error", "cancel"])
async def test_stream_cleanup_on_failure_or_cancellation(app, monkeypatch, finish):
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def stream(self, history):
        yield {"kind": "content", "chunk": "# Partial\n\n```python\na[i]"}
        waiting.set()
        await release.wait()
        raise httpx.ReadError("connection [lost]")

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test() as pilot:
        worker = start_turn(app, "go")
        await asyncio.wait_for(waiting.wait(), 5)
        if finish == "cancel":
            worker.cancel()
            await worker.wait()
            assert app._history[-1]["stopped"] is True
        else:
            release.set()
            await worker.wait()
            assert "connection [lost]" in plain(app.query_one(".error .message-body", Label))
        await pilot.pause()
        bubble = assistant(app)
        assert bubble._markdown_stream is None
        assert bubble.query_one(Markdown).source.endswith("a[i]")


async def test_removed_message_stops_its_markdown_stream(app):
    async with app.run_test():
        bubble = Message("assistant")
        await app.query_one("#messages").mount(bubble)
        await bubble.append_text("a[i] [bold]literal[/bold]")
        await bubble.remove()
        assert bubble._markdown_stream is None


async def test_empty_assistant_and_thinking_only_response(app, monkeypatch):
    async def stream(self, history):
        yield {"kind": "thinking", "chunk": "[bold]x[/bold] [/x]"}

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test():
        await start_turn(app, "think").wait()
        bubble = assistant(app)
        assert bubble.query_one(Markdown).source == ""
        assert plain(bubble.query_one(".thinking-section Label", Label)) == "[bold]x[/bold] [/x]"
        assert bubble._markdown_stream is None


async def test_tool_loop_flushes_each_assistant_body(app, monkeypatch):
    count = 0

    async def stream(self, history):
        nonlocal count
        count += 1
        if count == 1:
            yield {"kind": "content", "chunk": "# Before tool"}
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "id",
                "name": "read_file",
                "arguments": "{}",
            }
        else:
            yield {"kind": "content", "chunk": "# After tool"}

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args: "safe result")
    async with app.run_test():
        await start_turn(app, "go").wait()
        bubbles = list(app.query("Message.assistant"))
        assert [b.query_one(Markdown).source for b in bubbles] == ["# Before tool", "# After tool"]
        assert all(b._markdown_stream is None for b in bubbles)


async def test_keyboard_copy_raw_and_select_second_code_block(app, monkeypatch):
    copied = Mock()
    monkeypatch.setattr(app, "copy_to_clipboard", copied)
    source = "# Hi\n\n```python\nfirst()\n```\n\n~~~text\n  [red]second[/red]  \n\n~~~\n"
    async with app.run_test() as pilot:
        bubble = Message("assistant", source)
        await app.query_one("#messages").mount(bubble)
        bubble.focus()
        await pilot.press("ctrl+y")
        copied.assert_called_once_with(source)
        await pilot.press("ctrl+b")
        assert isinstance(app.screen, CodePicker)
        await pilot.press("down", "enter")
        copied.assert_called_with("  [red]second[/red]  \n\n")
        assert app.focused is bubble


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("```python\na[i] = 1  \npartial", "a[i] = 1  \npartial"),
        ("    indented()  \n    next()\n", "indented()  \nnext()\n"),
        ("- item\n\n  ```text\n  nested [bold]  \n  ```\n", "nested [bold]  \n"),
        ("````text\n```\ninside\n```\n````\n", "```\ninside\n```\n"),
    ],
)
async def test_copy_parser_preserves_code_semantics(app, monkeypatch, source, expected):
    copied = Mock()
    monkeypatch.setattr(app, "copy_to_clipboard", copied)
    async with app.run_test() as pilot:
        bubble = Message("assistant")
        await app.query_one("#messages").mount(bubble)
        await bubble.append_text(source)
        bubble.focus()
        await pilot.press("ctrl+b", "enter")
        copied.assert_called_once_with(expected)
        await bubble.finish_text()


async def test_copy_no_block_cancel_and_input_scope(app, monkeypatch):
    copied = Mock()
    monkeypatch.setattr(app, "copy_to_clipboard", copied)
    async with app.run_test() as pilot:
        bubble = Message("assistant", "no block")
        await app.query_one("#messages").mount(bubble)
        bubble.focus()
        await pilot.press("ctrl+b")
        assert not isinstance(app.screen, CodePicker)
        await bubble.append_text("\n```text\ncode\n```")
        await pilot.press("ctrl+b", "escape")
        copied.assert_not_called()
        assert app.focused is bubble
        app.query_one(ChatInput).focus()
        await pilot.press("y", "c", "ctrl+y", "ctrl+b")
        assert not isinstance(app.screen, CodePicker)
        copied.assert_not_called()
        await bubble.finish_text()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"detail":"Model [red]missing[/red]"}', "Model [red]missing[/red]"),
        (b'{"error":{"message":"Model load failed"}}', "Model load failed"),
        (b'{"error":"bad input"}', "bad input"),
        (b'{"message":"try later"}', "try later"),
        (b'{"detail":[{"secret":"hidden"}]}', None),
        (b'{"error":{"secret":"hidden"}}', None),
        (b"<html>hidden</html>", None),
        (b"[]", None),
        (b'"hidden"', None),
        (b"", None),
    ],
)
async def test_proxy_errors_show_message_not_envelope(app, monkeypatch, body, expected):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503, content=body))
    monkeypatch.setattr(
        "drove.tui.httpx.AsyncClient", lambda **kwargs: client_class(transport=transport, **kwargs)
    )
    async with app.run_test():
        await start_turn(app, "go").wait()
        error = plain(app.query_one(".error .message-body", Label))
        assert "HTTP 503" in error and "drove server status" in error
        assert "hidden" not in error
        if expected:
            assert expected in error
        assert not app.is_generating


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_http_error_advice(status):
    error = _http_error(status, b"not json")
    assert "not json" not in error
    expected = (
        "API key" if status in (401, 403) else "models list" if status == 404 else "server status"
    )
    assert expected in error


@pytest.mark.parametrize(
    "exception",
    [httpx.ConnectError("raw transport"), httpx.ReadTimeout("raw timeout"), RuntimeError("")],
)
async def test_transport_errors_are_actionable(app, monkeypatch, exception):
    async def stream(self, history):
        raise exception
        yield  # pragma: no cover

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    app._last_speed = 10
    async with app.run_test():
        await start_turn(app, "go").wait()
        error = plain(app.query_one(".error .message-body", Label))
        assert "drove server status" in error
        assert "\n" not in error
        assert "raw transport" not in error and "raw timeout" not in error
        assert not app.is_generating and app._last_speed is None


@pytest.mark.parametrize("elapsed", [0, 2])
async def test_speed_is_received_chunk_rate_not_tokens(app, monkeypatch, elapsed):
    clock = Mock(monotonic=Mock(side_effect=[10.0, 11.0, 11.0 + elapsed]))
    monkeypatch.setattr("drove.tui.time", clock)

    async def stream(self, history):
        yield {"kind": "thinking", "chunk": "many tokens in one chunk"}
        yield {"kind": "content", "chunk": "several more words"}
        yield {"kind": "content", "chunk": "last"}

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test():
        await start_turn(app, "go").wait()
        status = plain(app.query_one("#status-label", Label))
        assert "tok/s" not in status
        if elapsed:
            assert app._last_speed == 1.5
            assert "1.5 chunks/s (received)" in status
        else:
            assert app._last_speed is None


@pytest.mark.parametrize("failure", ["load", "save", "none", "session"])
async def test_theme_feedback_matches_persistence(app, monkeypatch, tmp_path, failure):
    app._config_path = None if failure == "session" else tmp_path / "config.toml"
    if failure == "load":
        monkeypatch.setattr("drove.config.load_config", Mock(side_effect=OSError("denied")))
    elif failure == "save":
        monkeypatch.setattr(Config, "save", Mock(side_effect=OSError("denied")))
    async with app.run_test():
        await app._dispatch_command("/theme textual-light")
        note = plain(list(app.query(".system-note .message-body"))[-1])
        assert app.theme == "textual-light"
        if failure in ("load", "save"):
            assert "Warning:" in note and "this session" in note
            assert "and saved" not in note
        elif failure == "session":
            assert "not saved" in note
        else:
            assert "and saved" in note
            assert 'tui_theme = "textual-light"' in app._config_path.read_text()


async def test_assistant_markup_like_text_is_literal_not_textual_styles(app):
    async with app.run_test() as pilot:
        bubble = Message("assistant", "[bold]x[/bold] [/x] <script>literal</script>")
        await app.query_one("#messages").mount(bubble)
        await pilot.pause()
        paragraph = bubble.query_one("MarkdownParagraph")
        assert paragraph.render().plain == "[bold]x[/bold] [/x] <script>literal</script>"


async def test_copy_replayed_message_preserves_original_source(app, monkeypatch):
    source = "# Stored\n\n```text\ntrailing  \n```\n"
    session = Session(
        "model",
        "saved",
        [{"role": "user", "content": "go"}, {"role": "assistant", "content": source}],
    )
    copied = Mock()
    monkeypatch.setattr(app, "copy_to_clipboard", copied)
    async with app.run_test() as pilot:
        await app._load_session(session)
        bubble = assistant(app)
        bubble.focus()
        await pilot.press("ctrl+y")
        copied.assert_called_once_with(source)
        await pilot.press("ctrl+b", "enter")
        copied.assert_called_with("trailing  \n")


async def test_tool_only_events_do_not_claim_content_throughput(app, monkeypatch):
    count = 0

    async def stream(self, history):
        nonlocal count
        count += 1
        if count == 1:
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "id",
                "name": "read_file",
                "arguments": "{}",
            }

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args: "safe result")
    async with app.run_test():
        await start_turn(app, "go").wait()
        assert app._last_speed is None
        assert "chunks/s" not in plain(app.query_one("#status-label", Label))


async def test_invalid_theme_retains_existing_selection(app):
    async with app.run_test():
        original = app.theme
        await app._dispatch_command("/theme does-not-exist")
        assert app.theme == original
        assert "Unknown theme" in plain(app.query_one(".system-note .message-body", Label))


async def test_markdown_burst_follows_bottom_after_final_layout(app, monkeypatch):
    async def stream(self, history):
        yield {"kind": "content", "chunk": "# Burst\n\n" + "- item\n" * 80}

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test(size=(80, 24)) as pilot:
        await start_turn(app, "go").wait()
        await pilot.pause()
        container = app.query_one("#messages")
        assert container.max_scroll_y > 50
        assert container.scroll_y == container.max_scroll_y


@pytest.mark.parametrize("manual_scroll", [False, True])
async def test_paced_markdown_follows_only_until_user_scrolls(app, monkeypatch, manual_scroll):
    rendered = asyncio.Event()
    next_chunk = asyncio.Queue()

    async def stream(self, history):
        yield {"kind": "content", "chunk": "# Begin\n\n" + "- first\n" * 50}
        rendered.set()
        for index in range(5):
            await next_chunk.get()
            yield {"kind": "content", "chunk": f"\n## Section {index}\n\n" + "line\n\n" * 4}

    monkeypatch.setattr(ChatApp, "_stream_chat", stream)
    async with app.run_test(size=(80, 24)) as pilot:
        worker = start_turn(app, "go")
        await asyncio.wait_for(rendered.wait(), 5)
        await pilot.pause()
        container = app.query_one("#messages")
        assert container.max_scroll_y > 20
        if manual_scroll:
            container.scroll_to(y=0, animate=False, immediate=True)
        for _ in range(5):
            next_chunk.put_nowait(None)
            await pilot.pause()
            if manual_scroll:
                assert container.scroll_y == 0
            else:
                assert container.scroll_y == container.max_scroll_y
        await worker.wait()
        await pilot.pause()
        assert container.scroll_y == (0 if manual_scroll else container.max_scroll_y)
