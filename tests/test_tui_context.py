"""The header context meter: fill level, fallbacks, warning and resets."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Label

from drove.chat_models import ChatModel
from drove.model_picker import ModelPicker
from drove.sessions import new_session
from drove.tui import ChatApp, _compact_tokens


def chat_app(tmp_path: Path) -> ChatApp:
    return ChatApp("http://localhost:8080", "test-model", tmp_path, None)


def known_size(monkeypatch, length: int | None, seen: list | None = None) -> None:
    async def lookup(base_url, api_key, model):
        if seen is not None:
            seen.append(model)
        return length

    monkeypatch.setattr("drove.tui.fetch_context_length", lookup)


def replies(*usages):
    """A fake stream per request, each reporting the next usage (None = no usage)."""
    pending = list(usages)

    async def stream(messages):
        usage = pending.pop(0)
        yield {"kind": "content", "chunk": "ok"}
        if usage is not None:
            yield {"kind": "usage", "usage": usage}

    return stream


def usage(prompt: int, completion: int) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def meter(app: ChatApp) -> Label:
    return app.query_one("#ctx-meter", Label)


async def reply(app: ChatApp, pilot, text: str = "hi") -> None:
    app._start_turn(text)
    async with asyncio.timeout(3):
        while app.is_generating:
            await pilot.pause()
    await pilot.pause()


async def test_meter_is_a_bar_with_exact_counts_in_tooltip_and_info(tmp_path, monkeypatch):
    known_size(monkeypatch, 8000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(3000, 200)))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert not meter(app).has_class("visible")
        await reply(app, pilot)
        assert meter(app).has_class("visible")
        text = meter(app).render().plain
        assert text == "▰" * 4 + "▱" * 6
        assert not any(char.isdigit() for char in text) and "%" not in text
        assert meter(app).tooltip == "3,200 / 8,000 tokens"
        assert not meter(app).has_class("warn")
        assert "Context: 3,200 / 8,000 tokens (40%)" in app._details_text()
        header = app.query_one("#chat-header")
        assert meter(app).region.right == header.content_region.right


@pytest.mark.parametrize("size", [(44, 20), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_meter_fits_beside_the_header_controls(tmp_path, monkeypatch, size, theme):
    known_size(monkeypatch, 1000)
    app = ChatApp("http://localhost:8080", "test-model", tmp_path, None, theme=theme)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(500, 0)))
    async with app.run_test(size=size) as pilot:
        await reply(app, pilot)
        for selector in ("#ctx-meter", "#chat-details", "#new-chat"):
            control = app.query_one(selector)
            assert control.region.area > 0, selector
            assert app.screen.region.contains_region(control.region), selector
        assert app.query_one("#chat-details").region.right < meter(app).region.x


@pytest.mark.parametrize(
    ("used", "warn", "filled"),
    [(790, False, 8), (800, True, 8), (1000, True, 10), (1300, True, 10)],
)
async def test_near_full_context_is_marked_and_overflow_keeps_real_counts(
    tmp_path, monkeypatch, used, warn, filled
):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(used, 0)))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).has_class("warn") is warn
        assert meter(app).render().plain == "▰" * filled + "▱" * (10 - filled)
        assert meter(app).tooltip == f"{used:,} / 1,000 tokens"
        assert f"({used / 1000:.0%})" in app._details_text()


async def test_unknown_size_shows_the_raw_count_never_a_bar(tmp_path, monkeypatch):
    known_size(monkeypatch, None)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(3000, 200)))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).render().plain == "ctx 3.2k"
        assert "▰" not in meter(app).render().plain
        assert "3,200" in str(meter(app).tooltip)
        assert "Context: 3,200 tokens (context size unknown)" in app._details_text()


@pytest.mark.parametrize(
    "reported",
    [
        None,
        {"prompt_tokens": 10},
        {"prompt_tokens": "10", "completion_tokens": 1},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": -1, "completion_tokens": 1},
    ],
)
@pytest.mark.parametrize("length", [None, 1000])
async def test_missing_or_partial_usage_hides_the_meter(tmp_path, monkeypatch, reported, length):
    known_size(monkeypatch, length)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(reported))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert not meter(app).has_class("visible")
        assert "Context: Not reported" in app._details_text()


async def test_known_zero_usage_is_an_empty_bar(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(0, 0)))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).has_class("visible")
        assert meter(app).render().plain == "▱" * 10


async def test_each_reply_replaces_usage_and_stays_visible_while_streaming(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    gate = asyncio.Event()
    calls = 0

    async def stream(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {"kind": "usage", "usage": usage(300, 0)}
        elif calls == 2:
            await gate.wait()
            yield {"kind": "usage", "usage": usage(500, 0)}
        elif calls == 3:
            # A tool round reports usage; the follow-up response does not.
            yield {"kind": "usage", "usage": usage(900, 0)}
            yield {
                "kind": "tool_call",
                "index": 0,
                "id": "t",
                "name": "read_file",
                "arguments": "{}",
            }
        else:
            yield {"kind": "content", "chunk": "done"}

    monkeypatch.setattr(app, "_stream_chat", stream)
    monkeypatch.setattr("drove.tui.execute_tool", lambda *args, **kwargs: "file")
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).tooltip == "300 / 1,000 tokens"
        app._start_turn("second")
        await pilot.pause()
        assert app.is_generating
        assert meter(app).tooltip == "300 / 1,000 tokens"  # the last known fill stays up
        gate.set()
        async with asyncio.timeout(3):
            while app.is_generating:
                await pilot.pause()
        assert meter(app).tooltip == "500 / 1,000 tokens"  # replaced, not summed
        await reply(app, pilot, "third")
        assert calls == 4
        assert not meter(app).has_class("visible")  # never the stale tool-round usage


async def test_stopped_reply_clears_the_meter(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    streaming = asyncio.Event()
    calls = 0

    async def stream(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield {"kind": "usage", "usage": usage(300, 0)}
            return
        streaming.set()
        yield {"kind": "content", "chunk": "par"}
        await asyncio.Event().wait()

    monkeypatch.setattr(app, "_stream_chat", stream)
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).has_class("visible")
        app._start_turn("again")
        await asyncio.wait_for(streaming.wait(), 2)
        app.action_stop_generation()
        async with asyncio.timeout(3):
            while app.is_generating:
                await pilot.pause()
        assert not meter(app).has_class("visible")


@pytest.mark.parametrize("command", ["/new", "/clear"])
async def test_new_conversation_resets_the_meter(tmp_path, monkeypatch, command):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(300, 0)))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        assert meter(app).has_class("visible")
        await app._dispatch_command(command)
        await pilot.pause()
        assert not meter(app).has_class("visible")
        assert app._context_length == 1000  # same model, same size


async def test_session_resume_resets_the_meter(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(300, 0)))
    async with app.run_test() as pilot:
        await reply(app, pilot)
        other = new_session("test-model")
        other.messages.append({"role": "user", "content": "earlier"})
        await app._load_session(other)
        await pilot.pause()
        assert not meter(app).has_class("visible")


async def test_startup_resume_shows_no_meter_until_a_reply(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    session = new_session("test-model")
    session.messages.append({"role": "user", "content": "earlier"})
    app = ChatApp("http://localhost:8080", "test-model", tmp_path, None, resume_session=session)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._context_length == 1000
        assert not meter(app).has_class("visible")


async def test_model_switch_resets_usage_and_uses_the_new_models_size(tmp_path, monkeypatch):
    known_size(monkeypatch, 1000)
    app = chat_app(tmp_path)
    monkeypatch.setattr(app, "_stream_chat", replies(usage(300, 0), usage(300, 0)))
    models = [ChatModel("test-model", context_length=1000), ChatModel("other", context_length=600)]
    async with app.run_test() as pilot:
        await reply(app, pilot)
        with patch("drove.tui.fetch_models_async", return_value=models):
            worker = app.run_worker(app._dispatch_command("/model"))
            await pilot.pause()
            assert isinstance(app.screen, ModelPicker)
            await pilot.press("down", "enter")
            await worker.wait()
        assert app._model == "other"
        assert app._context_length == 600
        assert not meter(app).has_class("visible")
        await reply(app, pilot)
        assert meter(app).tooltip == "300 / 600 tokens"


async def test_late_size_for_a_former_model_is_ignored(tmp_path, monkeypatch):
    release = asyncio.Event()
    seen: list[str] = []

    async def slow(base_url, api_key, model):
        seen.append(model)
        await release.wait()
        return 999

    monkeypatch.setattr("drove.tui.fetch_context_length", slow)
    app = chat_app(tmp_path)
    models = [ChatModel("test-model"), ChatModel("other", context_length=600)]
    async with app.run_test() as pilot:
        await pilot.pause()
        assert seen == ["test-model"]
        with patch("drove.tui.fetch_models_async", return_value=models):
            worker = app.run_worker(app._dispatch_command("/model"))
            await pilot.pause()
            await pilot.press("down", "enter")
            await worker.wait()
        assert app._model == "other"
        release.set()
        await pilot.pause()
        await pilot.pause()
        assert app._context_length == 600


async def test_unreachable_catalog_leaves_the_size_unknown(tmp_path, monkeypatch):
    from drove import chat_models

    async def offline(base_url, api_key):
        raise chat_models.CatalogError("offline")

    monkeypatch.setattr(chat_models, "fetch_models_async", offline)
    assert await chat_models.fetch_context_length("http://x", None, "m") is None

    async def listed(base_url, api_key):
        return [ChatModel("m", context_length=4096), ChatModel("n")]

    monkeypatch.setattr(chat_models, "fetch_models_async", listed)
    assert await chat_models.fetch_context_length("http://x", None, "m") == 4096
    assert await chat_models.fetch_context_length("http://x", None, "n") is None
    assert await chat_models.fetch_context_length("http://x", None, "absent") is None


@pytest.mark.parametrize(
    ("count", "text"),
    [(0, "0"), (999, "999"), (3200, "3.2k"), (128000, "128.0k"), (1_500_000, "1.5M")],
)
def test_compact_token_counts(count, text):
    assert _compact_tokens(count) == text
