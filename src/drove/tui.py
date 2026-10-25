"""Terminal UI for chatting with a running drove server."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from markdown_it import MarkdownIt
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Label,
    LoadingIndicator,
    Markdown,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.markdown import MarkdownStream
from textual.widgets.option_list import Option
from textual.worker import Worker

from drove.chat_models import CatalogError, fetch_models_async
from drove.model_picker import ModelPicker
from drove.sessions import Session, list_sessions, new_session, save_session
from drove.tools import TOOL_DEFINITIONS, execute_tool

# ── Styles ─────────────────────────────────────────────────────────────────────

CSS = """
ChatApp {
    background: $surface;
}

#messages {
    height: 1fr;
    padding: 1 2;
    overflow-y: auto;
    overflow-x: hidden;
}

.message {
    margin-bottom: 1;
    width: 100%;
}

.message:focus {
    border-left: solid $accent;
}

.message-header {
    text-style: bold;
    margin-bottom: 0;
}

.queued { color: $text-muted; }
.queued Label { color: $text-muted; }
.queued Button { margin-top: 1; }
#resume-queue { display: none; }
#resume-queue.visible { display: block; }

.user .message-header { color: $accent; }
.assistant .message-header { color: $success; }
.error .message-header { color: $error; }
.system-note .message-header { color: $warning; }

.message-body {
    padding-left: 2;
    color: $text;
    width: 1fr;
}

.error .message-body { color: $error; }
.system-note .message-body { color: $text-muted; text-style: italic; }

.thinking-section {
    margin-bottom: 0;
    padding-left: 2;
}

.thinking-section Label {
    color: $text-muted;
    text-style: italic;
    width: 1fr;
}

.tool-call-section {
    margin-bottom: 0;
    padding-left: 2;
}

.tool-call-section Label {
    color: $text-muted;
    width: 1fr;
}

#autocomplete {
    display: none;
    height: auto;
    max-height: 10;
    margin: 0 2;
    background: $panel;
    border: solid $accent;
}

#autocomplete.visible {
    display: block;
}

#input-bar {
    height: auto;
    padding: 1 2;
    border-top: solid $panel;
}

#user-input {
    width: 1fr;
    height: auto;
    min-height: 3;
    max-height: 12;
}

#status-bar {
    height: 1;
    padding: 0 2;
    background: $panel;
    color: $text-muted;
}

LoadingIndicator {
    height: 1;
    width: auto;
    display: none;
}

LoadingIndicator.visible { display: block; }

ToolConfirmationModal {
    align: center middle;
}

#tool-dialog {
    width: 90%;
    max-width: 80;
    height: 90%;
    max-height: 40;
    background: $surface;
    border: solid $accent;
    padding: 1 2;
}

#tool-dialog Label.title {
    text-style: bold;
    margin-bottom: 1;
    color: $accent;
}

#tool-dialog Label.message {
    margin-bottom: 1;
}

#tool-arguments-scroll {
    height: 1fr;
    overflow-x: hidden;
}

#tool-arguments {
    width: 1fr;
}

#tool-dialog .buttons {
    margin-top: 1;
    height: auto;
    align: center middle;
}

#tool-dialog Button {
    margin: 0 1;
}

CodePicker {
    align: center middle;
}

#code-dialog {
    width: 80%;
    height: 60%;
    background: $surface;
    border: solid $accent;
    padding: 1 2;
}

#code-table {
    height: 1fr;
}

/* Session picker modal */
SessionPicker {
    align: center middle;
}

#session-dialog {
    width: 80%;
    max-height: 70%;
    background: $surface;
    border: solid $accent;
    padding: 1 2;
}

#session-dialog Label.title {
    text-style: bold;
    margin-bottom: 1;
    color: $accent;
}

#session-table {
    height: 1fr;
}
"""


# ── Chat input ─────────────────────────────────────────────────────────────────


class ChatInput(TextArea):
    """TextArea that sends Enter as submit and Shift+Enter as newline."""

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
    ]

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    class Submitted(TextArea.Changed):
        pass

    def _on_key(self, event: events.Key) -> None:  # type: ignore[override]
        if event.key == "shift+enter":
            self.insert("\n")
            event.prevent_default()
            event.stop()


# ── Message widget ──────────────────────────────────────────────────────────────


def _markdown_parser() -> MarkdownIt:
    """Use identical Markdown semantics for display and exact code copying."""
    return MarkdownIt("gfm-like", {"html": False})


class Message(Static, can_focus=True):
    BINDINGS = [
        Binding("ctrl+y", "copy_message", "Copy message"),
        Binding("ctrl+b", "copy_code", "Copy code block"),
    ]

    def __init__(self, role: str, content: str = "", *, stopped: bool = False) -> None:
        super().__init__()
        self._role = role
        self._stopped = stopped
        self._content = content
        self._thinking = ""
        self._has_thinking_widget = False
        self._markdown_stream: MarkdownStream | None = None
        self.add_class("message", role)

    def compose(self) -> ComposeResult:
        labels = {
            "user": "You",
            "assistant": "Assistant",
            "error": "Error",
            "system-note": "System",
        }
        heading = labels.get(self._role, self._role)
        if self._stopped:
            heading += " · Stopped"
        yield Label(heading, classes="message-header", markup=False)
        if self._role == "assistant":
            yield Markdown(
                self._content,
                parser_factory=_markdown_parser,
                open_links=False,
                classes="message-body",
                id=f"body-{id(self)}",
            )
        else:
            yield Label(self._content, markup=False, classes="message-body", id=f"body-{id(self)}")

    def mark_stopped(self) -> None:
        self._stopped = True
        self.query_one(".message-header", Label).update("Assistant · Stopped")

    async def append_text(self, chunk: str) -> None:
        self._content += chunk
        if self._role == "assistant":
            if self._markdown_stream is None:
                body = self.query_one(f"#body-{id(self)}", Markdown)
                self._markdown_stream = Markdown.get_stream(body)
            await self._markdown_stream.write(chunk)
        else:
            self.query_one(f"#body-{id(self)}", Label).update(self._content)

    async def finish_text(self) -> None:
        """Flush the final fragment and release the buffered Markdown updater."""
        stream, self._markdown_stream = self._markdown_stream, None
        if stream is not None:
            await stream.stop()

    async def on_unmount(self) -> None:
        await self.finish_text()

    def action_copy_message(self) -> None:
        self.app.copy_to_clipboard(self._content)
        self.notify("Message sent to terminal clipboard.")

    def action_copy_code(self) -> None:
        blocks = [
            (token.info, token.content)
            for token in _markdown_parser().parse(self._content)
            if token.type in {"fence", "code_block"}
        ]
        if not blocks:
            self.notify("This message has no code blocks.")
            return

        def copy_selected(code: str | None) -> None:
            if code is not None:
                self.app.copy_to_clipboard(code)
                self.notify("Code block sent to terminal clipboard.")
            self.focus()

        self.app.push_screen(CodePicker(blocks), copy_selected)

    async def append_thinking(self, chunk: str) -> None:
        self._thinking += chunk
        if not self._has_thinking_widget:
            self._has_thinking_widget = True
            collapsible = Collapsible(
                Label(self._thinking, id=f"thinking-{id(self)}", markup=False),
                title="Thinking…",
                collapsed=True,
                classes="thinking-section",
            )
            # Insert before either the Markdown or literal body
            try:
                body = self.query_one(f"#body-{id(self)}")
                await self.mount(collapsible, before=body)
            except NoMatches:
                await self.mount(collapsible)
        else:
            try:
                self.query_one(f"#thinking-{id(self)}", Label).update(self._thinking)
            except NoMatches:
                pass

    def finish_thinking(self) -> None:
        """Update the collapsible title to show final state."""
        if self._has_thinking_widget:
            try:
                collapsible = self.query_one(".thinking-section", Collapsible)
                collapsible.title = f"Thinking ({len(self._thinking)} chars)"
            except NoMatches:
                pass

    async def append_tool_call(self, name: str, arguments: str, result: str) -> None:
        """Add a collapsible tool call section after the body."""
        result_preview = result[:200] + ("…" if len(result) > 200 else "")
        collapsible = Collapsible(
            Label(f"Arguments: {arguments}\n\nResult:\n{result}", markup=False),
            # Collapsible forwards Content to its title renderer despite its str annotation.
            title=Content(f"⚙ {name} → {result_preview}"),  # type: ignore[arg-type]
            collapsed=True,
            classes="tool-call-section",
        )
        await self.mount(collapsible)


class CodePicker(ModalScreen[str | None]):
    """Choose parsed code from the current raw source, including unfinished fences."""

    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, blocks: list[tuple[str, str]]) -> None:
        super().__init__()
        self._blocks = blocks

    def compose(self) -> ComposeResult:
        with Vertical(id="code-dialog"):
            yield Label("Copy code block — arrows and Enter; Esc cancels", markup=False)
            yield DataTable(id="code-table", cursor_type="row")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Block", "Language", "Preview")
        for index, (language, code) in enumerate(self._blocks, 1):
            preview = code.splitlines()[0] if code else "(empty)"
            table.add_row(str(index), Text(language or "plain"), Text(preview[:70]))
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.dismiss(self._blocks[event.cursor_row][1])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


# ── Session picker modal ────────────────────────────────────────────────────────


class ToolConfirmationModal(ModalScreen[str]):
    """Modal to confirm tool execution."""

    BINDINGS = [
        Binding("y", "choose('allow-once')", "Allow once"),
        Binding("n", "choose('deny-once')", "Deny once"),
        Binding("escape", "choose('deny-once')", "Deny once", show=False),
        Binding("s", "choose('allow-session')", "Allow for session"),
        Binding("a", "choose('allow-always')", "Allow always"),
    ]

    def action_choose(self, choice: str) -> None:
        self.dismiss(choice)

    def __init__(self, tool_name: str, arguments: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-dialog"):
            yield Label("Tool Call Confirmation", classes="title")
            with ScrollableContainer(id="tool-arguments-scroll"):
                yield Label(
                    f"The model wants to execute the tool: {self._tool_name}",
                    classes="message",
                    markup=False,
                )
                yield Label("Arguments:")
                yield Label(self._arguments, id="tool-arguments", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Allow Once (y)", variant="primary", id="allow-once")
                yield Button("Deny Once (n / Esc)", variant="error", id="deny-once")
            with Horizontal(classes="buttons"):
                yield Button("Allow for Session (s)", variant="default", id="allow-session")
                yield Button("Allow Always (a)", variant="default", id="allow-always")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id)


class SessionPicker(ModalScreen[Session | None]):
    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel"),
        Binding("enter", "select", "Load"),
    ]

    def __init__(self, sessions: list[Session]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="session-dialog"):
            yield Label("Load session", classes="title")
            table: DataTable[str] = DataTable(id="session-table", cursor_type="row")
            yield table

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Date", "Turns", "First message")
        for s in self._sessions:
            date = s.updated_at[:16].replace("T", " ")
            table.add_row(date, str(s.message_count), s.title)
        if self._sessions:
            table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        self.dismiss(self._sessions[idx])

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        table = self.query_one(DataTable)
        idx = table.cursor_row
        if 0 <= idx < len(self._sessions):
            self.dismiss(self._sessions[idx])


@dataclass
class QueuedPrompt:
    id: int
    text: str
    bubble: QueuedMessage


class QueuedMessage(Static):
    def __init__(self, prompt_id: int, text: str) -> None:
        super().__init__(classes="message queued")
        self.prompt_id = prompt_id
        self.text = text

    def compose(self) -> ComposeResult:
        yield Label("You · Queued", classes="message-header")
        yield Label(self.text, markup=False, classes="message-body")
        yield Button("Cancel queued message", id=f"cancel-queued-{self.prompt_id}")


@dataclass
class ChatTurn:
    session: Session
    history: list[dict[str, Any]]
    text: str
    started: bool = False
    stop_requested: bool = False
    record: dict[str, Any] = field(default_factory=lambda: {"role": "assistant", "content": ""})
    committed: bool = False
    bubble: Message | None = None
    tool_task: asyncio.Task[str] | None = None


def _http_error(status: int, body: bytes) -> str:
    """Extract known proxy/OpenAI messages without exposing a raw response envelope."""
    try:
        data = json.loads(body)
    except ValueError, UnicodeDecodeError:
        data = None
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        candidates = [
            data.get("detail"),
            error.get("message") if isinstance(error, dict) else error,
            data.get("message"),
        ]
        message = next((value for value in candidates if isinstance(value, str) and value), "")
    prefix = f"Request failed (HTTP {status})"
    explanation = f": {message}" if message else "."
    if status == 404:
        advice = "Check the model with drove models list."
    elif status in (401, 403):
        advice = "Check the endpoint and API key configuration."
    else:
        advice = "Check drove server status and the server logs."
    return f"{prefix}{explanation} {advice}"


def _chat_error(error: Exception, base_url: str) -> str:
    if isinstance(error, httpx.ConnectError):
        return (
            f"Cannot reach drove server at {base_url}. "
            "Try drove server status; start it with drove serve if needed."
        )
    if isinstance(error, httpx.TimeoutException):
        return "The server request timed out. Check drove server status and retry."
    return str(error) or f"Chat request failed ({type(error).__name__}). Check drove server status."


# ── Main app ────────────────────────────────────────────────────────────────────


class ChatApp(App[None]):
    async def push_screen_wait(self, screen: ModalScreen[Any]) -> Any:  # type: ignore[override]
        """Await this modal without leaving a callback or screen behind on cancellation."""
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def dismissed(result: Any) -> None:
            if not future.done():
                future.set_result(result)

        self.push_screen(screen, callback=dismissed)
        try:
            return await future
        finally:
            if screen in self.screen_stack and self.screen is screen:
                screen.dismiss(None)

    CSS = CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
        Binding("escape", "stop_generation", "Stop"),
    ]

    is_generating: reactive[bool] = reactive(False)

    def __init__(
        self,
        base_url: str,
        model: str,
        sessions_dir: Path,
        config_path: Path | None,
        system_prompt: str | None = None,
        resume_session: Session | None = None,
        theme: str = "textual-dark",
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._sessions_dir = sessions_dir
        self._config_path = config_path
        self._system_prompt = system_prompt
        self._api_key = api_key
        self._queue: deque[QueuedPrompt] = deque()
        self._next_queue_id = 0
        self._queue_paused = False
        self._generation_worker: Worker[None] | None = None
        self._turn: ChatTurn | None = None
        self._quit_requested = False
        self._model_picker_open = False
        self._last_speed: float | None = None  # received chunks/s from last response
        self._last_ttft: float | None = None  # time-to-first-token (seconds)
        self.theme = theme
        self._session_allowed_tools: set[str] = set()

        if resume_session:
            self._session = resume_session
            self._history = list(resume_session.messages)
        else:
            self._session = new_session(model, system_prompt)
            self._history = list(self._session.messages)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(id="messages")
        with Horizontal(id="status-bar"):
            yield Label("", id="status-label", markup=False)
            yield LoadingIndicator(id="spinner")
        yield OptionList(id="autocomplete")
        with Vertical(id="input-bar"):
            yield Button("Resume queued", id="resume-queue")
            yield ChatInput(id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "drove chat"
        self.sub_title = self._model
        self._update_status()
        self.query_one("#user-input", ChatInput).focus()

        # Replay existing messages when resuming
        if self._session.message_count > 0:
            self.call_after_refresh(self._replay_history)

    async def _replay_history(self) -> None:
        # Index tool results by tool_call_id for pairing
        tool_results: dict[str, str] = {}
        for msg in self._history:
            if msg["role"] == "tool":
                tool_results[msg["tool_call_id"]] = msg.get("content", "")

        for msg in self._history:
            role = msg["role"]
            if role in ("system", "tool"):
                continue
            content = msg.get("content") or ""
            bubble = Message(role, content, stopped=bool(msg.get("stopped")))
            await self.query_one("#messages").mount(bubble)

            # Render thinking as collapsible section
            thinking = msg.get("thinking", "")
            if thinking:
                await bubble.append_thinking(thinking)
                bubble.finish_thinking()

            # Render tool calls as collapsible sections
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "")
                result = tool_results.get(tc.get("id", ""), "")
                await bubble.append_tool_call(name, arguments, result)

        self._scroll_to_bottom(force=True)

    # ── Slash commands ──────────────────────────────────────────────────────────

    _COMMANDS = {
        "/help": "Show this help",
        "/sessions": "Browse and resume saved sessions",
        "/model": "Choose a model and start a new conversation",
        "/new": "Start a new session",
        "/clear": "Clear current chat (starts new session)",
        "/theme": "/theme [name] — list or set theme",
        "/save": "Save session now (auto-saves after each reply)",
    }

    async def _dispatch_command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if self.is_generating and cmd in {"/new", "/clear", "/sessions"}:
            await self._show_note(
                "Stop the reply and wait for any running tool before changing sessions."
            )
            return

        if cmd == "/help":
            lines = "\n".join(f"  {k:<12} {v}" for k, v in self._COMMANDS.items())
            await self._show_note(f"Available commands:\n{lines}")

        elif cmd == "/model":
            await self._choose_model()

        elif cmd in ("/clear", "/new"):
            self.query_one("#messages").remove_children()
            self._session = new_session(self._model, self._system_prompt)
            self._history = list(self._session.messages)
            self._clear_queue()
            self._session_allowed_tools.clear()
            self._update_status()

        elif cmd == "/sessions":
            sessions = list_sessions(self._sessions_dir, self._model)
            if not sessions:
                await self._show_note("No saved sessions for this model.")
                return
            result = await self.push_screen_wait(SessionPicker(sessions))
            if result is not None:
                await self._load_session(result)

        elif cmd == "/theme":
            if not arg:
                themes = ", ".join(sorted(self.available_themes))
                await self._show_note(f"Current: {self.theme}\nAvailable: {themes}")
            else:
                if arg not in self.available_themes:
                    await self._show_note(f"Unknown theme '{arg}'. Run /theme to list available.")
                else:
                    self.theme = arg
                    save_warning = self._save_theme(arg)
                    if save_warning:
                        await self._show_note(
                            f"Theme '{arg}' applied for this session. {save_warning}"
                        )
                    else:
                        await self._show_note(f"Theme set to '{arg}' and saved.")

        elif cmd == "/save":
            try:
                if self._turn is not None:
                    self._save_turn(self._turn, include_draft=True)
                else:
                    save_session(self._sessions_dir, self._session)
            except Exception as error:
                await self._show_note(f"Could not save the session: {error}")
            else:
                await self._show_note(f"Session saved  ({self._session.id})")

        else:
            await self._show_note(f"Unknown command '{cmd}'. Try /help.")

    async def _choose_model(self) -> None:
        if self._model_picker_open:
            return
        if self.is_generating or self._queue:
            await self._show_note("Wait for pending replies before switching models.")
            return
        self._model_picker_open = True
        try:
            try:
                models = await fetch_models_async(self._base_url, self._api_key)
            except CatalogError as exc:
                await self._show_note(str(exc))
                return
            if self.is_generating or self._queue:
                await self._show_note("Wait for pending replies before switching models.")
                return
            selected = await self.push_screen_wait(ModelPicker(models, self._model))
            if selected is None or selected == self._model:
                return
            if self.is_generating or self._queue:
                await self._show_note("Wait for pending replies before switching models.")
                return
            try:
                save_session(self._sessions_dir, self._session)
            except OSError:
                await self._show_note("Could not save this conversation. Model was not changed.")
                return
            system_prompt = self._session.system_prompt
            session = new_session(selected, system_prompt)
            # Switches can revisit a model in the same second. Do not reuse
            # the timestamp-only ID and overwrite its previous conversation.
            session.id += f"-{time.time_ns()}-{uuid4().hex}"
            self._session = session
            self._model = selected
            self._system_prompt = system_prompt
            self._history = list(session.messages)
            self._session_allowed_tools.clear()
            self._last_speed = self._last_ttft = None
            self.sub_title = selected
            await self.query_one("#messages").remove_children()
            self._update_status()
            self.query_one("#user-input", ChatInput).focus()
        finally:
            self._model_picker_open = False

    async def _show_note(self, text: str) -> None:
        bubble = Message("system-note", text)
        await self.query_one("#messages").mount(bubble)
        self._scroll_to_bottom()

    async def _load_session(self, session: Session) -> None:
        if self.is_generating:
            await self._show_note("Stop the reply before changing sessions.")
            return
        self._session = session
        self._history = list(session.messages)
        self.query_one("#messages").remove_children()
        self._clear_queue()
        self._update_status()
        await self._replay_history()

    def _save_theme(self, theme: str) -> str | None:
        if self._config_path is None:
            return "No config file was supplied; the theme was not saved."
        try:
            from drove.config import load_config

            cfg = load_config(self._config_path)
            cfg = cfg.model_copy(update={"tui_theme": theme})
            cfg.save(self._config_path)  # creates the file if it doesn't exist
        except Exception:
            return "Warning: could not save the theme. Check the config path and permissions."
        return None

    # ── Input handling ──────────────────────────────────────────────────────────

    def action_submit(self) -> None:
        textarea = self.query_one("#user-input", ChatInput)
        text = textarea.text.strip()
        if not text:
            return

        if self._model_picker_open:
            self.run_worker(self._show_note("Finish choosing a model before sending a message."))
            return

        self._hide_autocomplete()
        textarea.clear()

        if text.startswith("/"):
            self.run_worker(self._dispatch_command(text), exclusive=False)
            return

        if self.is_generating or self._queue:
            self._enqueue(text)
        else:
            self._start_turn(text)

    # ── Autocomplete ───────────────────────────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        # Only autocomplete when the entire input is a partial slash command
        if text.startswith("/") and "\n" not in text:
            prefix = text.lower()
            matches = [
                (cmd, desc) for cmd, desc in self._COMMANDS.items() if cmd.startswith(prefix)
            ]
            self._show_autocomplete(matches)
        else:
            self._hide_autocomplete()

    def _show_autocomplete(self, matches: list[tuple[str, str]]) -> None:
        ac = self.query_one("#autocomplete", OptionList)
        ac.clear_options()
        if not matches:
            self._hide_autocomplete()
            return
        for cmd, desc in matches:
            ac.add_option(Option(f"{cmd:<12}  {desc}", id=cmd))
        ac.highlighted = 0
        ac.add_class("visible")

    def _hide_autocomplete(self) -> None:
        ac = self.query_one("#autocomplete", OptionList)
        ac.remove_class("visible")

    def _accept_autocomplete(self) -> None:
        ac = self.query_one("#autocomplete", OptionList)
        if ac.highlighted is not None:
            option = ac.get_option_at_index(ac.highlighted)
            self._fill_command(option.id)

    def _fill_command(self, cmd: str | None) -> None:
        if cmd is None:
            return
        textarea = self.query_one("#user-input", ChatInput)
        textarea.clear()
        textarea.insert(cmd)
        self._hide_autocomplete()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "autocomplete":
            self._fill_command(event.option.id)
            self.query_one("#user-input", ChatInput).focus()

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.action_submit()

    def on_key(self, event: events.Key) -> None:
        ac = self.query_one("#autocomplete", OptionList)
        ac_visible = ac.has_class("visible")

        if not ac_visible:
            return

        if event.key == "up":
            if ac.highlighted is not None and ac.highlighted > 0:
                ac.highlighted -= 1
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            if ac.highlighted is not None and ac.highlighted < ac.option_count - 1:
                ac.highlighted += 1
            event.prevent_default()
            event.stop()
        elif event.key == "tab":
            self._accept_autocomplete()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            if self.is_generating:
                self.action_stop_generation()
            self._hide_autocomplete()
            event.prevent_default()
            event.stop()

    # ── Generation ──────────────────────────────────────────────────────────────

    def watch_is_generating(self, generating: bool) -> None:
        self.query_one("#spinner", LoadingIndicator).set_class(generating, "visible")
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "quit" and isinstance(self.screen, ModelPicker):
            return False
        if action == "stop_generation" and not self.is_generating:
            return None
        return super().check_action(action, parameters)

    async def action_quit(self) -> None:
        # Textual cancels workers without awaiting their cleanup on exit. Settle the
        # owned turn first so a running tool or saved partial reply is not abandoned.
        if self.is_generating:
            self._quit_requested = True
            self.action_stop_generation()
        else:
            self.exit()

    def action_stop_generation(self) -> None:
        self._hide_autocomplete()
        turn = self._turn
        if turn is None or turn.stop_requested:
            return
        turn.stop_requested = True
        self._queue_paused = True
        if turn.started and self._generation_worker is not None:
            self._generation_worker.cancel()
        self._update_status()

    def _enqueue(self, text: str) -> None:
        self._next_queue_id += 1
        bubble = QueuedMessage(self._next_queue_id, text)
        self._queue.append(QueuedPrompt(self._next_queue_id, text, bubble))
        self.query_one("#messages").mount(bubble)
        self._update_status()
        self._scroll_to_bottom()

    def _clear_queue(self) -> None:
        for prompt in self._queue:
            prompt.bubble.remove()
        self._queue.clear()
        self._queue_paused = False

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "resume-queue":
            event.stop()
            self._queue_paused = False
            await self._dispatch_queued()
        elif button_id.startswith("cancel-queued-"):
            event.stop()
            prompt_id = int(button_id.removeprefix("cancel-queued-"))
            for prompt in self._queue:
                if prompt.id == prompt_id:
                    self._queue.remove(prompt)
                    await prompt.bubble.remove()
                    break
            self._update_status()

    async def _dispatch_queued(self) -> None:
        if self.is_generating or self._queue_paused or not self._queue:
            return
        prompt = self._queue.popleft()
        # Reserve ownership before awaiting widget removal or another submit may overtake it.
        self._start_turn(prompt.text)
        await prompt.bubble.remove()

    def _start_turn(self, text: str) -> None:
        if self.is_generating:
            self._enqueue(text)
            return
        if not self._queue:
            self._queue_paused = False
        turn = ChatTurn(self._session, self._history, text)
        turn.history.append({"role": "user", "content": text})
        turn.session.messages = turn.history
        self._turn = turn
        self.is_generating = True
        self._update_status()
        self._generation_worker = self.run_worker(self._send_message(turn), group="generation")

    def _save_turn(self, turn: ChatTurn, *, include_draft: bool = False) -> None:
        messages = list(turn.history)
        if include_draft and not turn.committed:
            messages.append(dict(turn.record))
        turn.session.messages = messages
        save_session(self._sessions_dir, turn.session)

    def _commit_draft(self, turn: ChatTurn) -> None:
        if not turn.committed:
            turn.history.append(turn.record)
            turn.committed = True

    def _finish_tool_history(self, turn: ChatTurn) -> None:
        if not turn.committed:
            return
        # Call IDs are scoped to an assistant batch, not the whole conversation.
        # An earlier batch's result must not satisfy this batch's pending call.
        batch_start = next(i for i, msg in enumerate(turn.history) if msg is turn.record)
        completed = {
            msg.get("tool_call_id")
            for msg in turn.history[batch_start + 1 :]
            if msg["role"] == "tool"
        }
        for call in turn.record.get("tool_calls", []):
            if call["id"] not in completed:
                turn.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "Error: Tool not executed because the reply was stopped.",
                    }
                )

    async def _run_tool(self, turn: ChatTurn, name: str, arguments: str) -> str:
        task = asyncio.create_task(asyncio.to_thread(execute_tool, name, arguments))
        turn.tool_task = task
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Threads cannot be cancelled safely. Retain ownership until effects/result
                    # are known, and suppress later tools and requests after stop.
                    turn.stop_requested = True
                    self._queue_paused = True
                    self._update_status()
        finally:
            turn.tool_task = None

    async def _tool_allowed(self, name: str, arguments: str) -> bool:
        if name not in {"write_file", "shell_execute", "fetch_url"}:
            return True
        if name in self._session_allowed_tools:
            return True
        try:
            from drove.config import load_config

            if name in load_config(self._config_path).allowed_tools:
                return True
        except Exception:
            pass
        choice = await self.push_screen_wait(ToolConfirmationModal(name, arguments))
        if choice not in {"allow-once", "allow-session", "allow-always"}:
            return False
        if choice in {"allow-session", "allow-always"}:
            self._session_allowed_tools.add(name)
        if choice == "allow-always" and self._config_path is not None:
            try:
                from drove.config import load_config

                cfg = load_config(self._config_path)
                if name not in cfg.allowed_tools:
                    cfg.allowed_tools.append(name)
                    cfg.save(self._config_path)
            except Exception:
                pass
        return True

    async def _send_message(self, turn: ChatTurn) -> None:
        turn.started = True
        t_start = time.monotonic()
        t_first: float | None = None
        chunk_count = 0
        successful = False
        try:
            await self.query_one("#messages").mount(Message("user", turn.text))
            self._scroll_to_bottom(force=True)
            while True:
                turn.bubble = Message("assistant")
                await self.query_one("#messages").mount(turn.bubble)
                if turn.stop_requested:
                    raise asyncio.CancelledError
                tool_calls: dict[int, dict[str, str]] = {}
                try:
                    async with aclosing(self._stream_chat(turn.history)) as stream:
                        async for event in stream:
                            if t_first is None and event["kind"] in {"thinking", "content"}:
                                t_first = time.monotonic()
                            if event["kind"] == "thinking":
                                chunk_count += 1
                                turn.record["thinking"] = (
                                    turn.record.get("thinking", "") + event["chunk"]
                                )
                                await turn.bubble.append_thinking(event["chunk"])
                            elif event["kind"] == "content":
                                chunk_count += 1
                                turn.record["content"] += event["chunk"]
                                turn.bubble.finish_thinking()
                                await turn.bubble.append_text(event["chunk"])
                            elif event["kind"] == "tool_call":
                                tc = tool_calls.setdefault(
                                    event["index"],
                                    {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    },
                                )
                                for key in ("id", "name"):
                                    if event.get(key):
                                        tc[key] = event[key]
                                tc["arguments"] += event.get("arguments", "")
                            self._scroll_to_bottom()
                finally:
                    await turn.bubble.finish_text()
                turn.bubble.finish_thinking()
                if not tool_calls:
                    self._commit_draft(turn)
                    successful = True
                    break
                calls = sorted(tool_calls.values(), key=lambda tc: tc["id"])
                turn.record["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in calls
                ]
                self._commit_draft(turn)
                for tc in calls:
                    if turn.stop_requested:
                        break
                    allowed = await self._tool_allowed(tc["name"], tc["arguments"])
                    if turn.stop_requested:
                        break
                    result = (
                        await self._run_tool(turn, tc["name"], tc["arguments"])
                        if allowed
                        else "Error: Tool execution denied by user."
                    )
                    turn.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )
                    await turn.bubble.append_tool_call(tc["name"], tc["arguments"], result)
                    self._scroll_to_bottom()
                if turn.stop_requested:
                    break
                turn.record = {"role": "assistant", "content": ""}
                turn.committed = False
        except asyncio.CancelledError:
            turn.stop_requested = True
        except Exception as error:
            self._queue_paused = True
            await self.query_one("#messages").mount(
                Message("error", _chat_error(error, self._base_url))
            )
        finally:
            try:
                if turn.stop_requested:
                    turn.record["stopped"] = True
                    self._commit_draft(turn)
                    self._finish_tool_history(turn)
                    self._queue_paused = True
                elif not successful:
                    # Keep partial output and a valid tool history after an unexpected failure too.
                    if turn.record.get("content") or turn.record.get("thinking"):
                        self._commit_draft(turn)
                    self._finish_tool_history(turn)
                if turn.stop_requested and (turn.bubble is None or not turn.bubble.is_mounted):
                    turn.bubble = Message(
                        "assistant", turn.record.get("content") or "", stopped=True
                    )
                    await self.query_one("#messages").mount(turn.bubble)
                    if turn.record.get("thinking"):
                        await turn.bubble.append_thinking(turn.record["thinking"])
                if turn.bubble is not None and turn.bubble.is_mounted:
                    turn.bubble.finish_thinking()
                    if turn.stop_requested:
                        turn.bubble.mark_stopped()
                self._save_turn(turn)
            except Exception as error:
                successful = False
                self._queue_paused = True
                await self.query_one("#messages").mount(
                    Message("error", f"Could not save the session: {error}")
                )
            finally:
                elapsed = time.monotonic() - t_first if t_first is not None else 0
                self._last_ttft = t_first - t_start if t_first is not None else None
                self._last_speed = chunk_count / elapsed if elapsed > 0 else None
                if self._turn is turn:
                    self._turn = None
                    self._generation_worker = None
                    self.is_generating = False
                    self._update_status()
                    self.query_one("#user-input", ChatInput).focus()
            if self._quit_requested:
                self.exit()
            elif successful and not turn.stop_requested:
                await self._dispatch_queued()

    async def _stream_chat(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any]]:
        """Yield event dicts: {kind: "thinking"/"content"/"tool_call", ...}."""
        payload = {
            "model": self._model,
            "messages": [{k: v for k, v in msg.items() if k != "stopped"} for msg in messages],
            "stream": True,
            "tools": TOOL_DEFINITIONS,
        }
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(_http_error(resp.status_code, body))
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"]

                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            yield {"kind": "thinking", "chunk": reasoning}

                        content = delta.get("content", "")
                        if content:
                            yield {"kind": "content", "chunk": content}

                        for tc in delta.get("tool_calls", []):
                            yield {
                                "kind": "tool_call",
                                "index": tc.get("index", 0),
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": tc.get("function", {}).get("arguments", ""),
                            }
                    except KeyError, json.JSONDecodeError:
                        continue

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        parts = [f"  model: {self._model}", self._base_url, f"session: {self._session.id}"]
        if self._last_speed is not None:
            parts.append(f"{self._last_speed:.1f} chunks/s (received)")
        if self._last_ttft is not None:
            parts.append(f"TTFT {self._last_ttft:.2f}s")
        if self._queue:
            paused = " (paused)" if self._queue_paused else ""
            parts.append(f"{len(self._queue)} queued{paused}")
        if (
            self._turn is not None
            and self._turn.stop_requested
            and self._turn.tool_task is not None
        ):
            parts.append("Stopping: waiting for running tool")
        resume = self.query_one("#resume-queue", Button)
        resume.set_class(bool(self._queue) and not self.is_generating, "visible")
        self.query_one("#status-label", Label).update("  •  ".join(parts))

    def _scroll_to_bottom(self, force: bool = False) -> None:
        container = self.query_one("#messages", ScrollableContainer)
        # Anchoring follows later Markdown layout updates, not just queued source.
        # Textual releases the anchor when the user scrolls, and restores it only
        # when they return to the bottom. Do not re-enable it on every chunk.
        if force or (not container.is_anchored and container.scroll_y >= container.max_scroll_y):
            container.anchor()
