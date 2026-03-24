"""Terminal UI for chatting with a running vllama server."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import AsyncIterator

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Static,
)

from vllama.sessions import Session, list_sessions, new_session, save_session


# ── Styles ─────────────────────────────────────────────────────────────────────

CSS = """
ChatApp {
    background: $surface;
}

#messages {
    height: 1fr;
    padding: 1 2;
    overflow-y: auto;
}

.message {
    margin-bottom: 1;
    width: 100%;
}

.message-header {
    text-style: bold;
    margin-bottom: 0;
}

.user .message-header { color: $accent; }
.assistant .message-header { color: $success; }
.error .message-header { color: $error; }
.system-note .message-header { color: $warning; }

.message-body {
    padding-left: 2;
    color: $text;
}

.error .message-body { color: $error; }
.system-note .message-body { color: $text-muted; text-style: italic; }

#input-bar {
    height: auto;
    padding: 1 2;
    border-top: solid $panel;
}

#user-input { width: 1fr; }

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


# ── Message widget ──────────────────────────────────────────────────────────────


class Message(Static):
    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self._role = role
        self._content = content
        self.add_class("message", role)

    def compose(self) -> ComposeResult:
        labels = {
            "user": "You",
            "assistant": "Assistant",
            "error": "Error",
            "system-note": "System",
        }
        yield Label(labels.get(self._role, self._role), classes="message-header")
        yield Label(self._content, classes="message-body", id=f"body-{id(self)}")

    def append_text(self, chunk: str) -> None:
        self._content += chunk
        try:
            self.query_one(f"#body-{id(self)}", Label).update(self._content)
        except NoMatches:
            pass


# ── Session picker modal ────────────────────────────────────────────────────────


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


# ── Main app ────────────────────────────────────────────────────────────────────


class ChatApp(App[None]):
    CSS = CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "quit", "Quit"),
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
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._sessions_dir = sessions_dir
        self._config_path = config_path
        self._system_prompt = system_prompt
        self._queue: deque[str] = deque()
        self.theme = theme

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
            yield Label("", id="status-label")
            yield LoadingIndicator(id="spinner")
        with Vertical(id="input-bar"):
            yield Input(placeholder="Message or /help for commands…", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "vllama chat"
        self.sub_title = self._model
        self._update_status()
        self.query_one("#user-input", Input).focus()

        # Replay existing messages when resuming
        if self._session.message_count > 0:
            self.call_after_refresh(self._replay_history)

    @work
    async def _replay_history(self) -> None:
        for msg in self._history:
            if msg["role"] == "system":
                continue
            bubble = Message(msg["role"], msg["content"])
            await self.query_one("#messages").mount(bubble)
        self._scroll_to_bottom()

    # ── Slash commands ──────────────────────────────────────────────────────────

    _COMMANDS = {
        "/help": "Show this help",
        "/sessions": "Browse and resume saved sessions",
        "/new": "Start a new session",
        "/clear": "Clear current chat (starts new session)",
        "/theme": "/theme [name] — list or set theme",
        "/save": "Save session now (auto-saves after each reply)",
    }

    async def _dispatch_command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            lines = "\n".join(f"  {k:<12} {v}" for k, v in self._COMMANDS.items())
            await self._show_note(f"Available commands:\n{lines}")

        elif cmd in ("/clear", "/new"):
            self.query_one("#messages").remove_children()
            self._session = new_session(self._model, self._system_prompt)
            self._history = list(self._session.messages)
            self._queue.clear()
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
                    self._save_theme(arg)
                    await self._show_note(f"Theme set to '{arg}'.")

        elif cmd == "/save":
            save_session(self._sessions_dir, self._session)
            await self._show_note(f"Session saved  ({self._session.id})")

        else:
            await self._show_note(f"Unknown command '{cmd}'. Try /help.")

    async def _show_note(self, text: str) -> None:
        bubble = Message("system-note", text)
        await self.query_one("#messages").mount(bubble)
        self._scroll_to_bottom()

    async def _load_session(self, session: Session) -> None:
        self._session = session
        self._history = list(session.messages)
        self.query_one("#messages").remove_children()
        self._queue.clear()
        self._update_status()
        await self._replay_history()

    def _save_theme(self, theme: str) -> None:
        if self._config_path is None:
            return
        try:
            from vllama.config import load_config

            cfg = load_config(self._config_path)
            cfg = cfg.model_copy(update={"tui_theme": theme})
            cfg.save(self._config_path)  # creates the file if it doesn't exist
        except Exception:
            pass

    # ── Input handling ──────────────────────────────────────────────────────────

    @on(Input.Submitted, "#user-input")
    def handle_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            self.run_worker(self._dispatch_command(text), exclusive=False)
            return

        if self.is_generating:
            self._queue.append(text)
            self._update_status()
        else:
            self._send_message(text)

    # ── Generation ──────────────────────────────────────────────────────────────

    def watch_is_generating(self, generating: bool) -> None:
        spinner = self.query_one("#spinner", LoadingIndicator)
        if generating:
            spinner.add_class("visible")
        else:
            spinner.remove_class("visible")
            if self._queue:
                next_msg = self._queue.popleft()
                self._update_status()
                self._send_message(next_msg)

    @work(exclusive=True)
    async def _send_message(self, text: str) -> None:
        self.is_generating = True

        user_bubble = Message("user", text)
        await self.query_one("#messages").mount(user_bubble)
        self._history.append({"role": "user", "content": text})
        self._session.messages = self._history
        self._scroll_to_bottom()

        assistant_bubble = Message("assistant")
        await self.query_one("#messages").mount(assistant_bubble)
        self._scroll_to_bottom()

        full_response = ""
        try:
            async for chunk in self._stream_chat(self._history):
                assistant_bubble.append_text(chunk)
                full_response += chunk
                self._scroll_to_bottom()
        except Exception as e:
            await self.query_one("#messages").mount(Message("error", str(e)))

        if full_response:
            self._history.append({"role": "assistant", "content": full_response})
            self._session.messages = self._history
            save_session(self._sessions_dir, self._session)

        self.is_generating = False

    async def _stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {"model": self._model, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self._base_url}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"Server error {resp.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except KeyError, json.JSONDecodeError:
                        continue

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        parts = [f"  model: {self._model}", self._base_url, f"session: {self._session.id}"]
        if self._queue:
            parts.append(f"{len(self._queue)} queued")
        self.query_one("#status-label", Label).update("  •  ".join(parts))

    def _scroll_to_bottom(self) -> None:
        self.query_one("#messages", ScrollableContainer).scroll_end(animate=False)
