"""Terminal UI for chatting with a running drove server."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
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
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

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

#tool-dialog {
    width: 60;
    height: auto;
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

#tool-dialog .buttons {
    margin-top: 1;
    height: auto;
    align: center middle;
}

#tool-dialog Button {
    margin: 0 1;
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

    def _on_key(self, event) -> None:
        if event.key == "shift+enter":
            self.insert("\n")
            event.prevent_default()
            event.stop()


# ── Message widget ──────────────────────────────────────────────────────────────


class Message(Static):
    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self._role = role
        self._content = content
        self._thinking = ""
        self._has_thinking_widget = False
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

    async def append_thinking(self, chunk: str) -> None:
        self._thinking += chunk
        if not self._has_thinking_widget:
            self._has_thinking_widget = True
            collapsible = Collapsible(
                Label(self._thinking, id=f"thinking-{id(self)}"),
                title="Thinking…",
                collapsed=True,
                classes="thinking-section",
            )
            # Insert before the body label
            try:
                body = self.query_one(f"#body-{id(self)}", Label)
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
            title=f"⚙ {name} → {result_preview}",
            collapsed=True,
            classes="tool-call-section",
        )
        await self.mount(collapsible)


# ── Session picker modal ────────────────────────────────────────────────────────


class ToolConfirmationModal(ModalScreen[str]):
    """Modal to confirm tool execution."""

    def __init__(self, tool_name: str, arguments: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-dialog"):
            yield Label("Tool Call Confirmation", classes="title")
            yield Label(
                f"The model wants to execute the tool: [bold]{self._tool_name}[/bold]",
                classes="message",
            )
            yield Label(f"Arguments: {self._arguments}", classes="message")
            with Horizontal(classes="buttons"):
                yield Button("Allow Once", variant="primary", id="allow-once")
                yield Button("Deny Once", variant="error", id="deny-once")
            with Horizontal(classes="buttons"):
                yield Button("Allow for Session", variant="default", id="allow-session")
                yield Button("Allow Always", variant="default", id="allow-always")

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


# ── Main app ────────────────────────────────────────────────────────────────────


class ChatApp(App[None]):
    async def push_screen_wait(self, screen: ModalScreen) -> Any:
        """Push a screen and wait for it to dismiss, returning the result."""
        from asyncio import Future

        future: Future[Any] = Future()
        self.push_screen(screen, callback=future.set_result)
        return await future

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
        api_key: str | None = None,
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._sessions_dir = sessions_dir
        self._config_path = config_path
        self._system_prompt = system_prompt
        self._api_key = api_key
        self._queue: deque[str] = deque()
        self._last_speed: float | None = None  # tok/s from last response
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
            yield Label("", id="status-label")
            yield LoadingIndicator(id="spinner")
        yield OptionList(id="autocomplete")
        with Vertical(id="input-bar"):
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
            bubble = Message(role, content)
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
            from drove.config import load_config

            cfg = load_config(self._config_path)
            cfg = cfg.model_copy(update={"tui_theme": theme})
            cfg.save(self._config_path)  # creates the file if it doesn't exist
        except Exception:
            pass

    # ── Input handling ──────────────────────────────────────────────────────────

    def action_submit(self) -> None:
        textarea = self.query_one("#user-input", ChatInput)
        text = textarea.text.strip()
        if not text:
            return

        self._hide_autocomplete()
        textarea.clear()

        if text.startswith("/"):
            self.run_worker(self._dispatch_command(text), exclusive=False)
            return

        if self.is_generating:
            self._queue.append(text)
            self._update_status()
        else:
            self._send_message(text)

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

    def on_key(self, event) -> None:
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
            self._hide_autocomplete()
            event.prevent_default()
            event.stop()

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
        self._scroll_to_bottom(force=True)

        t_start = time.monotonic()
        t_first: float | None = None
        token_count = 0

        try:
            while True:
                assistant_bubble = Message("assistant")
                await self.query_one("#messages").mount(assistant_bubble)
                self._scroll_to_bottom()

                full_response = ""
                full_thinking = ""
                was_thinking = False
                tool_calls: dict[int, dict] = {}

                async for event in self._stream_chat(self._history):
                    kind = event["kind"]
                    if t_first is None:
                        t_first = time.monotonic()

                    if kind == "thinking":
                        token_count += 1
                        was_thinking = True
                        full_thinking += event["chunk"]
                        await assistant_bubble.append_thinking(event["chunk"])
                    elif kind == "content":
                        token_count += 1
                        if was_thinking:
                            assistant_bubble.finish_thinking()
                            was_thinking = False
                        assistant_bubble.append_text(event["chunk"])
                        full_response += event["chunk"]
                    elif kind == "tool_call":
                        idx = event["index"]
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": event.get("id", ""),
                                "name": event.get("name", ""),
                                "arguments": "",
                            }
                        if event.get("id"):
                            tool_calls[idx]["id"] = event["id"]
                        if event.get("name"):
                            tool_calls[idx]["name"] = event["name"]
                        tool_calls[idx]["arguments"] += event.get("arguments", "")
                    self._scroll_to_bottom()

                if was_thinking:
                    assistant_bubble.finish_thinking()

                if not tool_calls:
                    # No tool calls — save the final assistant message and break
                    if full_response or full_thinking:
                        assistant_msg: dict = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        if full_thinking:
                            assistant_msg["thinking"] = full_thinking
                        self._history.append(assistant_msg)
                        self._session.messages = self._history
                        save_session(self._sessions_dir, self._session)
                    break

                # Build the assistant message with tool_calls for history
                assistant_msg: dict = {"role": "assistant"}
                if full_response:
                    assistant_msg["content"] = full_response
                else:
                    assistant_msg["content"] = None
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in sorted(tool_calls.values(), key=lambda t: t["id"])
                ]
                self._history.append(assistant_msg)

                # Execute each tool call and show results
                for tc in sorted(tool_calls.values(), key=lambda t: t["id"]):
                    name = tc["name"]
                    args_str = tc["arguments"]

                    allowed = False
                    sensitive_tools = {"write_file", "shell_execute", "fetch_url"}

                    if name not in sensitive_tools:
                        allowed = True
                    else:
                        # Check session permission
                        if name in self._session_allowed_tools:
                            allowed = True
                        else:
                            # Check persistent permission
                            try:
                                from drove.config import load_config

                                cfg = load_config(self._config_path)
                                if name in cfg.allowed_tools:
                                    allowed = True
                            except Exception:
                                pass

                    if not allowed:
                        res = await self.push_screen_wait(ToolConfirmationModal(name, args_str))
                        if res == "allow-once":
                            allowed = True
                        elif res == "allow-session":
                            allowed = True
                            self._session_allowed_tools.add(name)
                        elif res == "allow-always":
                            allowed = True
                            self._session_allowed_tools.add(name)
                            # Persist to config
                            try:
                                from drove.config import load_config

                                cfg = load_config(self._config_path)
                                if name not in cfg.allowed_tools:
                                    cfg.allowed_tools.append(name)
                                    cfg.save(self._config_path)
                            except Exception:
                                pass
                        else:
                            allowed = False

                    if allowed:
                        result = execute_tool(name, args_str)
                    else:
                        result = "Error: Tool execution denied by user."

                    # Append tool result to history
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )

                    # Show as collapsible section
                    await assistant_bubble.append_tool_call(
                        tc["name"],
                        tc["arguments"],
                        result,
                    )
                    self._scroll_to_bottom()

                self._session.messages = self._history
                # Loop back to get the model's next response

        except Exception as e:
            await self.query_one("#messages").mount(Message("error", str(e)))

        t_end = time.monotonic()
        if t_first is not None:
            self._last_ttft = t_first - t_start
            gen_duration = t_end - t_first
            self._last_speed = token_count / gen_duration if gen_duration > 0 else None
        else:
            self._last_speed = None
            self._last_ttft = None

        self._update_status()
        self.is_generating = False

    async def _stream_chat(
        self,
        messages: list[dict],
    ) -> AsyncIterator[dict]:
        """Yield event dicts: {kind: "thinking"/"content"/"tool_call", ...}."""
        payload = {
            "model": self._model,
            "messages": messages,
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
            parts.append(f"{self._last_speed:.1f} tok/s")
        if self._last_ttft is not None:
            parts.append(f"TTFT {self._last_ttft:.2f}s")
        if self._queue:
            parts.append(f"{len(self._queue)} queued")
        self.query_one("#status-label", Label).update("  •  ".join(parts))

    def _scroll_to_bottom(self, force: bool = False) -> None:
        container = self.query_one("#messages", ScrollableContainer)
        # Auto-scroll only if already near the bottom or forced
        near_bottom = (container.max_scroll_y - container.scroll_y) <= 5
        if force or near_bottom:
            container.scroll_end(animate=False)
