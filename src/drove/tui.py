"""Terminal UI for chatting with a running drove server."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    Label,
    Markdown,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.markdown import MarkdownStream
from textual.widgets.option_list import Option
from textual.worker import Worker

from drove.chat_models import CatalogError, fetch_context_length, fetch_models_async
from drove.config import ChatHint, load_config, save_tui_completed_hints
from drove.model_picker import ModelPicker
from drove.sessions import Session, list_sessions, new_session, save_session, session_endpoint
from drove.tool_policy import (
    Decision,
    Grant,
    PolicyContext,
    cap_output,
    cap_tool_messages,
    decide,
    fetch_origins,
    is_local_endpoint,
    origin_decision,
    parse_grants,
    read_roots,
)
from drove.tools import TOOL_DEFINITIONS, OriginApproval, ToolContext, ToolFailure, execute_tool

# ── Styles ─────────────────────────────────────────────────────────────────────

CSS = """
ChatApp { background: $background; }
#chat-header { height: 1; padding: 0 1; background: $surface; }
#brand { width: 1fr; color: $accent; text-style: bold; }
#ctx-meter { display: none; width: auto; margin-left: 1; color: $text-muted; }
#ctx-meter.visible { display: block; }
#ctx-meter.warn { color: $warning; }
#chat-header Button {
    width: auto; min-width: 0; height: 1; border: none;
    background: transparent; margin-left: 1;
}
#chat-header Button:hover, #chat-header Button:focus { background: $boost; color: $accent; }
#conversation { height: 1fr; }
#messages {
    height: 1fr; padding: 0 1; overflow-y: auto; overflow-x: hidden;
    scrollbar-size: 1 1;
}
#conversation.empty #messages { display: none; }
#welcome { display: none; height: 1fr; align: center middle; padding: 0 2; }
#conversation.empty #welcome { display: block; }
#welcome-content { width: 100%; max-width: 68; height: auto; }
#welcome-title { text-style: bold; color: $text; }
#welcome-description { color: $text-muted; margin-bottom: 1; width: 100%; }
#welcome-actions { height: auto; }
#welcome-actions Button {
    width: auto; min-width: 0; height: 1; border: none; margin-right: 1;
    background: $surface; color: $text;
}
#welcome-actions Button:hover, #welcome-actions Button:focus { background: $boost; }
.message {
    margin: 0 0 1 0; padding: 0 0 0 1; border-left: solid $panel; width: 100%; height: auto;
}
.message:focus { border-left: solid $accent; background: $boost; }
.message-header { text-style: bold; margin: 0; }
.user { border-left: solid $accent; }
.user .message-header { color: $accent; }
.assistant .message-header { color: $success; }
.error { border-left: solid $error; }
.error .message-header { color: $error; }
.system-note .message-header { color: $warning; }
.queued { color: $text-muted; border-left: dashed $panel; }
.queued Label { color: $text-muted; }
.queued Button, #resume-queue { min-width: 0; height: 1; border: none; }
#resume-queue { display: none; }
#resume-queue.visible { display: block; }
.message-body { color: $text; width: 1fr; padding: 0; }
.message-body MarkdownListItem MarkdownParagraph { margin: 0; }
.message-body MarkdownHeader { margin: 1 0 0 0; }
.message-body MarkdownFence > Label { padding: 0 1; }
.error .message-body { color: $error; }
.system-note .message-body { color: $text-muted; }
.thinking-section, .tool-call-section { margin: 0; padding: 0; }
.thinking-section Label { color: $text-muted; text-style: italic; width: 1fr; }
.tool-call-section Label { color: $text-muted; width: 1fr; }
.tool-call { height: auto; }
.tool-call.-running CollapsibleTitle { color: $accent; }
.tool-call.-done CollapsibleTitle { color: $success; }
.tool-call.-failed CollapsibleTitle { color: $error; }
.tool-call.-denied CollapsibleTitle, .tool-call.-stopped CollapsibleTitle { color: $warning; }
.tool-output {
    display: none; color: $text-muted; width: 1fr; padding: 0 0 0 2;
    border-left: dashed $panel;
}
.tool-call.-streaming .tool-output { display: block; }
#autocomplete {
    display: none; height: auto; max-height: 6; margin: 0 1;
    background: $surface; border-left: solid $accent;
}
#autocomplete.visible { display: block; }
#input-bar { height: auto; padding: 0 1; }
#composer { height: auto; border-top: solid $panel; }
#composer:focus-within { border-top: solid $accent; }
#user-input {
    width: 1fr; height: auto; min-height: 1; max-height: 6;
    border: none; background: $surface; padding: 0 1;
}
#send-message, #stop-reply {
    width: 6; min-width: 6; height: 1; border: none;
    background: $surface; padding: 0;
}
#send-message { color: $accent; }
#stop-reply { display: none; color: $error; }
#composer.generating #send-message { display: none; }
#composer.generating #stop-reply { display: block; }
#guidance { height: 1; padding: 0 1; }
#guidance-label { width: 1fr; color: $text-muted; text-wrap: nowrap; text-overflow: ellipsis; }
#dismiss-tip { width: 3; min-width: 3; height: 1; border: none; background: transparent; }
#status-bar { display: none; height: 1; padding: 0 1; color: $text-muted; }
#status-bar.visible { display: block; }
#activity-label { width: auto; margin-right: 1; color: $accent; }
#status-label { width: 1fr; text-overflow: ellipsis; text-wrap: nowrap; }
ChatApp.narrow #change-model, ChatApp.narrow #open-sessions { display: none; }
ChatApp.narrow #welcome-actions { layout: vertical; }
ChatApp.narrow #welcome-actions Button { margin: 0; }
ChatApp.short #welcome-description { display: none; }
ChatApp.short #user-input { max-height: 4; }
ChatDetails { align: center middle; }
#details-dialog {
    width: 90%; max-width: 76; height: 14; max-height: 85%;
    background: $surface; border: round $panel; padding: 0 1;
}
#details-title { text-style: bold; color: $accent; }
#details-scroll { height: 1fr; overflow-x: hidden; }
#details-content { width: 100%; }
#close-details { height: 1; min-width: 0; border: none; margin-top: 1; }

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

#tool-dialog #tool-reason {
    border-left: solid $warning;
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
SessionPicker, LegacySessionConfirmation {
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

#legacy-session-dialog {
    width: 90%; max-width: 90; height: 70%; max-height: 16;
    background: $surface; border: round $accent; padding: 0 1;
}
#legacy-session-dialog .title { height: 1; text-style: bold; }
#legacy-session-explanation { height: 1fr; }
#legacy-session-actions { height: 1; }
#legacy-session-actions Button {
    width: auto; min-width: 0; height: 1; border: none; margin-right: 2;
}

#session-table {
    height: 1fr;
}
"""


# ── Chat input ─────────────────────────────────────────────────────────────────


class ChatInput(TextArea):
    """TextArea that sends Enter as submit and Shift+Enter as newline."""

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True, show=False),
    ]

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self))

    class Submitted(TextArea.Changed):
        pass

    class NewlineInserted(TextArea.Changed):
        pass

    def _on_key(self, event: events.Key) -> None:  # type: ignore[override]
        if event.key == "shift+enter":
            self.insert("\n")
            self.post_message(self.NewlineInserted(self))
            event.prevent_default()
            event.stop()


# ── Message widget ──────────────────────────────────────────────────────────────


def _markdown_parser() -> MarkdownIt:
    """Use identical Markdown semantics for display and exact code copying."""
    return MarkdownIt("gfm-like", {"html": False})


TOOL_DENIED = "Error: Tool execution denied by user."
CONTEXT_METER_CELLS = 10
#: Fill fraction from which the context meter is shown as nearly full.
CONTEXT_WARN_FRACTION = 0.8
#: Most recent tool output kept for the live preview; the model always gets all of it.
TOOL_OUTPUT_TAIL_CHARS = 4096
#: Lines of that tail shown under a running tool.
TOOL_OUTPUT_LINES = 6
TOOL_ARGUMENTS_PREVIEW = 60


def _compact_tokens(count: int) -> str:
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def _context_used(usage: dict[str, Any]) -> int | None:
    """Tokens in context after the last reply, or None unless both counts were reported."""
    counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if not all(type(count) is int and count >= 0 for count in counts):
        return None
    return sum(counts)  # type: ignore[arg-type]


def _preview_arguments(arguments: str) -> str:
    flat = " ".join(arguments.split())
    if len(flat) > TOOL_ARGUMENTS_PREVIEW:
        return flat[: TOOL_ARGUMENTS_PREVIEW - 1] + "…"
    return flat


class ToolOutputTail:
    """Thread-safe, bounded tail of a running tool's output, read by the UI timer."""

    def __init__(self, limit: int = TOOL_OUTPUT_TAIL_CHARS) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._text = ""
        self._trimmed = False
        self._dirty = False
        self._closed = False

    def append(self, text: str) -> None:
        """Called from the tool's worker thread; never touches widgets."""
        with self._lock:
            if self._closed:
                return
            self._text += text
            if len(self._text) > self._limit:
                self._text = self._text[-self._limit :]
                self._trimmed = True
            self._dirty = True

    def take(self) -> str | None:
        """The text to show if it changed since the last call, else None."""
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            lines = self._text.rstrip("\n").split("\n")
            shown = "\n".join(lines[-TOOL_OUTPUT_LINES:])
            if self._trimmed or len(lines) > TOOL_OUTPUT_LINES:
                shown = "… earlier output omitted\n" + shown
            return shown

    def close(self) -> None:
        with self._lock:
            self._closed = True


class ToolCallEntry(Vertical):
    """One tool call, shown from the moment it starts and updated in place when it ends."""

    def __init__(self, name: str, arguments: str, result: str | None = None) -> None:
        super().__init__(classes="tool-call")
        self.tool_name = name
        self.arguments = arguments
        self.result = result
        self.started: float | None = None
        if result is None:
            self.state = "pending"
        else:
            # Saved sessions do not record the outcome, so only a denial is known.
            self.state = "denied" if result == TOOL_DENIED else "finished"
        self.add_class(f"-{self.state}")

    def compose(self) -> ComposeResult:
        yield Collapsible(
            Label(self._body_text(), markup=False),
            # Collapsible forwards Content to its title renderer despite its str annotation.
            title=Content(self._title_text()),  # type: ignore[arg-type]
            collapsed=True,
            classes="tool-call-section",
        )
        yield Label("", markup=False, classes="tool-output")

    def elapsed(self) -> int:
        return int(time.monotonic() - self.started) if self.started is not None else 0

    def _title_text(self) -> str:
        name, args = self.tool_name, _preview_arguments(self.arguments)
        if self.state == "pending":
            return f"⚙ {name}  {args} · pending"
        if self.state == "running":
            return f"⚙ {name}  {args} · running {self.elapsed()}s"
        if self.state == "denied":
            return f"⊘ {name} denied  {args}"
        if self.state == "stopped":
            return f"⊘ {name} not run  {args}"
        result = self.result or ""
        preview = result[:200] + ("…" if len(result) > 200 else "")
        mark = {"done": "✓ ", "failed": "✗ "}.get(self.state, "⚙ ")
        failed = " failed" if self.state == "failed" else ""
        return f"{mark}{name}{failed} → {preview}"

    def _body_text(self) -> str:
        if self.result is None:
            return f"Arguments: {self.arguments}"
        return f"Arguments: {self.arguments}\n\nResult:\n{self.result}"

    def _set_state(self, state: str) -> None:
        self.remove_class(f"-{self.state}")
        self.state = state
        self.add_class(f"-{state}")
        self.refresh_title()

    def refresh_title(self) -> None:
        if not self.is_mounted:
            return
        self.query_one(Collapsible).title = Content(self._title_text())  # type: ignore[assignment]

    def start(self) -> None:
        self.started = time.monotonic()
        self._set_state("running")

    def show_output(self, text: str) -> None:
        if not self.is_mounted:
            return
        self.query_one(".tool-output", Label).update(text)
        self.add_class("-streaming")

    def finish(self, result: str | None, state: str) -> None:
        self.result = result
        self.remove_class("-streaming")
        if self.is_mounted:
            self.query_one(".tool-call-section Label", Label).update(self._body_text())
        self._set_state(state)


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

    async def append_tool_call(
        self, name: str, arguments: str, result: str | None = None
    ) -> ToolCallEntry:
        """Add a tool call section after the body: pending without a result, else finished."""
        entry = ToolCallEntry(name, arguments, result)
        await self.mount(entry)
        return entry


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


# Tool dialog choices that grant beyond one call, and every choice that allows it.
_SCOPED_CHOICES = frozenset({"allow-session", "allow-always"})
_ALLOW_CHOICES = _SCOPED_CHOICES | {"allow-once"}


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
        if self._per_call and choice in _SCOPED_CHOICES:
            return
        self.dismiss(choice)

    def __init__(
        self,
        tool_name: str,
        arguments: str,
        *,
        reason: str = "",
        scope: str = "",
        per_call: bool = False,
    ) -> None:
        """*per_call* hides the session/always choices; *scope* says what they allow."""
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments
        self._reason = reason
        self._scope = scope
        self._per_call = per_call

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-dialog"):
            yield Label("Tool Call Confirmation", classes="title")
            with ScrollableContainer(id="tool-arguments-scroll"):
                yield Label(
                    f"The model wants to execute the tool: {self._tool_name}",
                    classes="message",
                    markup=False,
                )
                if self._reason:
                    yield Label(self._reason, classes="message", id="tool-reason", markup=False)
                if self._scope and not self._per_call:
                    yield Label(
                        f"Session and Always allow {self._scope}.",
                        classes="message",
                        id="tool-scope",
                        markup=False,
                    )
                yield Label("Arguments:")
                yield Label(self._arguments, id="tool-arguments", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Allow Once (y)", variant="primary", id="allow-once")
                yield Button("Deny Once (n / Esc)", variant="error", id="deny-once")
            if not self._per_call:
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
        table.add_columns("Date", "Turns", "First message", "Endpoint")
        for s in self._sessions:
            date = s.updated_at[:16].replace("T", " ")
            table.add_row(
                date, str(s.message_count), Text(s.title), Text(s.endpoint or "Unassigned (legacy)")
            )
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


class LegacySessionConfirmation(ModalScreen[bool]):
    """Require an explicit destination for a conversation saved before endpoint tracking."""

    BINDINGS = [
        Binding("escape,n", "cancel", "Cancel", priority=True),
        Binding("y", "assign", "Assign and load", priority=True),
    ]

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self._endpoint = endpoint

    def compose(self) -> ComposeResult:
        with Vertical(id="legacy-session-dialog"):
            yield Label("Assign saved conversation", classes="title")
            with ScrollableContainer(id="legacy-session-explanation"):
                yield Label(
                    "This older conversation has no saved endpoint. Loading it assigns it to "
                    f"{self._endpoint}. Future messages send its history to this endpoint.",
                    markup=False,
                )
            with Horizontal(id="legacy-session-actions"):
                yield Button("Cancel · Esc", id="cancel-legacy", compact=True)
                yield Button("Assign · y", id="assign-legacy", compact=True)

    def on_mount(self) -> None:
        self.query_one("#cancel-legacy", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_assign(self) -> None:
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "assign-legacy")


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
    tool_entry: ToolCallEntry | None = None
    tool_output: ToolOutputTail | None = None


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


class ChatDetails(ModalScreen[None]):
    """Connection and last-reply diagnostics, shown only on request."""

    BINDINGS = [Binding("escape,enter", "close", "Close")]

    def __init__(self, details: str) -> None:
        super().__init__()
        self.details = details

    def compose(self) -> ComposeResult:
        with Vertical(id="details-dialog"):
            yield Label("Chat info", id="details-title")
            with ScrollableContainer(id="details-scroll"):
                yield Label(self.details, id="details-content", markup=False)
            yield Button("Close · Esc", id="close-details", compact=True)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()


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
        self._endpoint = session_endpoint(self._base_url)
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
        self._last_usage: dict[str, int] = {}
        self._context_length: int | None = None  # tokens the active model can hold, if known
        self._context_request = 0  # invalidates context lookups for a former model
        self._tool_seconds = 0
        self._completed_hints: set[ChatHint] = set()
        self._hint_save_failed = False
        if config_path is not None:
            try:
                self._completed_hints.update(load_config(config_path).tui_completed_hints)
            except OSError, ValueError:
                pass
        self.theme = theme
        # Grant tokens (see drove.tool_policy) allowed for this session only.
        self._session_allowed_tools: set[str] = set()
        self._cwd = Path.cwd().resolve()  # reads inside it need no approval
        self._remote_endpoint = not is_local_endpoint(self._base_url)
        self._call_grant: Grant | None = None  # the scope approved for the next tool run

        if resume_session:
            if resume_session.model != model:
                raise ValueError("Saved conversation belongs to a different model.")
            if resume_session.endpoint not in {None, self._endpoint}:
                raise ValueError("Saved conversation belongs to a different endpoint.")
            # Callers explicitly supplying an unassigned session own its assignment.
            # CLI auto-resume excludes it; /sessions asks before loading it.
            resume_session.endpoint = self._endpoint
            self._session = resume_session
            self._system_prompt = resume_session.system_prompt
            self._history = cap_tool_messages(resume_session.messages)
        else:
            self._session = new_session(model, system_prompt, endpoint=self._endpoint)
            self._history = list(self._session.messages)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-header"):
            yield Label("drove", id="brand")
            yield Button(
                "Model", id="change-model", compact=True, tooltip="Choose a model · /model"
            )
            yield Button(
                "Sessions", id="open-sessions", compact=True, tooltip="Saved conversations"
            )
            yield Button("+ New", id="new-chat", compact=True, tooltip="New conversation · /new")
            yield Button(
                "Info", id="chat-details", compact=True, tooltip="Connection and reply details"
            )
            yield Label("", id="ctx-meter", markup=False)
        with Vertical(id="conversation", classes="empty"):
            with Vertical(id="welcome"):
                with Vertical(id="welcome-content"):
                    yield Label("What are we working on?", id="welcome-title")
                    yield Label(
                        "Ask a question, explore an idea, or work through code.",
                        id="welcome-description",
                    )
                    with Horizontal(id="welcome-actions"):
                        yield Button("Choose model", id="welcome-model", compact=True)
                        yield Button("Resume session", id="welcome-sessions", compact=True)
                        yield Button("Commands", id="welcome-help", compact=True)
            yield ScrollableContainer(id="messages")
        with Horizontal(id="status-bar"):
            yield Label("", id="activity-label", markup=False)
            yield Label("", id="status-label", markup=False)
        yield OptionList(id="autocomplete")
        with Vertical(id="input-bar"):
            yield Button("Resume queued", id="resume-queue", compact=True)
            with Horizontal(id="composer"):
                yield ChatInput(id="user-input", placeholder="Message…")
                yield Button("Send", id="send-message", compact=True, tooltip="Send message")
                yield Button("Stop", id="stop-reply", compact=True, tooltip="Stop reply")
        with Horizontal(id="guidance"):
            yield Label("", id="guidance-label", markup=False)
            yield Button("×", id="dismiss-tip", compact=True, tooltip="Dismiss this tip")

    def action_show_details(self) -> None:
        self._learn_hint("details")

        def focus_input(result: None) -> None:
            self.query_one("#user-input", ChatInput).focus()

        self.push_screen(ChatDetails(self._details_text()), focus_input)

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 50, "narrow")
        self.set_class(event.size.height < 24, "short")

    def on_mount(self) -> None:
        self.title = "drove chat"
        self._tool_timer = self.set_interval(0.1, self._tick_tool, pause=True)
        self.run_worker(self._refresh_context_length(), exit_on_error=False)
        self._update_status()
        self.query_one("#user-input", ChatInput).focus()

        # Replay existing messages when resuming
        if self._session.message_count > 0:
            self.call_after_refresh(self._replay_history)

    async def _refresh_context_length(self) -> None:
        self._context_request += 1
        request, model = self._context_request, self._model
        length = await fetch_context_length(self._base_url, self._api_key, model)
        if request == self._context_request and model == self._model:
            self._context_length = length
            self._update_status()

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
        "/help": "Show commands and keyboard shortcuts",
        "/info": "Show connection, context, and last-reply details",
        "/tips": "/tips [off] — restart guidance or hide all tips",
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

        if cmd == "/info":
            self.action_show_details()
        elif cmd == "/tips":
            if arg not in {"", "off"}:
                await self._show_note("Use /tips to restart guidance or /tips off to hide it.")
                return
            self._completed_hints = set(self._HINTS) if arg == "off" else set()
            self._save_hint_progress()
            self._update_guidance()
        elif cmd == "/help":
            self._learn_hint("commands")
            lines = "\n".join(f"  {k:<12} {v}" for k, v in self._COMMANDS.items())
            await self._show_note(
                f"Available commands:\n{lines}\n\n"
                "Enter send · Shift+Enter newline · Esc stop reply\n"
                "Ctrl+C / Ctrl+Q quit · Tab focus messages and controls\n"
                "Ctrl+Y copy focused message · Ctrl+B copy its code"
            )

        elif cmd == "/model":
            await self._choose_model()

        elif cmd in ("/clear", "/new"):
            await self.query_one("#messages").remove_children()
            self._session = new_session(self._model, self._system_prompt, endpoint=self._endpoint)
            self._history = list(self._session.messages)
            self._clear_queue()
            self._session_allowed_tools.clear()
            self._last_speed = self._last_ttft = None
            self._last_usage = {}
            self._update_status()

        elif cmd == "/sessions":
            # Skipped-file notes go to the Textual log, not the terminal it owns.
            sessions = list_sessions(
                self._sessions_dir,
                self._model,
                endpoint=self._endpoint,
                include_legacy=True,
                warn=self.log.warning,
            )
            if not sessions:
                await self._show_note("No saved sessions for this model and endpoint.")
                return
            result = await self.push_screen_wait(SessionPicker(sessions))
            if result is not None:
                if result.endpoint is None and not await self.push_screen_wait(
                    LegacySessionConfirmation(self._endpoint)
                ):
                    return
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
            except OSError, ValueError:
                await self._show_note("Could not save this conversation. Model was not changed.")
                return
            system_prompt = self._session.system_prompt
            self._session = new_session(selected, system_prompt, endpoint=self._endpoint)
            self._model = selected
            self._context_request += 1
            self._context_length = next(
                (model.context_length for model in models if model.id == selected), None
            )
            self._system_prompt = system_prompt
            self._history = list(self._session.messages)
            self._session_allowed_tools.clear()
            self._last_speed = self._last_ttft = None
            self._last_usage = {}
            await self.query_one("#messages").remove_children()
            self._update_status()
            self.query_one("#user-input", ChatInput).focus()
        finally:
            self._model_picker_open = False

    async def _show_note(self, text: str) -> None:
        bubble = Message("system-note", text)
        await self.query_one("#messages").mount(bubble)
        self._update_status()
        self._scroll_to_bottom()

    async def _load_session(self, session: Session) -> None:
        if self.is_generating:
            await self._show_note("Stop the reply before changing sessions.")
            return
        if session.model != self._model or session.endpoint not in {None, self._endpoint}:
            await self._show_note("This conversation belongs to a different model or endpoint.")
            return
        if session.endpoint is None:
            session.endpoint = self._endpoint
            try:
                save_session(self._sessions_dir, session)
            except OSError, ValueError:
                session.endpoint = None
                await self._show_note(
                    "Could not assign this conversation. Session was not changed."
                )
                return
        self._session = session
        self._system_prompt = session.system_prompt
        self._session_allowed_tools.clear()
        self._call_grant = None
        self._history = cap_tool_messages(session.messages)
        self._last_speed = self._last_ttft = None
        self._last_usage = {}
        await self.query_one("#messages").remove_children()
        self._clear_queue()
        self._update_status()
        await self._replay_history()

    def _save_theme(self, theme: str) -> str | None:
        if self._config_path is None:
            return "No config file was supplied; the theme was not saved."
        try:
            from drove.config import set_config_value

            set_config_value("tui_theme", theme, self._config_path)  # creates a missing file
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

        self._learn_hint("send")
        self._hide_autocomplete()
        textarea.clear()

        if text.startswith("/"):
            self._learn_hint("commands")
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

    def on_chat_input_newline_inserted(self, event: ChatInput.NewlineInserted) -> None:
        self._learn_hint("newline")

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
        self.query_one("#composer").set_class(generating, "generating")
        self.query_one("#user-input", ChatInput).placeholder = (
            "Queue a follow-up…" if generating else "Message…"
        )
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
        self._learn_hint("stop")
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
        commands = {
            "change-model": "/model",
            "welcome-model": "/model",
            "open-sessions": "/sessions",
            "welcome-sessions": "/sessions",
            "new-chat": "/new",
            "welcome-help": "/help",
        }
        if button_id == "chat-details":
            event.stop()
            self.action_show_details()
        elif button_id == "dismiss-tip":
            event.stop()
            hint = self._next_hint()
            if hint is not None:
                self._learn_hint(hint)
            self.query_one(ChatInput).focus()
        elif button_id in commands:
            event.stop()
            self.run_worker(self._dispatch_command(commands[button_id]))
            self.query_one("#user-input", ChatInput).focus()
        elif button_id == "send-message":
            event.stop()
            self.action_submit()
            self.query_one("#user-input", ChatInput).focus()
        elif button_id == "stop-reply":
            event.stop()
            self.action_stop_generation()
        elif button_id == "resume-queue":
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

    def _tick_tool(self) -> None:
        """Refresh the running tool's preview and elapsed time (at most 10 times a second)."""
        turn = self._turn
        entry = turn.tool_entry if turn is not None else None
        if turn is None or entry is None or entry.state != "running":
            return
        if turn.tool_output is not None and (text := turn.tool_output.take()) is not None:
            entry.show_output(text)
            self._scroll_to_bottom()
        seconds = entry.elapsed()
        if seconds != self._tool_seconds:
            self._tool_seconds = seconds
            entry.refresh_title()
            self._update_status()

    async def _run_tool_call(self, turn: ChatTurn, call: dict[str, str]) -> str | None:
        """Show, approve and run one call; return its result, or None when stopped first."""
        assert turn.bubble is not None
        entry = await turn.bubble.append_tool_call(call["name"], call["arguments"])
        turn.tool_entry = entry
        self._scroll_to_bottom()
        try:
            allowed = await self._tool_allowed(call["name"], call["arguments"])
            if turn.stop_requested:
                return None
            if not allowed:
                entry.finish(TOOL_DENIED, "denied")
                return TOOL_DENIED
            turn.tool_output = ToolOutputTail()
            entry.start()
            self._tool_seconds = 0
            self._tool_timer.resume()
            self._update_status()
            try:
                result = await self._run_tool(turn, call["name"], call["arguments"])
            except Exception as error:
                # The tool ran and crashed: show it as failed, then let the turn report the error.
                entry.finish(f"Error executing tool: {error}", "failed")
                raise
            entry.finish(str(result), "failed" if isinstance(result, ToolFailure) else "done")
            return str(result)
        finally:
            if entry.state in {"pending", "running"}:
                entry.finish(None, "stopped")
            if turn.tool_output is not None:
                turn.tool_output.close()
            turn.tool_entry = turn.tool_output = None
            self._tool_timer.pause()
            self._update_status()

    async def _run_tool(self, turn: ChatTurn, name: str, arguments: str) -> str:
        on_output = turn.tool_output.append if turn.tool_output is not None else None
        grant, self._call_grant = self._call_grant, None
        policy = self._policy_context()
        origins, internal_origins = fetch_origins(policy, grant)
        context = ToolContext(
            cwd=self._cwd,
            read_roots=read_roots(name, policy, grant),
            approved_origins=origins,
            internal_origins=internal_origins,
            approve_origin=self._origin_approver(turn),
        )
        task = asyncio.create_task(
            asyncio.to_thread(execute_tool, name, arguments, on_output=on_output, context=context)
        )
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

    def _policy_context(self) -> PolicyContext:
        tokens = set(self._session_allowed_tools)
        try:
            from drove.config import load_config

            tokens.update(load_config(self._config_path).allowed_tools)
        except Exception:
            pass
        return PolicyContext(self._cwd, self._remote_endpoint, parse_grants(tokens))

    async def _tool_allowed(self, name: str, arguments: str) -> bool:
        decision = decide(name, arguments, self._policy_context())
        allowed = not decision.prompt or await self._confirm(name, arguments, decision)
        self._call_grant = decision.grant if allowed else None
        return allowed

    async def _confirm(self, name: str, arguments: str, decision: Decision) -> bool:
        """Ask about one call; a session/always choice records only the decision's scope."""
        grant = decision.grant if decision.scoped_choices else None
        modal = ToolConfirmationModal(
            name,
            arguments,
            reason=decision.reason,
            scope=grant.describe() if grant is not None else "",
            per_call=grant is None,
        )
        choice = await self.push_screen_wait(modal)
        if choice not in _ALLOW_CHOICES:
            return False
        if grant is not None and choice in _SCOPED_CHOICES:
            self._session_allowed_tools.add(grant.token())
            if choice == "allow-always" and self._config_path is not None:
                try:
                    from drove.config import add_allowed_tool

                    add_allowed_tool(grant.token(), self._config_path)
                except Exception:
                    self.notify(
                        "The approval could not be saved to config; it applies to this session.",
                        severity="warning",
                    )
        return True

    def _origin_approver(self, turn: ChatTurn) -> OriginApproval:
        """Let fetch_url ask, from its worker thread, about an origin it meets mid-fetch."""
        loop = asyncio.get_running_loop()

        def approve(origin: str, internal: bool) -> bool:
            if turn.stop_requested:
                return False
            future = asyncio.run_coroutine_threadsafe(
                self._approve_origin(turn, origin, internal), loop
            )
            # Poll, so a stop, quit or closed loop ends the wait: cancelling the approval
            # dismisses its dialog (push_screen_wait cleans up), and the fetch is refused.
            while True:
                try:
                    return future.result(timeout=0.2)
                except TimeoutError:
                    if turn.stop_requested or loop.is_closed():
                        future.cancel()
                        return False
                except Exception:
                    return False

        return approve

    async def _approve_origin(self, turn: ChatTurn, origin: str, internal: bool) -> bool:
        if turn.stop_requested:
            return False
        arguments = json.dumps({"origin": origin})
        allowed = await self._confirm("fetch_url", arguments, origin_decision(origin, internal))
        return allowed and not turn.stop_requested

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
                usage: dict[str, int] = {}
                try:
                    async with aclosing(self._stream_chat(turn.history)) as stream:
                        async for event in stream:
                            if t_first is None and event["kind"] in {"thinking", "content"}:
                                t_first = time.monotonic()
                            if event["kind"] == "usage":
                                usage = event["usage"]
                            elif event["kind"] == "thinking":
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
                # Each response reports the whole context; keep only the latest one.
                self._last_usage = usage
                self._update_status()
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
                    result = await self._run_tool_call(turn, tc)
                    if result is None:
                        break
                    turn.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": cap_output(result),
                        }
                    )
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
            if not successful:
                # A stopped or failed reply changed the history after the last usage report.
                self._last_usage = {}
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
            "stream_options": {"include_usage": True},
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
                        usage = obj.get("usage")
                        if isinstance(usage, dict):
                            yield {
                                "kind": "usage",
                                "usage": {
                                    key: value
                                    for key, value in usage.items()
                                    if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                                    and type(value) is int
                                    and value >= 0
                                },
                            }
                        if not obj.get("choices"):
                            continue
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

    _HINTS: dict[ChatHint, str] = {
        "send": "Enter sends your message.",
        "newline": "Shift+Enter adds a new line.",
        "commands": "Type / to explore commands.",
        "details": "Info keeps model, connection, and reply details out of your way.",
        "stop": "Esc stops the current reply; your partial text is saved.",
    }

    def _next_hint(self) -> ChatHint | None:
        if self.is_generating and "stop" not in self._completed_hints:
            return "stop"
        return next(
            (key for key in self._HINTS if key != "stop" and key not in self._completed_hints), None
        )

    def _update_guidance(self) -> None:
        hint = self._next_hint()
        self.query_one("#guidance").display = hint is not None
        self.query_one("#guidance-label", Label).update(self._HINTS[hint] if hint else "")

    def _learn_hint(self, hint: ChatHint) -> None:
        if hint in self._completed_hints:
            return
        self._completed_hints.add(hint)
        self._save_hint_progress()
        self._update_guidance()

    def _save_hint_progress(self) -> None:
        if self._config_path is None:
            return
        try:
            save_tui_completed_hints(list(self._completed_hints), self._config_path)
        except OSError, ValueError:
            if not self._hint_save_failed:
                self.notify(
                    "Tip progress could not be saved; remembered for this run only.",
                    severity="warning",
                )
                self._hint_save_failed = True

    def _details_text(self) -> str:
        speed = (
            f"{self._last_speed:.1f} chunks/s (received)"
            if self._last_speed is not None
            else "Not available"
        )
        ttft = f"{self._last_ttft:.2f}s" if self._last_ttft is not None else "Not available"
        prompt = self._last_usage.get("prompt_tokens")
        completion = self._last_usage.get("completion_tokens")
        used = _context_used(self._last_usage)
        total = self._context_length
        if used is None:
            context = "Not reported"
        elif total is None:
            context = f"{used:,} tokens (context size unknown)"
        else:
            context = f"{used:,} / {total:,} tokens ({used / total:.0%})"
        return (
            f"Model: {self._model}\nEndpoint: {self._base_url}\n"
            f"Session: {self._session.id}\n"
            f"Context: {context}\n"
            f"Context messages: {len(self._history)}\n"
            f"Last prompt tokens: {prompt if prompt is not None else 'Not reported'}\n"
            "Last completion tokens: "
            f"{completion if completion is not None else 'Not reported'}\n"
            f"Last reply speed: {speed}\nTime to first token: {ttft}"
        )

    def _update_context_meter(self) -> None:
        meter = self.query_one("#ctx-meter", Label)
        used = _context_used(self._last_usage)
        total = self._context_length
        meter.set_class(used is not None, "visible")
        if used is None:
            meter.set_class(False, "warn")
            return
        if total is None:
            # Without a known size a fill level would be a guess; show the count instead.
            meter.update(f"ctx {_compact_tokens(used)}")
            meter.tooltip = f"{used:,} tokens in context · context size unknown"
            meter.set_class(False, "warn")
            return
        fraction = used / total
        filled = min(CONTEXT_METER_CELLS, round(fraction * CONTEXT_METER_CELLS))
        meter.update("▰" * filled + "▱" * (CONTEXT_METER_CELLS - filled))
        meter.tooltip = f"{used:,} / {total:,} tokens"
        meter.set_class(fraction >= CONTEXT_WARN_FRACTION, "warn")

    def _update_status(self) -> None:
        self.query_one("#conversation").set_class(
            not any(msg["role"] != "system" for msg in self._history)
            and not self.query_one("#messages").children,
            "empty",
        )
        activity = "Replying…" if self.is_generating else ""
        entry = self._turn.tool_entry if self._turn is not None else None
        if entry is not None and entry.state == "running":
            activity = f"Running {entry.tool_name}… {entry.elapsed()}s"
        if self._turn is not None and self._turn.stop_requested:
            activity = "Stopping…"
        elif self._queue_paused and self._queue:
            activity = "Queue paused"
        self.query_one("#activity-label", Label).update(activity)
        parts = []
        if self._queue:
            paused = " (paused)" if self._queue_paused else ""
            parts.append(f"{len(self._queue)} queued{paused}")
        if (
            self._turn is not None
            and self._turn.stop_requested
            and self._turn.tool_task is not None
        ):
            parts.append("Stopping: waiting for running tool")
        self.query_one("#resume-queue", Button).set_class(
            bool(self._queue) and not self.is_generating, "visible"
        )
        self.query_one("#status-label", Label).update("  ·  ".join(parts))
        self.query_one("#status-bar").set_class(bool(activity or parts), "visible")
        self._update_context_meter()
        if isinstance(self.screen, ChatDetails):
            self.screen.query_one("#details-content", Label).update(self._details_text())
        self._update_guidance()

    def _scroll_to_bottom(self, force: bool = False) -> None:
        container = self.query_one("#messages", ScrollableContainer)
        # Anchoring follows later Markdown layout updates, not just queued source.
        # Textual releases the anchor when the user scrolls, and restores it only
        # when they return to the bottom. Do not re-enable it on every chunk.
        if force or (not container.is_anchored and container.scroll_y >= container.max_scroll_y):
            container.anchor()
