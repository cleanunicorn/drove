"""Terminal UI for chatting with a running vllama server."""

from __future__ import annotations

import asyncio
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

from vllama.agents.bash_procs import BgProcs
from vllama.agents.evaluator import check_done
from vllama.agents.llm_call import call_chat_json
from vllama.agents.permissions import AbortTurn, Policy, PromptDecision
from vllama.agents.router import select_tools
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools import ToolContext, all_specs
from vllama.config import load_config
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


# ── Permission modal ────────────────────────────────────────────────────────────
class PermissionModal(ModalScreen[str]):
    """Modal that asks the user how to handle a prompt-tier tool call.

    Dismisses with one of: "allow" | "session_allow" | "deny_continue" | "deny_abort".
    """

    BINDINGS = [
        Binding("a", "allow", "Allow"),
        Binding("s", "session_allow", "Session-allow"),
        Binding("d", "deny_continue", "Deny & Continue"),
        Binding("x", "deny_abort", "Deny & Abort"),
        Binding("escape", "deny_abort", "Cancel"),
    ]

    def __init__(self, name: str, args: dict[str, object]) -> None:
        super().__init__()
        self._name = name
        self._args = args

    def compose(self) -> ComposeResult:
        import json as _json

        try:
            pretty = _json.dumps(self._args, indent=2, default=str)
        except TypeError, ValueError:
            pretty = repr(self._args)
        if len(pretty) > 1024:
            pretty = pretty[:1024] + "\n… (truncated)"

        yield Vertical(
            Static(f"Tool call: {self._name}", id="perm-title"),
            Static(pretty, id="perm-args"),
            Horizontal(
                Static(
                    "[A]llow  [S]ession-allow  [D]eny&Continue  e[X]it-turn",
                    id="perm-help",
                ),
            ),
            id="perm-modal",
        )

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_session_allow(self) -> None:
        self.dismiss("session_allow")

    def action_deny_continue(self) -> None:
        self.dismiss("deny_continue")

    def action_deny_abort(self) -> None:
        self.dismiss("deny_abort")


# ── Main app ────────────────────────────────────────────────────────────────────


def render_permits_summary(runtime: ToolRuntime) -> str:
    """Human-readable summary of runtime permissions state, for /permits."""
    lines: list[str] = []
    lines.append("Tier defaults: read=auto, mutate=prompt, exec=prompt")
    if runtime.policy.trust_all:
        lines.append("Policy: TRUST MODE (all tools auto-approved)")
    elif runtime.policy.overrides:
        lines.append("Config overrides:")
        for name, decision in sorted(runtime.policy.overrides.items()):
            lines.append(f"  {name} = {decision.value}")
    else:
        lines.append("Config overrides: (none; tier defaults apply)")
    if runtime.session_permits:
        lines.append("Session permits: " + ", ".join(sorted(runtime.session_permits)))
    else:
        lines.append("Session permits: (none)")
    return "\n".join(lines)


def render_bg_listing(bg_procs: BgProcs) -> str:
    """Human-readable table of background procs, for /bg."""
    items = bg_procs.list()
    if not items:
        return "No background shells."
    lines = ["Background shells:"]
    for bp in items:
        status = f"exit {bp.exit_code}" if bp.exit_code is not None else "running"
        preview = bp.command if len(bp.command) < 60 else bp.command[:57] + "..."
        lines.append(f"  {bp.shell_id}  pid={bp.pid}  {status}  {preview}")
    return "\n".join(lines)


def render_todos_summary(todos: list[dict[str, Any]]) -> str:
    """Human-readable todo checklist, for /todos."""
    if not todos:
        return "No todos."
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    lines = ["Todos:"]
    for t in todos:
        mark = marks.get(t.get("status", ""), "[?]")
        lines.append(f"  {mark} {t.get('id', '?')}: {t.get('content', '')}")
    return "\n".join(lines)


class ChatApp(App[None]):
    CSS = CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "quit", "Quit"),
    ]

    is_generating: reactive[bool] = reactive(False)
    iteration: reactive[int] = reactive(0)

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
        self._follow = True  # auto-scroll to bottom on new content
        self.theme = theme

        if resume_session:
            self._session = resume_session
            self._history = list(resume_session.messages)
        else:
            self._session = new_session(model, system_prompt)
            self._history = list(self._session.messages)

        cfg = load_config()
        self._bg_procs = BgProcs()
        self._tool_ctx = ToolContext(
            cwd=Path.cwd(),
            cap_bytes=8192,
            cap_bytes_bash=32768,
            bg_procs=self._bg_procs,
        )
        self._runtime = ToolRuntime(
            policy=Policy.from_config(cfg.agents.permissions),
            ctx=self._tool_ctx,
            prompt_hook=self._prompt_hook,
        )
        self._max_iterations = cfg.agents.max_iterations
        self._router_config = cfg.agents.router
        self._evaluator_config = cfg.agents.evaluator

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
        self.title = "vllama chat"
        self.sub_title = self._model
        self._update_status()
        self.query_one("#user-input", ChatInput).focus()

        container = self.query_one("#messages", ScrollableContainer)

        def _on_scroll_y(value: float) -> None:
            self._follow = (container.max_scroll_y - value) <= 2

        self.watch(container, "scroll_y", _on_scroll_y)

        # Replay existing messages when resuming
        if self._session.message_count > 0:
            self.call_after_refresh(self._replay_history)

    async def on_unmount(self) -> None:
        await self._bg_procs.shutdown()

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
        "/permits": "Show current permission policy and session permits",
        "/bg": "List background shells",
        "/kill": "/kill <shell_id> — terminate a background shell",
        "/todos": "Show the current in-session todo list",
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

        elif cmd == "/permits":
            await self._show_note(render_permits_summary(self._runtime))

        elif cmd == "/bg":
            await self._show_note(render_bg_listing(self._bg_procs))

        elif cmd == "/kill":
            if not arg:
                await self._show_note("Usage: /kill <shell_id>")
            elif self._bg_procs.get(arg) is None:
                await self._show_note(f"Unknown shell_id: {arg}")
            else:
                ok = await self._bg_procs.kill(arg)
                if ok:
                    await self._show_note(f"Killed {arg}")
                else:
                    await self._show_note(f"Could not kill {arg} (already exited?)")

        elif cmd == "/todos":
            await self._show_note(render_todos_summary(self._tool_ctx.todos))

        else:
            await self._show_note(f"Unknown command '{cmd}'. Try /help.")

    async def _show_note(self, text: str) -> None:
        bubble = Message("system-note", text)
        await self.query_one("#messages").mount(bubble)
        self._scroll_to_bottom()

    async def _prompt_hook(self, name: str, args: dict[str, object]) -> PromptDecision:
        """Push PermissionModal and await user's choice."""
        future: asyncio.Future[PromptDecision] = asyncio.get_running_loop().create_future()

        def _on_close(choice: str | None) -> None:
            if choice is None:
                future.set_result("deny_abort")
            elif not future.done():
                future.set_result(choice)  # type: ignore[arg-type]

        await self.push_screen(PermissionModal(name=name, args=args), _on_close)
        return await future

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
            for iter_idx in range(1, self._max_iterations + 1):
                self.iteration = iter_idx
                self._update_status()

                # 1. Router — pick a tool subset for this iteration.
                selected = await select_tools(
                    history=self._history,
                    all_specs=all_specs(),
                    llm_call=self._llm_json_call,
                    config=self._router_config,
                    iteration=iter_idx,
                )
                tools_payload = [s.definition for s in selected]

                # 2. Main stream.
                assistant_bubble = Message("assistant")
                await self.query_one("#messages").mount(assistant_bubble)
                self._scroll_to_bottom()

                full_response = ""
                full_thinking = ""
                was_thinking = False
                tool_calls: dict[int, dict] = {}

                async for event in self._stream_chat(self._history, tools_payload):
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
                    # Save final assistant message first, then ask the evaluator.
                    if full_response or full_thinking:
                        assistant_msg_final: dict = {
                            "role": "assistant",
                            "content": full_response,
                        }
                        if full_thinking:
                            assistant_msg_final["thinking"] = full_thinking
                        self._history.append(assistant_msg_final)
                        self._session.messages = self._history
                        save_session(self._sessions_dir, self._session)

                    verdict = await check_done(
                        history=self._history,
                        llm_call=self._llm_json_call,
                        config=self._evaluator_config,
                        todos=self._tool_ctx.todos,
                    )
                    if verdict.done:
                        break
                    self._history.append(
                        {
                            "role": "system",
                            "content": (
                                f"Task not yet complete: {verdict.reason}. Continue."
                            ),
                        }
                    )
                    continue

                # Build the assistant message with tool_calls for history.
                assistant_msg_tc: dict = {"role": "assistant"}
                assistant_msg_tc["content"] = full_response if full_response else None
                assistant_msg_tc["tool_calls"] = [
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
                self._history.append(assistant_msg_tc)

                try:
                    for tc in sorted(tool_calls.values(), key=lambda t: t["id"]):
                        tool_result = await self._runtime.dispatch(
                            tc["name"], tc["arguments"]
                        )
                        result = tool_result.content

                        self._history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            }
                        )

                        await assistant_bubble.append_tool_call(
                            tc["name"],
                            tc["arguments"],
                            result,
                        )
                        self._scroll_to_bottom()
                except AbortTurn:
                    await self._show_note("Turn aborted by user.")
                    return

                self._session.messages = self._history
            else:
                # for/else: runs only when the loop completes without break.
                await self._show_note(
                    f"Max iterations ({self._max_iterations}) reached. Turn ended."
                )

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
        self.iteration = 0

    async def _llm_json_call(self, messages: list[dict[str, Any]]) -> str:
        """Router/evaluator JSON call — non-streaming, short-lived client."""
        async with httpx.AsyncClient() as client:
            return await call_chat_json(
                client=client,
                base_url=self._base_url,
                model=self._model,
                messages=messages,
                api_key=self._api_key,
                temperature=0.0,
                timeout=30.0,
            )

    async def _stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield event dicts: {kind: "thinking"/"content"/"tool_call", ...}."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "tools": tools,
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
        if self.iteration > 0:
            parts.append(f"iter {self.iteration}/{self._max_iterations}")
        self.query_one("#status-label", Label).update("  •  ".join(parts))

    def _scroll_to_bottom(self, force: bool = False) -> None:
        container = self.query_one("#messages", ScrollableContainer)
        if force:
            self._follow = True
        if self._follow:
            container.scroll_end(animate=False)
