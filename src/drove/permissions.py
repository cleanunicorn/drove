"""Conversation-only editor for chat tool approvals; changes are staged until Apply."""

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Switch

from drove.tool_policy import READ_TOOLS, Grant, canonical_origin

TOOLS = (
    ("Read file", "read_file"),
    ("List files", "list_files"),
    ("Search files", "search_files"),
    ("Write file", "write_file"),
    ("Shell command", "shell_execute"),
    ("Fetch URL", "fetch_url"),
)
TOOL_LABELS = dict((value, label) for label, value in TOOLS)


def make_rule(tool: str, scope: str, cwd: Path, internal: bool = False) -> Grant:
    """Validate readable editor fields, without parsing serialized configuration tokens."""
    if tool not in TOOL_LABELS or not scope.strip():
        raise ValueError("Choose a tool and enter its scope.")
    if tool == "fetch_url":
        origin = canonical_origin(scope.strip())
        if origin is None:
            raise ValueError("Enter an http or https origin, for example https://example.com:443.")
        return Grant(tool, origin=origin[0], internal=internal)
    if tool == "shell_execute":
        return Grant(tool, directory=str(cwd), command=scope.strip())
    directory = Path(scope.strip()).expanduser()
    if not directory.is_absolute():
        directory = cwd / directory
    try:
        directory = directory.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Could not resolve that directory.") from exc
    if directory.exists() and not directory.is_dir():
        raise ValueError("Enter a directory, not a file.")
    return Grant(tool, directory=str(directory))


class PermissionsModal(ModalScreen[tuple[frozenset[Grant], bool] | None]):
    """A private draft: opening, editing, or cancelling never mutates chat policy."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    PermissionsModal { align: center middle; }
    #permissions-dialog { width: 90; max-width: 100%; height: 95%; border: solid $accent;
        background: $surface; padding: 0 1; }
    #permissions-dialog Label { height: auto; }
    #permission-rules { height: 8; min-height: 3; }
    #permission-fields { height: auto; }
    #permissions-dialog Horizontal { height: auto; }
    #permissions-dialog Button { min-width: 9; width: 1fr; }
    #permissions-dialog Switch { height: 3; }
    #permission-error { color: $error; }
    """

    def __init__(
        self,
        grants: frozenset[Grant],
        allow_all: bool,
        cwd: Path,
        remote: bool,
        configured: frozenset[Grant] = frozenset(),
    ) -> None:
        super().__init__()
        self.rules = sorted(grants, key=lambda g: g.describe())
        self._allow_all = allow_all
        self._cwd = cwd
        self._remote = remote
        self._configured = configured
        self._editing: int | None = None

    def _row(self, grant: Grant) -> str:
        scope = (
            f"exact {grant.command!r}, cwd {grant.directory}"
            if grant.command is not None
            else grant.directory or grant.origin or ""
        )
        if grant.internal:
            scope += " (including private addresses)"
        lifetime = (
            "saved default; override this conversation"
            if grant in self._configured
            else "this conversation"
        )
        inactive = (
            " — inactive: remote reads always ask"
            if self._remote and grant.tool in READ_TOOLS
            else ""
        )
        return f"{TOOL_LABELS[grant.tool]} | {scope} | {lifetime}{inactive}"

    def compose(self) -> ComposeResult:
        with Vertical(id="permissions-dialog"):
            yield Label("Chat permissions — current conversation")
            with VerticalScroll():
                yield Label(
                    "Apply changes only this conversation. New/clear/model/session load "
                    "restores saved defaults. Removing a saved default here hides it until then. "
                    "Cancel discards edits.",
                    markup=False,
                )
                yield Label(
                    "Local reads inside the working directory need no rule. Remote reads always "
                    "ask unless Allow all is on. Background-task status needs no approval.",
                    markup=False,
                )
                yield Label(f"Working directory: {self._cwd}", markup=False)
                yield Label(
                    "Allow all — every chat tool, including private fetches, without prompts"
                )
                yield Switch(self._allow_all, id="permission-allow-all")
                yield OptionList(
                    *(self._row(g) for g in self.rules), id="permission-rules", markup=False
                )
                with Vertical(id="permission-fields"):
                    yield Select(TOOLS, value="read_file", allow_blank=False, id="permission-tool")
                    yield Label("Scope: directory / exact shell command / http(s) origin")
                    yield Input(placeholder="Directory, command, or origin", id="permission-scope")
                    yield Label("Fetch only: include private/internal addresses")
                    yield Switch(False, id="permission-internal")
                    yield Label(
                        "Select a rule to edit; Save rule updates it. New rule starts another."
                    )
                    with Horizontal():
                        yield Button("New rule", id="permission-new")
                        yield Button("Save rule", id="permission-save")
                        yield Button("Remove", id="permission-remove")
                    yield Label("", id="permission-error", markup=False)
            with Horizontal():
                yield Button("Apply", variant="primary", id="permissions-apply")
                yield Button("Cancel", id="permissions-cancel")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._editing = event.option_index
        grant = self.rules[event.option_index]
        self.query_one("#permission-tool", Select).value = grant.tool
        self.query_one("#permission-scope", Input).value = (
            grant.command if grant.command is not None else grant.directory or grant.origin or ""
        )
        self.query_one("#permission-internal", Switch).value = grant.internal

    def _refresh_rules(self) -> None:
        self.query_one("#permission-rules", OptionList).clear_options().add_options(
            self._row(g) for g in self.rules
        )
        self._editing = None
        self.query_one("#permission-scope", Input).value = ""
        self.query_one("#permission-error", Label).update("")

    def _has_unsaved_rule(self) -> bool:
        scope = self.query_one("#permission-scope", Input).value
        if not scope.strip():
            return False
        if self._editing is None:
            return True
        try:
            draft = make_rule(
                str(self.query_one("#permission-tool", Select).value),
                scope,
                self._cwd,
                self.query_one("#permission-internal", Switch).value,
            )
        except ValueError:
            return True
        return draft != self.rules[self._editing]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button.id
        if button == "permissions-cancel":
            self.action_cancel()
        elif button == "permissions-apply":
            if self._has_unsaved_rule():
                self.query_one("#permission-error", Label).update(
                    "Save the rule or clear its scope first."
                )
                return
            self.dismiss(
                (frozenset(self.rules), self.query_one("#permission-allow-all", Switch).value)
            )
        elif button == "permission-new":
            self._editing = None
            self.query_one("#permission-scope", Input).value = ""
            self.query_one("#permission-internal", Switch).value = False
        elif button == "permission-remove":
            if self._editing is not None:
                del self.rules[self._editing]
                self._refresh_rules()
        elif button == "permission-save":
            try:
                rule = make_rule(
                    str(self.query_one("#permission-tool", Select).value),
                    self.query_one("#permission-scope", Input).value,
                    self._cwd,
                    self.query_one("#permission-internal", Switch).value,
                )
            except ValueError as exc:
                self.query_one("#permission-error", Label).update(str(exc))
                return
            if self._editing is not None:
                self.rules[self._editing] = rule
            elif rule not in self.rules:
                self.rules.append(rule)
            self._refresh_rules()
