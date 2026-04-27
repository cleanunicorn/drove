"""TUI for browsing observed API requests and responses."""

from __future__ import annotations

import json
from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    Tree,
)
from textual.widgets._tree import TreeNode

from drove.observe import ObserveRecord, list_records, load_record, record_matches

# ── Styles ─────────────────────────────────────────────────────────────────────

CSS = """
ObserveApp {
    background: $surface;
}

#main-container {
    height: 1fr;
}

#record-list {
    width: 1fr;
    min-width: 40;
    max-width: 80;
    height: 1fr;
    border-right: solid $panel;
}

#search-input {
    margin: 0;
    border: tall $panel;
}

#record-table {
    height: 1fr;
}

#detail-pane {
    width: 2fr;
    height: 1fr;
    padding: 1 2;
    overflow-y: auto;
}

#detail-placeholder {
    color: $text-muted;
    text-style: italic;
    padding: 2 4;
}

.detail-section {
    margin-bottom: 1;
}

.detail-section Label {
    width: 1fr;
}

.section-content {
    padding-left: 2;
    color: $text;
    width: 1fr;
}

.detail-section Tree {
    height: auto;
    max-height: 30;
    padding-left: 2;
}

.metrics-label {
    padding-left: 2;
    color: $accent;
}

#status-bar {
    height: 1;
    padding: 0 2;
    background: $panel;
    color: $text-muted;
}
"""


def _pretty_json(raw: str | None) -> str:
    """Try to pretty-print JSON, return as-is if not valid JSON."""
    if raw is None:
        return "(empty)"
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError, ValueError:
        return raw


def _syntax_widget(text: str, lexer: str = "json") -> Static:
    """Return a Static widget with syntax-highlighted content."""
    syntax = Syntax(
        text,
        lexer,
        theme="monokai",
        word_wrap=True,
        padding=(0, 1),
    )
    widget = Static(syntax, classes="section-content")
    return widget


def _headers_widget(headers: dict[str, str]) -> Static:
    """Return a Static widget with highlighted HTTP headers."""
    if not headers:
        content = Text("(none)", style="dim italic")
    else:
        content = Text()
        for i, (k, v) in enumerate(headers.items()):
            if i > 0:
                content.append("\n")
            content.append(k, style="bold #9cdcfe")
            content.append(": ")
            content.append(v, style="#ce9178")
    return Static(content, classes="section-content")


def _json_tree(raw: str | None, root_label: str = "root") -> Tree[str]:
    """Build a Textual Tree widget from a JSON string."""
    tree: Tree[str] = Tree(root_label, classes="section-content")
    tree.show_root = False
    if raw is None:
        tree.root.add_leaf("(empty)")
        return tree
    try:
        data = json.loads(raw)
    except json.JSONDecodeError, ValueError:
        tree.root.add_leaf(raw)
        return tree
    _add_json_node(tree.root, data)
    tree.root.expand_all()
    return tree


def _add_json_node(node: TreeNode[str], value: object, key: str | None = None) -> None:
    """Recursively add JSON data to a tree node."""
    prefix = Text()
    if key is not None:
        prefix.append(f'"{key}"', style="bold #9cdcfe")
        prefix.append(": ")

    if isinstance(value, dict):
        label = prefix + Text("{...}", style="dim") if value else prefix + Text("{}")
        branch = node.add(label)
        for k, v in value.items():
            _add_json_node(branch, v, key=k)
    elif isinstance(value, list):
        count = len(value)
        label = prefix + Text(f"[{count} items]", style="dim") if value else prefix + Text("[]")
        branch = node.add(label)
        for i, v in enumerate(value):
            _add_json_node(branch, v, key=str(i))
    elif isinstance(value, str):
        val_text = Text(json.dumps(value), style="#ce9178")
        node.add_leaf(prefix + val_text)
    elif isinstance(value, bool):
        val_text = Text(str(value).lower(), style="bold #569cd6")
        node.add_leaf(prefix + val_text)
    elif isinstance(value, int | float):
        val_text = Text(str(value), style="#b5cea8")
        node.add_leaf(prefix + val_text)
    elif value is None:
        val_text = Text("null", style="bold #569cd6")
        node.add_leaf(prefix + val_text)
    else:
        node.add_leaf(prefix + Text(str(value)))


def _truncate(text: str, max_len: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ── Detail widget ──────────────────────────────────────────────────────────────


class RecordDetail(Static):
    """Shows the full detail of a selected observe record."""

    def __init__(self) -> None:
        super().__init__()
        self._record: ObserveRecord | None = None

    def compose(self) -> ComposeResult:
        yield Label("Select a request from the list to inspect.", id="detail-placeholder")

    async def show_record(self, record: ObserveRecord) -> None:
        self._record = record
        await self.remove_children()

        # Metrics summary
        metrics_parts = [f"Status: {record.response_status}"]
        metrics_parts.append(f"Duration: {record.duration_seconds:.3f}s")
        if record.ttft_seconds is not None:
            metrics_parts.append(f"TTFT: {record.ttft_seconds:.3f}s")
        if record.tokens_prompt or record.tokens_completion:
            metrics_parts.append(
                f"Tokens: {record.tokens_prompt} prompt + {record.tokens_completion} completion"
            )
        if record.tokens_per_second is not None:
            metrics_parts.append(f"Speed: {record.tokens_per_second:.1f} tok/s")

        await self.mount(
            Label(
                f"[bold]{record.method} /{record.endpoint}[/bold]  "
                f"({record.model or 'unknown'})  {record.timestamp}",
                markup=True,
            )
        )
        await self.mount(Label("  ".join(metrics_parts), classes="metrics-label"))

        # Request headers
        await self.mount(
            Collapsible(
                _headers_widget(record.request_headers),
                title="Request Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Request body
        await self.mount(
            Collapsible(
                _json_tree(record.request_body, "request"),
                title=f"Request Body ({len(record.request_body or '')} chars)",
                collapsed=False,
                classes="detail-section",
            )
        )

        # Response headers
        await self.mount(
            Collapsible(
                _headers_widget(record.response_headers),
                title="Response Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Response body (assembled/readable)
        body_len = len(record.response_body)
        await self.mount(
            Collapsible(
                _json_tree(record.response_body, "response"),
                title=f"Response ({body_len} chars)",
                collapsed=False,
                classes="detail-section",
            )
        )

        # Raw response (SSE stream) — hidden by default
        if record.response_body_raw and record.response_body_raw != record.response_body:
            raw_len = len(record.response_body_raw)
            await self.mount(
                Collapsible(
                    _syntax_widget(record.response_body_raw, "text"),
                    title=f"Raw Response ({raw_len} chars)",
                    collapsed=True,
                    classes="detail-section",
                )
            )


# ── Main app ───────────────────────────────────────────────────────────────────


class ObserveApp(App[None]):
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("r", "refresh_list", "Refresh"),
        Binding("/", "focus_search", "Search"),
    ]

    def __init__(
        self,
        observe_dir: Path,
        model: str | None = None,
        theme: str = "textual-dark",
    ) -> None:
        super().__init__()
        self._observe_dir = observe_dir
        self._model = model
        self.theme = theme
        self._records: list[tuple[Path, ObserveRecord]] = []
        self._all_records: list[tuple[Path, ObserveRecord]] = []
        self._search: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="record-list"):
                yield Input(placeholder="Search anything…  (press / to focus)", id="search-input")
                table: DataTable[str] = DataTable(
                    id="record-table", cursor_type="row", zebra_stripes=True
                )
                yield table
            with ScrollableContainer(id="detail-pane"):
                yield RecordDetail()
        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "drove observe"
        if self._model:
            self.sub_title = f"model: {self._model}"
        self._load_records()
        if self._records:
            self.query_one("#record-table", DataTable).focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self._search = event.value
            self._apply_filter()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input" and self._records:
            self.query_one("#record-table", DataTable).focus()

    def _load_records(self) -> None:
        self._all_records = list_records(self._observe_dir, self._model)
        self._apply_filter()

    def _apply_filter(self) -> None:
        if self._search:
            self._records = [
                (p, r) for (p, r) in self._all_records if record_matches(r, self._search)
            ]
        else:
            self._records = list(self._all_records)

        table = self.query_one("#record-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Time", "Model", "Endpoint", "Status", "Tokens", "Speed")

        for _path, record in self._records:
            time_str = record.timestamp[11:19] if len(record.timestamp) >= 19 else record.timestamp
            model = _truncate(record.model or "?", 15)
            endpoint = _truncate(record.endpoint, 20)
            status = str(record.response_status)
            tokens = (
                f"{record.tokens_prompt}+{record.tokens_completion}"
                if record.tokens_prompt or record.tokens_completion
                else "-"
            )
            speed = f"{record.tokens_per_second:.1f}" if record.tokens_per_second else "-"
            table.add_row(time_str, model, endpoint, status, tokens, speed)

        total = len(self._all_records)
        shown = len(self._records)
        if self._search:
            status_text = f"  {shown}/{total} records  (search: {self._search!r})"
        else:
            status_text = f"  {total} records"
        if self._model:
            status_text += f" (model: {self._model})"
        self.query_one("#status-bar", Label).update(status_text)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._records):
            path, record = self._records[idx]
            # Reload full record from disk in case list had a stale copy
            try:
                full_record = load_record(path)
            except Exception:
                full_record = record
            detail = self.query_one(RecordDetail)
            self.run_worker(detail.show_record(full_record), exclusive=True)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        idx = event.cursor_row
        if 0 <= idx < len(self._records):
            path, record = self._records[idx]
            try:
                full_record = load_record(path)
            except Exception:
                full_record = record
            detail = self.query_one(RecordDetail)
            self.run_worker(detail.show_record(full_record), exclusive=True)

    def action_refresh_list(self) -> None:
        self._load_records()
