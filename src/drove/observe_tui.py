"""TUI for browsing observed API requests and responses."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.theme import Theme
from textual.timer import Timer
from textual.widgets import (
    Button,
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

from drove.observe import ObserveRecord, list_records_page, search_records_page

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
    color: $text;
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
    color: $text;
}

#status-bar {
    height: 1;
    padding: 0 2;
    background: $panel;
    color: $text;
}

#page-controls { height: 3; }
#page-controls Button { min-width: 10; width: 1fr; }
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


def _luminance(color: Color) -> float:
    channels = [value / 255 for value in (color.r, color.g, color.b)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
    return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))


def _contrast(first: Color, second: Color) -> float:
    a, b = sorted((_luminance(first), _luminance(second)))
    return (b + 0.05) / (a + 0.05)


@dataclass(frozen=True)
class DetailPalette:
    text: str
    key: str
    string: str
    number: str
    literal: str
    background: str
    dark: bool

    @classmethod
    def from_app(cls, app: App[None]) -> DetailPalette:
        variables = app.get_css_variables()
        background = Color.parse(variables["surface"])
        foreground = Color.parse(variables["foreground"])

        def readable(name: str) -> str:
            color = Color.parse(variables[name])
            color = background.blend(color, color.a)
            return (color if _contrast(color, background) >= 4.5 else foreground).hex

        return cls(
            foreground.hex,
            readable("accent"),
            readable("success"),
            readable("warning"),
            readable("primary"),
            background.hex,
            app.current_theme.dark,
        )


def _syntax_widget(text: str, palette: DetailPalette, lexer: str = "json") -> Static:
    """Return a Static widget with syntax-highlighted content."""
    syntax = Syntax(
        text,
        lexer,
        theme="monokai" if palette.dark else "friendly",
        background_color=palette.background,
        word_wrap=True,
        padding=(0, 1),
    )
    widget = Static(syntax, classes="section-content")
    return widget


def _headers_widget(headers: dict[str, str], palette: DetailPalette) -> Static:
    """Return a Static widget with highlighted HTTP headers."""
    if not headers:
        content = Text("(none)", style=f"italic {palette.text}")
    else:
        content = Text()
        for i, (k, v) in enumerate(headers.items()):
            if i > 0:
                content.append("\n")
            content.append(k, style=f"bold {palette.key}")
            content.append(": ")
            content.append(v, style=palette.string)
    return Static(content, classes="section-content")


def _json_tree(raw: str | None, palette: DetailPalette, root_label: str = "root") -> Tree[str]:
    """Build a Textual Tree widget from a JSON string."""
    tree: Tree[str] = Tree(root_label, classes="section-content")
    tree.show_root = False
    if raw is None:
        tree.root.add_leaf("(empty)")
        return tree
    try:
        data = json.loads(raw)
    except json.JSONDecodeError, ValueError:
        tree.root.add_leaf(Text(raw, style=palette.text))
        return tree
    _add_json_node(tree.root, data, palette)
    tree.root.expand_all()
    return tree


def _add_json_node(
    node: TreeNode[str], value: object, palette: DetailPalette, key: str | None = None
) -> None:
    """Recursively add JSON data to a tree node."""
    prefix = Text()
    if key is not None:
        prefix.append(f'"{key}"', style=f"bold {palette.key}")
        prefix.append(": ")

    if isinstance(value, dict):
        label = prefix + Text("{...}", style=palette.text) if value else prefix + Text("{}")
        branch = node.add(label)
        for k, v in value.items():
            _add_json_node(branch, v, palette, key=k)
    elif isinstance(value, list):
        count = len(value)
        label = (
            prefix + Text(f"[{count} items]", style=palette.text) if value else prefix + Text("[]")
        )
        branch = node.add(label)
        for i, v in enumerate(value):
            _add_json_node(branch, v, palette, key=str(i))
    elif isinstance(value, str):
        val_text = Text(json.dumps(value), style=palette.string)
        node.add_leaf(prefix + val_text)
    elif isinstance(value, bool):
        val_text = Text(str(value).lower(), style=f"bold {palette.literal}")
        node.add_leaf(prefix + val_text)
    elif isinstance(value, int | float):
        val_text = Text(str(value), style=palette.number)
        node.add_leaf(prefix + val_text)
    elif value is None:
        val_text = Text("null", style=f"bold {palette.literal}")
        node.add_leaf(prefix + val_text)
    else:
        node.add_leaf(prefix + Text(str(value)))


# ── Detail widget ──────────────────────────────────────────────────────────────


class RecordDetail(Static):
    """Shows the full detail of a selected observe record."""

    def __init__(self) -> None:
        super().__init__()
        self._record: ObserveRecord | None = None
        self._revision = 0
        self._lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield Label("Select a request from the list to inspect.", id="detail-placeholder")

    async def show_record(self, record: ObserveRecord | None, message: str = "") -> None:
        self._revision += 1
        revision = self._revision
        async with self._lock:
            if revision != self._revision or not self.is_mounted:
                return
            await self.remove_children()
            if revision != self._revision or not self.is_mounted:
                return
            self._record = record
            if record is None:
                await self.mount(Label(message, id="detail-placeholder", markup=False))
            else:
                await self._mount_record(record)

    async def _mount_record(self, record: ObserveRecord) -> None:
        palette = DetailPalette.from_app(self.app)

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
                f"{record.method} /{record.endpoint}  "
                f"({record.model or 'unknown'})  {record.timestamp}",
                markup=False,
            )
        )
        await self.mount(Label("  ".join(metrics_parts), classes="metrics-label"))

        # Request headers
        await self.mount(
            Collapsible(
                _headers_widget(record.request_headers, palette),
                title="Request Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Request body
        await self.mount(
            Collapsible(
                _json_tree(record.request_body, palette, "request"),
                title=f"Request Body ({len(record.request_body or '')} chars)",
                collapsed=False,
                classes="detail-section",
            )
        )

        # Response headers
        await self.mount(
            Collapsible(
                _headers_widget(record.response_headers, palette),
                title="Response Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Response body (assembled/readable)
        body_len = len(record.response_body)
        await self.mount(
            Collapsible(
                _json_tree(record.response_body, palette, "response"),
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
                    _syntax_widget(record.response_body_raw, palette, "text"),
                    title=f"Raw Response ({raw_len} chars)",
                    collapsed=True,
                    classes="detail-section",
                )
            )


# ── Main app ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageRequest:
    generation: int
    query: str
    offset: int


class ObserveApp(App[None]):
    CSS = CSS
    PAGE_SIZE = 100
    SEARCH_DELAY = 0.2

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "escape_search", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("r", "refresh_list", "Refresh"),
        Binding("/", "focus_search", "Search"),
    ]

    def __init__(
        self, observe_dir: Path, model: str | None = None, theme: str = "textual-dark"
    ) -> None:
        super().__init__()
        self._observe_dir = observe_dir
        self._model = model
        self.theme = theme
        self._records: list[tuple[Path, ObserveRecord]] = []
        self._rows: dict[str, ObserveRecord] = {}
        self._search = ""
        self._offset = 0
        self._total = 0
        self._generation = 0
        self._pending: PageRequest | None = None
        self._query_running = False
        self._loading = False
        self._closed = False
        self._stop_scan = Event()
        self._debounce: Timer | None = None
        self._selected: str | None = None
        self._detail_version = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="record-list"):
                yield Input(placeholder="Search anything…  (press / to focus)", id="search-input")
                yield DataTable(id="record-table", cursor_type="row", zebra_stripes=True)
                with Horizontal(id="page-controls"):
                    yield Button("Previous", id="previous-page", disabled=True)
                    yield Button("Next", id="next-page", disabled=True)
            with ScrollableContainer(id="detail-pane"):
                yield RecordDetail()
        yield Label("", id="status-bar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "drove observe"
        if self._model:
            self.sub_title = f"model: {self._model}"
        table = self.query_one(DataTable)
        table.add_columns("Time", "Model", "Endpoint", "Status", "Tokens", "Speed")
        table.focus()
        self.theme_changed_signal.subscribe(self, self._theme_changed)
        self._request_page()

    def on_unmount(self) -> None:
        self._stop_scan.set()
        self._closed = True
        self._generation += 1
        self._pending = None
        if self._debounce:
            self._debounce.stop()

    def _theme_changed(self, theme: Theme) -> None:
        if self._selected in self._rows:
            self._show_detail(self._rows[self._selected])

    def action_focus_search(self) -> None:
        self.query_one(Input).focus()

    def action_escape_search(self) -> None:
        search = self.query_one(Input)
        if search.has_focus:
            if search.value:
                search.value = ""
            else:
                self.query_one(DataTable).focus()
        else:
            self.exit()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input" and event.value != self._search:
            self._search = event.value
            self._offset = 0
            self._request_page(debounce=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            self.query_one(DataTable).focus()

    def _request_page(self, *, debounce: bool = False) -> None:
        self._generation += 1
        self._pending = None
        self._selected = None
        self._rows.clear()
        self._records.clear()
        self._loading = True
        if self._debounce:
            self._debounce.stop()
            self._debounce = None
        self.query_one(DataTable).clear()
        self._set_status("Searching…" if self._search else "Loading requests…")
        self._show_detail(None, "Searching…" if self._search else "Loading requests…")
        self._update_buttons()
        if debounce:
            self._debounce = self.set_timer(self.SEARCH_DELAY, self._enqueue_page)
        else:
            self._enqueue_page()

    def _enqueue_page(self) -> None:
        if self._closed:
            return
        self._pending = PageRequest(self._generation, self._search, self._offset)
        if not self._query_running:
            self._query_running = True
            self.run_worker(self._query_loop(), group="pages", exit_on_error=False)

    def _read_page(self, request: PageRequest) -> tuple[list[tuple[Path, ObserveRecord]], int]:
        if request.query:
            return search_records_page(
                self._observe_dir,
                request.query,
                self._model,
                request.offset,
                self.PAGE_SIZE,
                stop=self._stop_scan,
            )
        return list_records_page(
            self._observe_dir, self._model, request.offset, self.PAGE_SIZE, stop=self._stop_scan
        )

    async def _query_loop(self) -> None:
        try:
            while not self._closed and self._pending is not None:
                request, self._pending = self._pending, None
                try:
                    records, total = await asyncio.to_thread(self._read_page, request)
                except Exception:
                    if request.generation == self._generation and not self._closed:
                        self._loading = False
                        self._set_status("Unable to read requests. Press r to retry.")
                        self._show_detail(None, "Unable to read requests. Press r to retry.")
                        self._update_buttons()
                    continue
                if self._closed or request.generation != self._generation:
                    continue
                if request.offset and request.offset >= total:
                    self._offset = max(0, ((total - 1) // self.PAGE_SIZE) * self.PAGE_SIZE)
                    self._request_page()
                    continue
                self._records = records
                self._total = total
                self._loading = False
                self._render_page(request)
        finally:
            self._query_running = False

    def _render_page(self, request: PageRequest) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self._rows.clear()
        for path, record in self._records:
            key = f"{request.generation}:{path}"
            self._rows[key] = record
            tokens = (
                f"{record.tokens_prompt}+{record.tokens_completion}"
                if record.tokens_prompt or record.tokens_completion
                else "-"
            )
            table.add_row(
                Text(record.timestamp[11:19]),
                Text(record.model or "?"),
                Text(record.endpoint),
                str(record.response_status),
                tokens,
                f"{record.tokens_per_second:.1f}" if record.tokens_per_second else "-",
                key=key,
            )
        if self._records:
            end = min(request.offset + self.PAGE_SIZE, self._total)
            noun = "matches" if request.query else "records"
            self._set_status(
                f"{request.offset + 1}–{end} of {self._total} {noun} "
                f"({len(self._records)} readable). ←/→ scroll columns."
            )
            self._show_detail(None, "Select a request from the list to inspect.")
        else:
            if self._total:
                message = "No readable requests on this page. Try another page or refresh (r)."
            elif request.query:
                message = f"No matches for {request.query!r}. Clear the search to see all requests."
            else:
                message = (
                    "No requests recorded yet. Enable observe logging with observe = true "
                    "in your config, then make a request."
                )
            self._set_status(message)
            self._show_detail(None, message)
        self._update_buttons()

    def _set_status(self, text: str) -> None:
        if self._model:
            text += f" (model: {self._model})"
        self.query_one("#status-bar", Label).update(text)

    def _update_buttons(self) -> None:
        self.query_one("#previous-page", Button).disabled = self._loading or self._offset == 0
        self.query_one("#next-page", Button).disabled = (
            self._loading or self._offset + self.PAGE_SIZE >= self._total
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._loading:
            return
        if event.button.id == "previous-page" and self._offset:
            self._offset = max(0, self._offset - self.PAGE_SIZE)
        elif event.button.id == "next-page" and self._offset + self.PAGE_SIZE < self._total:
            self._offset += self.PAGE_SIZE
        else:
            return
        self._request_page()

    def _show_detail(self, record: ObserveRecord | None, message: str = "") -> None:
        self._detail_version += 1
        version = self._detail_version

        async def render() -> None:
            if version == self._detail_version and not self._closed:
                await self.query_one(RecordDetail).show_record(record, message)

        self.run_worker(render(), group="details")

    def _select_row(self, key: str | None) -> None:
        if key is None or key not in self._rows:
            return
        self._selected = key
        self._show_detail(self._rows[key])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_row(event.row_key.value)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_row(event.row_key.value)

    def action_refresh_list(self) -> None:
        self._offset = 0
        self._request_page()
