"""TUI for browsing observed API requests and responses."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Collapsible,
    DataTable,
    Footer,
    Header,
    Label,
    Static,
)

from vllama.observe import ObserveRecord, list_records, load_record

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
        req_headers_text = "\n".join(f"  {k}: {v}" for k, v in record.request_headers.items())
        await self.mount(
            Collapsible(
                Label(req_headers_text or "(none)", markup=False, classes="section-content"),
                title="Request Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Request body
        req_body = _pretty_json(record.request_body)
        await self.mount(
            Collapsible(
                Label(req_body, markup=False, classes="section-content"),
                title=f"Request Body ({len(record.request_body or '')} chars)",
                collapsed=False,
                classes="detail-section",
            )
        )

        # Response headers
        resp_headers_text = "\n".join(f"  {k}: {v}" for k, v in record.response_headers.items())
        await self.mount(
            Collapsible(
                Label(resp_headers_text or "(none)", markup=False, classes="section-content"),
                title="Response Headers",
                collapsed=True,
                classes="detail-section",
            )
        )

        # Response body (assembled/readable)
        resp_body = _pretty_json(record.response_body)
        body_len = len(record.response_body)
        await self.mount(
            Collapsible(
                Label(resp_body, markup=False, classes="section-content"),
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
                    Label(record.response_body_raw, markup=False, classes="section-content"),
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="record-list"):
                table: DataTable[str] = DataTable(
                    id="record-table", cursor_type="row", zebra_stripes=True
                )
                yield table
            with ScrollableContainer(id="detail-pane"):
                yield RecordDetail()
        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "vllama observe"
        if self._model:
            self.sub_title = f"model: {self._model}"
        self._load_records()

    def _load_records(self) -> None:
        self._records = list_records(self._observe_dir, self._model)

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

        status_text = f"  {len(self._records)} records"
        if self._model:
            status_text += f" (model: {self._model})"
        self.query_one("#status-bar", Label).update(status_text)

        if self._records:
            table.focus()

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
