"""Interaction, scheduling and rendering regressions for the observe browser."""

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.syntax import Syntax
from rich.text import Text
from textual.color import Color
from textual.widgets import Button, DataTable, Input, Label, Static, Tree

from drove.observe import ObserveRecord, save_record
from drove.observe_tui import DetailPalette, ObserveApp, PageRequest, RecordDetail, _contrast


def record(name: str = "org/model-with-a-long-name:Q4", number: int = 1) -> ObserveRecord:
    return ObserveRecord(
        id=f"20261025-{number:06d}-abcd",
        timestamp="2026-10-25T12:00:00",
        model=name,
        endpoint="v1/chat/completions-with-long-name",
        method="POST",
        request_headers={"Authorization": "literal [value]"},
        request_body='{"nested": [true, null, 12, "hello"]}',
        response_status=200,
        response_headers={"Content-Type": "application/json"},
        response_body='{"reply": "hello"}',
        response_body_raw="data: [literal]",
    )


async def test_escape_in_search_never_quits(tmp_path: Path) -> None:
    app = ObserveApp(tmp_path)
    async with app.run_test() as pilot:
        search = app.query_one(Input)
        search.focus()
        await pilot.press("q", "r", "/")
        assert search.value == "qr/"
        await pilot.press("escape")
        assert app.is_running
        assert search.value == ""
        await pilot.press("escape")
        assert app.is_running
        assert app.query_one(DataTable).has_focus


async def test_model_and_endpoint_are_not_truncated(tmp_path: Path) -> None:
    names = ["org/identical-prefix-long-name:Q4", "org/identical-prefix-long-name:Q8"]
    for i, name in enumerate(names):
        save_record(tmp_path, record(name, i))
    app = ObserveApp(tmp_path)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        table = app.query_one(DataTable)
        rows = [table.get_row_at(i) for i in range(table.row_count)]
        assert {str(row[1]) for row in rows} == set(names)
        assert all(str(row[2]) == "v1/chat/completions-with-long-name" for row in rows)


async def until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


async def loaded(app: ObserveApp) -> None:
    await until(lambda: not app._loading and not app._query_running)


def labels(app: ObserveApp) -> str:
    return "\n".join(str(label.content) for label in app.query(Label))


@pytest.mark.parametrize("missing", [False, True])
async def test_empty_and_literal_no_match_states(tmp_path: Path, missing: bool) -> None:
    directory = tmp_path / "missing" if missing else tmp_path
    app = ObserveApp(directory, model="org/[bold]literal[/bold]")
    app.SEARCH_DELAY = 0.01
    async with app.run_test() as pilot:
        await loaded(app)
        await pilot.pause()
        assert "No requests recorded yet" in labels(app)
        assert "observe = true" in labels(app)
        search = app.query_one(Input)
        search.value = "[bold]missing[/bold]"
        await until(lambda: app._search == search.value)
        await loaded(app)
        await pilot.pause()
        assert "No matches for '[bold]missing[/bold]'" in labels(app)
        assert "model: org/[bold]literal[/bold]" in labels(app)
        search.focus()
        await pilot.press("escape")
        await loaded(app)
        await pilot.pause()
        assert "No requests recorded yet" in labels(app)


async def test_paging_search_refresh_and_model_scope(tmp_path: Path) -> None:
    for n in range(205):
        save_record(tmp_path, record("alpha", n))
    save_record(tmp_path, record("beta", 999))
    app = ObserveApp(tmp_path, model="alpha")
    app.SEARCH_DELAY = 0.01
    async with app.run_test(size=(120, 35)) as pilot:
        await loaded(app)
        table = app.query_one(DataTable)
        assert table.row_count == 100
        assert app._total == 205
        assert app._records[0][1].id.startswith("20261025-000204")
        await pilot.click("#next-page")
        await loaded(app)
        assert app._offset == 100
        assert table.row_count == 100
        await pilot.click("#next-page")
        await loaded(app)
        assert app._offset == 200
        assert table.row_count == 5
        assert app.query_one("#next-page", Button).disabled
        await pilot.click("#previous-page")
        await loaded(app)
        assert app._offset == 100
        app.query_one(Input).value = "000003"
        await until(lambda: app._search == "000003")
        await loaded(app)
        assert app._offset == 0
        assert table.row_count == 1
        assert app._total == 1
        assert app._records[0][1].id.startswith("20261025-000003")
        app.action_refresh_list()
        await loaded(app)
        assert table.row_count == 1
        assert app._search == "000003"


async def test_corrupt_window_and_shrinking_directory(tmp_path: Path) -> None:
    folder = tmp_path / "alpha"
    folder.mkdir()
    for n in range(1, 101):
        (folder / f"20261025-{n:06d}-bad.json").write_text("broken")
    path = save_record(tmp_path, record("alpha", 0))
    app = ObserveApp(tmp_path)
    async with app.run_test() as pilot:
        await loaded(app)
        await pilot.pause()
        assert app._total == 101
        assert app.query_one(DataTable).row_count == 0
        assert "No readable requests on this page" in labels(app)
        assert not app.query_one("#next-page", Button).disabled
        await pilot.click("#next-page")
        await loaded(app)
        assert app._offset == 100
        assert app.query_one(DataTable).row_count == 1
        for file in folder.glob("*bad.json"):
            file.unlink()
        app._request_page()
        await loaded(app)
        assert app._offset == 0
        assert app._total == 1
        assert app._records[0][0] == path


async def test_io_error_is_visible_and_refresh_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = ObserveApp(tmp_path)
    original = app._read_page

    def fail(request: PageRequest) -> tuple[list[tuple[Path, ObserveRecord]], int]:
        raise PermissionError("secret path")

    monkeypatch.setattr(app, "_read_page", fail)
    async with app.run_test() as pilot:
        await loaded(app)
        await pilot.pause()
        assert "Unable to read requests" in labels(app)
        assert "secret path" not in labels(app)
        monkeypatch.setattr(app, "_read_page", original)
        app.action_refresh_list()
        await loaded(app)
        await pilot.pause()
        assert "No requests recorded yet" in labels(app)


@pytest.mark.parametrize("stale_error", [False, True])
async def test_search_coalesces_and_rejects_stale_work_without_blocking_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_error: bool
) -> None:
    app = ObserveApp(tmp_path)
    app.SEARCH_DELAY = 0.04
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []
    active = 0
    maximum = 0

    def slow(request: PageRequest) -> tuple[list[tuple[Path, ObserveRecord]], int]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        calls.append(request.query)
        try:
            if request.query == "old":
                started.set()
                assert release.wait(5)
                if stale_error:
                    raise OSError("stale error")
            found = record(request.query or "initial")
            return [(tmp_path / "record.json", found)], 1
        finally:
            active -= 1

    monkeypatch.setattr(app, "_read_page", slow)
    try:
        async with app.run_test() as pilot:
            await loaded(app)
            search = app.query_one(Input)
            search.value = "old"
            await until(started.is_set)
            search.focus()
            await pilot.press("end", "a", "b", "c")
            assert search.value == "oldabc"
            assert app.is_running
            await asyncio.sleep(app.SEARCH_DELAY * 2)
            assert calls == ["", "old"]
            release.set()
            await loaded(app)
            await pilot.pause()
            assert calls == ["", "old", "oldabc"]
            assert maximum == 1
            assert app._records[0][1].model == "oldabc"
            assert "Unable to read" not in labels(app)
    finally:
        release.set()


async def test_clear_search_invalidates_inflight_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = ObserveApp(tmp_path)
    app.SEARCH_DELAY = 0.01
    started, release = threading.Event(), threading.Event()

    def slow(request: PageRequest) -> tuple[list[tuple[Path, ObserveRecord]], int]:
        if request.query:
            started.set()
            assert release.wait(5)
            return [(tmp_path / "old", record("stale"))], 1
        return [], 0

    monkeypatch.setattr(app, "_read_page", slow)
    try:
        async with app.run_test() as pilot:
            await loaded(app)
            search = app.query_one(Input)
            search.focus()
            search.value = "old"
            await until(started.is_set)
            await pilot.press("escape")
            await asyncio.sleep(0.03)
            release.set()
            await loaded(app)
            await pilot.pause()
            assert app._records == []
            assert "No requests recorded yet" in labels(app)
            assert app.is_running
    finally:
        release.set()


async def test_teardown_does_not_start_queued_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = ObserveApp(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def slow(request: PageRequest) -> tuple[list[tuple[Path, ObserveRecord]], int]:
        calls.append(request.query)
        started.set()
        assert release.wait(5)
        return [], 0

    monkeypatch.setattr(app, "_read_page", slow)
    try:
        async with app.run_test():
            await until(started.is_set)
            app._search = "queued"
            app._request_page()
        release.set()
        await asyncio.sleep(0.05)
        assert calls == [""]
        assert app._closed
    finally:
        release.set()


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_details_follow_theme_and_preserve_literal_content(
    tmp_path: Path, theme: str
) -> None:
    found = record("org/[bold]literal")
    found.response_body = "invalid [bold] json [/x]"
    save_record(tmp_path, found)
    app = ObserveApp(tmp_path, theme=theme)
    async with app.run_test(size=(140, 45)) as pilot:
        await loaded(app)
        await pilot.press("enter")
        detail = app.query_one(RecordDetail)
        await until(lambda: len(detail.query(Tree)) == 2)
        palette = DetailPalette.from_app(app)
        for color in (palette.text, palette.key, palette.string, palette.number, palette.literal):
            assert _contrast(Color.parse(color), Color.parse(palette.background)) >= 4.5
        assert found.model in labels(app)
        trees = list(detail.query(Tree))
        assert str(trees[1].root.children[0].label) == found.response_body
        headers = [
            widget.content for widget in detail.query(Static) if isinstance(widget.content, Text)
        ]
        assert any("literal [value]" in str(content) for content in headers)
        raw = [
            widget.content for widget in detail.query(Static) if isinstance(widget.content, Syntax)
        ]
        assert raw and raw[0].code == "data: [literal]"
        old_palette = palette
        app.theme = "textual-light" if theme == "textual-dark" else "textual-dark"
        await pilot.pause()
        await until(lambda: DetailPalette.from_app(app).dark != old_palette.dark)
        await pilot.pause()
        new_headers = [
            widget.content for widget in detail.query(Static) if isinstance(widget.content, Text)
        ]
        assert new_headers != headers
        assert detail._record is found or detail._record.id == found.id


async def test_obsolete_row_keys_cannot_restore_details(tmp_path: Path) -> None:
    save_record(tmp_path, record("original"))
    app = ObserveApp(tmp_path)
    app.SEARCH_DELAY = 0.01
    async with app.run_test() as pilot:
        await loaded(app)
        key = next(iter(app._rows))
        app._select_row(key)
        await pilot.pause()
        app.query_one(Input).value = "missing"
        await until(lambda: app._search == "missing")
        await loaded(app)
        app._select_row(key)
        await pilot.pause()
        assert app.query_one(RecordDetail)._record is None
        assert "No matches" in labels(app)


async def test_narrow_table_can_scroll_to_distinct_suffixes(tmp_path: Path) -> None:
    for n in range(2):
        save_record(tmp_path, record("org/" + "long-name-" * 8 + f":Q{n}", n))
    app = ObserveApp(tmp_path)
    async with app.run_test(size=(80, 30)) as pilot:
        await loaded(app)
        table = app.query_one(DataTable)
        table.focus()
        await pilot.press(*(["right"] * 12))
        await pilot.pause()
        assert table.scroll_x > 0
        values = [str(table.get_row_at(i)[1]) for i in range(2)]
        assert values[0] != values[1]
        assert values[0].endswith(":Q1") and values[1].endswith(":Q0")


async def test_q_is_visible_quit_key(tmp_path: Path) -> None:
    app = ObserveApp(tmp_path)
    async with app.run_test() as pilot:
        await loaded(app)
        await pilot.pause()
        assert any(getattr(key, "key", None) == "q" for key in app.query("FooterKey"))
        app.query_one(DataTable).focus()
        await pilot.press("q")
        assert not app.is_running


@pytest.mark.parametrize("query", ["", "absent"])
def test_run_quits_during_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str) -> None:
    """Exercise App.run's executor shutdown, rather than an existing pytest loop."""
    import time

    import drove.observe as storage

    for n in range(40):
        save_record(tmp_path, record(number=n))
    started = threading.Event()
    reads = 0
    load = storage.load_record

    def slow_load(path: Path) -> ObserveRecord:
        nonlocal reads
        reads += 1
        started.set()
        time.sleep(0.04)
        return load(path)

    monkeypatch.setattr(storage, "load_record", slow_load)
    app = ObserveApp(tmp_path)
    app._search = query
    quit_at = 0.0

    async def quit_when_reading(pilot):
        nonlocal quit_at
        async with asyncio.timeout(5):
            while not started.is_set():
                await asyncio.sleep(0.005)
        quit_at = time.monotonic()
        await pilot.press("q")

    app.run(headless=True, auto_pilot=quit_when_reading)
    assert quit_at
    assert time.monotonic() - quit_at < 0.8
    assert reads < 10  # A full uncancelled scan takes at least 1.6 seconds.
    assert app._stop_scan.is_set()


def test_scan_cancellation_during_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import CancelledError

    from drove.observe import list_records_page

    folder = tmp_path / "model"
    folder.mkdir()
    stop = threading.Event()
    visited = []

    def entries(path: Path):
        for n in range(100):
            visited.append(n)
            if n == 3:
                stop.set()
            yield path / f"{n}.json"

    monkeypatch.setattr(Path, "iterdir", entries)
    with pytest.raises(CancelledError):
        list_records_page(tmp_path, model="model", stop=stop)
    assert visited == [0, 1, 2, 3]
