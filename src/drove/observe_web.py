"""Web UI for browsing observed API requests and responses."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from drove.observe import (
    _record_to_dict,
    list_records,
    list_records_page,
    load_record,
    record_matches,
)

_STATIC_DIR = Path(__file__).parent / "observe_web_static"
_INDEX_PATH = _STATIC_DIR / "index.html"

_RECORD_SUMMARY_FIELDS = (
    "id",
    "timestamp",
    "model",
    "endpoint",
    "method",
    "response_status",
    "tokens_prompt",
    "tokens_completion",
    "tokens_per_second",
    "ttft_seconds",
    "duration_seconds",
)


def _summarize(record: object) -> dict[str, object]:
    return {field: getattr(record, field) for field in _RECORD_SUMMARY_FIELDS}


def create_observe_app(observe_dir: Path, model: str | None = None) -> FastAPI:
    app = FastAPI(title="drove observe")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/api/records")
    async def get_records(
        search: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> JSONResponse:
        if search:
            # Search must scan every record to filter, so the full set is
            # loaded and then paginated in memory.
            items = [
                _summarize(record)
                for _, record in list_records(observe_dir, model)
                if record_matches(record, search)
            ]
            total = len(items)
            page = items[offset : offset + limit]
        else:
            # No search: only the requested page is read from disk.
            records, total = list_records_page(observe_dir, model, offset, limit)
            page = [_summarize(record) for _, record in records]
        return JSONResponse({"items": page, "total": total})

    @app.get("/api/records/{record_id}")
    async def get_record(record_id: str) -> JSONResponse:
        for path, record in list_records(observe_dir, model):
            if record.id != record_id:
                continue
            try:
                full = load_record(path)
            except Exception:
                full = record
            return JSONResponse(_record_to_dict(full))
        return JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        filter_label = f" (model: {model})" if model else ""
        return _INDEX_PATH.read_text(encoding="utf-8").replace(
            "{{FILTER_LABEL}}", filter_label
        )

    return app
