"""Synthetic API fixture serving the installed observe frontend, never user logs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

from drove.observe_web import create_observe_app

app = create_observe_app(Path(sys.argv[1]))


@app.middleware("http")
async def isolate_network(request: Request, call_next):
    response = await call_next(request)
    if len(sys.argv) < 4 or sys.argv[3] != "--baseline":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline' 'unsafe-eval'; object-src 'none'; frame-src 'none'"
        )
    return response


app.router.routes = [
    route for route in app.router.routes if not getattr(route, "path", "").startswith("/api/")
]
fixture: dict[str, Any] = {}


@app.post("/__fixture")
async def set_fixture(request: Request) -> dict[str, bool]:
    global fixture
    fixture = await request.json()
    return {"ok": True}


@app.get("/api/records")
async def records() -> dict[str, Any]:
    return {"items": [fixture] if fixture else [], "total": int(bool(fixture))}


@app.get("/api/records/{record_id:path}")
async def record(record_id: str) -> JSONResponse:
    return JSONResponse(fixture)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
