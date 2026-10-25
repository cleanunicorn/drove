"""Run real-browser state/accessibility checks against an isolated synthetic observe API."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from drove.observe import ObserveRecord, save_record
from drove.observe_web import create_observe_app


def sample(index: int, model: str = "fixture-model") -> dict[str, Any]:
    from datetime import datetime

    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "# Heading\n\n- first\n- second\n\n```python\n"
                        "print('hello')\n```\n\n[Safe link](https://example.com)"
                    ),
                }
            ],
            "nested": {"boolean": True, "nothing": None, "number": 42},
            "long": "long-value-" * 60,
        }
    )
    return {
        "id": f"record-{index}",
        "timestamp": datetime.now().isoformat(timespec="seconds")
        if index == 0
        else "2025-01-02T12:34:56",
        "model": model,
        "endpoint": "v1/chat/completions",
        "method": "POST",
        "response_status": 200 if index % 2 == 0 else 503,
        "tokens_prompt": 10,
        "tokens_completion": 5,
        "tokens_per_second": 20.0,
        "ttft_seconds": 0.1,
        "duration_seconds": 0.5,
        "request_body": body,
        "response_body": body,
        "request_headers": {"content-type": "application/json"},
        "response_headers": {"x-test": "header value"},
        "response_body_raw": "raw response",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(tempfile.mkdtemp(prefix="observe-states-"))
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Check original main assets for regression evidence"
    )
    args = parser.parse_args()
    if shutil.which("betterwright") is None:
        parser.error("betterwright must be installed to run browser checks")
    output = (args.output or Path(tempfile.mkdtemp(prefix="observe-states-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="drove-observe-states-") as directory:
        data_dir = Path(directory)
        real = sample(999)
        real["id"] = "20260408-120000-abcd1234"
        save_record(data_dir, ObserveRecord(**real))
        if args.baseline:
            import drove.observe_web as web

            assets = data_dir / "assets"
            assets.mkdir()
            for name in ("app.js", "styles.css", "index.html"):
                content = subprocess.check_output(
                    ["git", "show", f"5a0b9ac:src/drove/observe_web_static/{name}"]
                )
                (assets / name).write_bytes(content)
            web._STATIC_DIR = assets
            web._INDEX_PATH = assets / "index.html"
        app = create_observe_app(data_dir)
        fixture: dict[str, Any] = {"rules": [], "seen": [], "total": 4213, "mode": "synthetic"}
        releases: dict[str, asyncio.Event] = {}

        @app.middleware("http")
        async def fixtures(request: Request, call_next: Any) -> Response:
            path = request.url.path
            if path == "/__test/config":
                values = await request.json()
                fixture.update(values)
                return JSONResponse({"ok": True})
            if path == "/__test/seen":
                return JSONResponse(fixture["seen"])
            if path == "/__test/release":
                values = await request.json()
                releases.setdefault(values["name"], asyncio.Event()).set()
                return JSONResponse({"ok": True})
            if not path.startswith("/api/records") or fixture["mode"] == "live":
                return await call_next(request)
            query = request.query_params.get("search", "")
            offset = int(request.query_params.get("offset", "0"))
            fixture["seen"].append({"path": path, "query": query, "offset": offset})
            rule = None
            for candidate in fixture["rules"]:
                if candidate["path"] != path:
                    continue
                if "query" in candidate and candidate["query"] != query:
                    continue
                if "offset" in candidate and candidate["offset"] != offset:
                    continue
                rule = candidate
                fixture["rules"].remove(candidate)
                break
            if rule:
                if name := rule.get("hold"):
                    await releases.setdefault(name, asyncio.Event()).wait()
                if rule.get("malformed"):
                    return Response("{invalid-json", media_type="application/json")
                if rule.get("disconnect"):
                    from starlette.responses import StreamingResponse

                    async def broken_body() -> Any:
                        yield b'{"items":'
                        raise RuntimeError("fixture disconnect")

                    return StreamingResponse(broken_body(), media_type="application/json")
                if "json" in rule or "status" in rule:
                    return JSONResponse(
                        rule.get("json", {"error": "fixture failure"}),
                        status_code=rule.get("status", 200),
                    )
            if path == "/api/records":
                count = 0 if query == "empty" else 37 if query == "short" else 130
                total = count if query else fixture["total"]
                limit = int(request.query_params.get("limit", "100"))
                records = [
                    sample(i, query or "fixture-model")
                    for i in range(offset, min(count, offset + limit))
                ]
                return JSONResponse({"items": records, "total": total})
            try:
                index = int(path.rsplit("-", 1)[-1])
            except ValueError:
                return JSONResponse({"error": "not found"}, status_code=404)
            record = sample(index)
            if index == 777:
                record.update(
                    model="org/" + "model-long-" * 50,
                    endpoint="long-endpoint/" * 50,
                    timestamp="invalid time",
                )
            return JSONResponse(record)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", lifespan="off", timeout_graceful_shutdown=1)
        )
        thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        thread.start()
        try:
            script = Path(__file__).with_name("observe_web_states.js").read_text()
            script = (
                f"const base = {json.dumps(address)};\n"
                f"const baseline = {json.dumps(args.baseline)};\n" + script
            )
            result = subprocess.run(
                ["betterwright", "run", "--session", "observe-web-states", "-"],
                input=script,
                capture_output=True,
                text=True,
            )
            (output / "browser-result.json").write_text(result.stdout)
            (output / "browser-stderr.log").write_text(result.stderr)
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(result.stdout[-2000:])
                print(result.stderr[-2000:])
                return 1
            if payload.get("ok"):
                for key, name in (
                    ("longMobileProof", "long-mobile-proof.png"),
                    ("mobileProof", "mobile-proof.png"),
                    ("desktopProof", "desktop-proof.png"),
                ):
                    proof = (payload.get("result") or {}).get(key)
                    if proof and proof.get("path"):
                        shutil.copy2(proof["path"], output / name)
            print(
                json.dumps(
                    {key: payload.get(key) for key in ("ok", "result", "error", "artifacts")},
                    indent=2,
                )
            )
            return 0 if result.returncode == 0 and payload.get("ok") else 1
        finally:
            subprocess.run(
                ["betterwright", "close", "--session", "observe-web-states"], capture_output=True
            )
            server.should_exit = True
            thread.join(timeout=10)
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
