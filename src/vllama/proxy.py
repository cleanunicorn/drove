"""FastAPI reverse proxy to llama-server."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllama.config import Config, load_config
from vllama.server_manager import ServerManager
from vllama.stats import ProxyStats

logger = logging.getLogger(__name__)

# Poll interval for config file change detection
_CONFIG_POLL_SECONDS = 5

# Settings that cannot be changed without restarting the proxy process
_RESTART_REQUIRED = {"listen_host", "listen_port"}

# Header names to strip when forwarding (hop-by-hop)
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    ]
)


def create_app(config: Config, config_path: Path | None = None) -> FastAPI:
    manager = ServerManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        tasks: list[asyncio.Task[None]] = []

        if config_path is not None:
            tasks.append(
                asyncio.create_task(
                    _config_watcher(_app, config_path),
                    name="config-watcher",
                )
            )
            _setup_sighup(_app, config_path)

        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await manager.stop()
            await client.aclose()

    stats = ProxyStats()

    app = FastAPI(title="vllama", lifespan=lifespan)
    app.state.manager = manager
    app.state.config = config
    app.state.stats = stats

    client = httpx.AsyncClient(timeout=300.0)
    app.state.client = client  # kept here so lifespan can close it

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "model": manager.current_model,
                "server_running": manager.is_running,
            }
        )

    @app.get("/status")
    async def status() -> JSONResponse:
        now = time.time()

        model_data: dict[str, object] = {"loaded": manager.is_running}
        if manager.is_running and manager.current_model:
            model_data["name"] = manager.current_model
            if manager.model_loaded_at:
                model_data["loaded_seconds"] = round(now - manager.model_loaded_at, 1)
            model_data["idle_seconds"] = round(manager.idle_seconds, 1)
            model_data["idle_timeout_seconds"] = config.idle_timeout_seconds

        return JSONResponse(
            {
                "server": {
                    "uptime_seconds": round(now - stats.started_at, 1),
                    "listen": f"{config.listen_host}:{config.listen_port}",
                },
                "model": model_data,
                "process": manager.get_process_stats(),
                "requests": {
                    "total": stats.request_count,
                    "active": stats.active_requests,
                    "errors": stats.error_count,
                },
                "tokens": {
                    "prompt": stats.tokens_prompt,
                    "completion": stats.tokens_completion,
                    "total": stats.tokens_prompt + stats.tokens_completion,
                    "speed": {
                        "last_tok_per_sec": (
                            round(stats.last_tokens_per_second, 1)
                            if stats.last_tokens_per_second is not None
                            else None
                        ),
                        "avg_tok_per_sec": (
                            round(stats.avg_tokens_per_second, 1)
                            if stats.avg_tokens_per_second is not None
                            else None
                        ),
                    },
                    "ttft": {
                        "last_seconds": (
                            round(stats.last_ttft, 3) if stats.last_ttft is not None else None
                        ),
                        "avg_seconds": (
                            round(stats.avg_ttft, 3) if stats.avg_ttft is not None else None
                        ),
                    },
                },
            }
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> StreamingResponse:
        stats.request_started()
        try:
            model_name = await _extract_model(request)

            if model_name is None:
                if not manager.is_running:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "No model is loaded. Specify a 'model' field in your request "
                            "or load a model first."
                        ),
                    )
            else:
                try:
                    await manager.ensure_running(model_name)
                except FileNotFoundError as e:
                    raise HTTPException(status_code=404, detail=str(e)) from e
                except TimeoutError as e:
                    raise HTTPException(status_code=503, detail=str(e)) from e
                except RuntimeError as e:
                    raise HTTPException(status_code=503, detail=str(e)) from e

            manager.record_request()

            headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
            body = await request.body()

            try:
                upstream = client.build_request(
                    method=request.method,
                    url=f"{manager.base_url}/{path}",
                    params=request.query_params,
                    headers=headers,
                    content=body,
                )
                resp = await client.send(upstream, stream=True)
            except httpx.TransportError as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

            response_headers = {
                k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP
            }

            return StreamingResponse(
                content=_counting_stream(resp.aiter_raw(), stats),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
                background=None,
            )
        except Exception:
            stats.request_finished()
            stats.request_error()
            raise

    return app


async def _config_watcher(app: FastAPI, config_path: Path) -> None:
    """Background task: reload config when the file's mtime changes."""
    mtime = _mtime(config_path)
    while True:
        await asyncio.sleep(_CONFIG_POLL_SECONDS)
        new_mtime = _mtime(config_path)
        if new_mtime != mtime:
            mtime = new_mtime
            _reload_config(app, config_path)


def _reload_config(app: FastAPI, config_path: Path) -> None:
    """Load config from disk and apply hot-reloadable fields to the running app."""
    try:
        new_config = load_config(config_path)
    except Exception as e:
        logger.error("Failed to reload config: %s", e)
        return

    old_config: Config = app.state.config
    changed: list[str] = []
    skipped: list[str] = []

    new_dump = new_config.model_dump()
    old_dump = old_config.model_dump()

    for field, new_val in new_dump.items():
        if new_val == old_dump.get(field):
            continue
        if field in _RESTART_REQUIRED:
            skipped.append(f"{field} (restart required)")
        else:
            changed.append(f"{field}: {old_dump.get(field)!r} → {new_val!r}")

    if not changed and not skipped:
        return

    # Build updated config preserving non-reloadable fields from the old config
    merged = new_config.model_copy(update={f: old_dump[f] for f in _RESTART_REQUIRED})

    app.state.config = merged
    app.state.manager._config = merged

    if changed:
        logger.info("Config reloaded — changed: %s", "; ".join(changed))
    if skipped:
        logger.warning("Config changes ignored (restart required): %s", "; ".join(skipped))


def _setup_sighup(app: FastAPI, config_path: Path) -> None:
    """Reload config on SIGHUP (kill -HUP <pid>)."""
    loop = asyncio.get_event_loop()

    def _handler() -> None:
        logger.info("SIGHUP received — reloading config")
        _reload_config(app, config_path)

    try:
        loop.add_signal_handler(signal.SIGHUP, _handler)
    except OSError, NotImplementedError:
        pass  # Windows or environments without SIGHUP


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


async def _extract_model(request: Request) -> str | None:
    """Try to read the 'model' field from a JSON body without consuming the stream."""
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        body = await request.json()
        return body.get("model") if isinstance(body, dict) else None
    except Exception:
        return None


async def _counting_stream(
    raw_iter: AsyncIterator[bytes], stats: ProxyStats
) -> AsyncIterator[bytes]:
    """Wrap a response stream, collecting bytes to extract token usage afterwards."""
    try:
        chunks: list[bytes] = []
        t_start = time.monotonic()
        t_first_chunk: float | None = None
        async for chunk in raw_iter:
            if t_first_chunk is None:
                t_first_chunk = time.monotonic()
            chunks.append(chunk)
            yield chunk
        t_end = time.monotonic()
        if t_first_chunk is not None:
            stats.record_ttft(t_first_chunk - t_start)
        _record_usage(b"".join(chunks), stats, t_end - t_start)
    except Exception:
        stats.request_error()
        raise
    finally:
        stats.request_finished()


def _extract_tokens_from_obj(data: dict[str, object], out: dict[str, int | float | None]) -> None:
    """Extract token counts and speed from a response object (usage or timings)."""
    # OpenAI-style usage
    usage = data.get("usage")
    if isinstance(usage, dict):
        out["prompt"] = usage.get("prompt_tokens", 0) or out["prompt"]
        out["completion"] = usage.get("completion_tokens", 0) or out["completion"]

    # llama.cpp timings (always present in last streaming chunk)
    timings = data.get("timings")
    if isinstance(timings, dict):
        if not out["prompt"]:
            out["prompt"] = timings.get("prompt_n", 0)
        if not out["completion"]:
            out["completion"] = timings.get("predicted_n", 0)
        if timings.get("predicted_per_second"):
            out["speed"] = timings["predicted_per_second"]


def _record_usage(body: bytes, stats: ProxyStats, duration: float = 0.0) -> None:
    """Best-effort extraction of token usage from a response body."""
    text = body.decode("utf-8", errors="replace")
    tokens_out: dict[str, int | float | None] = {"prompt": 0, "completion": 0, "speed": None}

    # Non-streaming: plain JSON with a top-level "usage" field
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            _extract_tokens_from_obj(data, tokens_out)
    except json.JSONDecodeError, ValueError:
        pass

    # Streaming SSE: scan backwards for the last data: line containing usage/timings
    if not tokens_out["completion"]:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if not stripped.startswith("data: ") or stripped == "data: [DONE]":
                continue
            try:
                data = json.loads(stripped[6:])
                if not isinstance(data, dict):
                    continue
                _extract_tokens_from_obj(data, tokens_out)
                if tokens_out["completion"]:
                    break
            except json.JSONDecodeError, ValueError:
                continue

    prompt_tokens = tokens_out["prompt"]
    completion_tokens = tokens_out["completion"]
    tok_per_sec = tokens_out["speed"]

    if prompt_tokens or completion_tokens:
        stats.add_tokens(prompt_tokens, completion_tokens)

    # Record speed: prefer server-reported timing, fall back to our own measurement
    if tok_per_sec and tok_per_sec > 0:
        stats.record_completion_speed(completion_tokens, completion_tokens / tok_per_sec)
    elif completion_tokens > 0 and duration > 0:
        stats.record_completion_speed(completion_tokens, duration)
