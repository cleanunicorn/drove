"""FastAPI reverse proxy to llama-server."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from vllama.config import Config, load_config
from vllama.server_manager import ServerManager

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
            tasks.append(asyncio.create_task(
                _config_watcher(_app, config_path),
                name="config-watcher",
            ))
            _setup_sighup(_app, config_path)

        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await manager.stop()
            await client.aclose()

    app = FastAPI(title="vllama", lifespan=lifespan)
    app.state.manager = manager
    app.state.config = config

    client = httpx.AsyncClient(base_url=manager.base_url, timeout=300.0)
    app.state.client = client  # kept here so lifespan can close it

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> StreamingResponse:
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

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        body = await request.body()

        try:
            upstream = client.build_request(
                method=request.method,
                url=f"/{path}",
                params=request.query_params,
                headers=headers,
                content=body,
            )
            resp = await client.send(upstream, stream=True)
        except httpx.TransportError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

        response_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }

        return StreamingResponse(
            content=resp.aiter_raw(),
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
            background=None,
        )

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
    merged = new_config.model_copy(update={
        f: old_dump[f] for f in _RESTART_REQUIRED
    })

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
    except (OSError, NotImplementedError):
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
