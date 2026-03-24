"""FastAPI reverse proxy to llama-server."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from vllama.config import Config
from vllama.server_manager import ServerManager

logger = logging.getLogger(__name__)

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


def create_app(config: Config) -> FastAPI:
    manager = ServerManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.stop()

    app = FastAPI(title="vllama", lifespan=lifespan)
    # Attach for access from tests / CLI
    app.state.manager = manager
    app.state.config = config

    client = httpx.AsyncClient(base_url=manager.base_url, timeout=300.0)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> StreamingResponse:
        # Extract model name from request body (OpenAI format) or use current
        model_name = await _extract_model(request)

        if model_name is None:
            # No model in request — use whichever is currently loaded, or error
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

        # Forward request
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
