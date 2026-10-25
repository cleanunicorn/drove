"""Chat catalog metadata and compatible OpenAI model-list clients."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from drove.config import Config
from drove.integrations import describe_model
from drove.model_store import ModelStore


@dataclass(frozen=True)
class ChatModel:
    id: str
    size_bytes: int | None = None
    loaded: bool | None = None
    capabilities: tuple[str, ...] | None = None


class CatalogError(ValueError):
    """The endpoint could not provide a usable chat catalog."""


def model_catalog(config: Config, loaded: set[str]) -> list[dict[str, object]]:
    """Describe every model without loading a backend; call off the event loop."""
    models: list[dict[str, object]] = []
    store = ModelStore(config.models_dir)
    for entry in store.list():
        metadata: dict[str, object] = {
            "chat": None,
            "loaded": entry.name in loaded,
            "size_bytes": entry.total_bytes,
            "capabilities": None,
        }
        try:
            primary = store.resolve(entry.name)
            spec = describe_model(config, entry.name, primary)
        except OSError, ValueError:
            # Unservable weights and broken sidecars stay in the inventory,
            # but must not be advertised as suitable for chat.
            pass
        else:
            metadata["chat"] = spec is not None
            if spec is not None:
                capabilities = ["text"]
                if spec.vision:
                    capabilities.append("vision")
                if spec.tool_call:
                    capabilities.append("tools")
                if spec.reasoning:
                    capabilities.append("reasoning")
                metadata["capabilities"] = capabilities
            else:
                metadata["capabilities"] = []
        models.append(
            {
                "id": entry.name,
                "object": "model",
                "created": 0,
                "owned_by": "local",
                "drove": metadata,
            }
        )
    return models


def parse_models(payload: object) -> list[ChatModel]:
    """Keep legacy endpoints usable without guessing unknown capabilities."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise CatalogError("Invalid response from /v1/models.")
    models: dict[str, ChatModel] = {}
    for row in payload["data"]:
        if not isinstance(row, dict):
            continue
        name = row.get("id")
        if not isinstance(name, str) or not name.strip():
            continue
        if "drove" not in row:
            models.setdefault(name, ChatModel(name))
            continue
        metadata = row["drove"]
        if not isinstance(metadata, dict) or metadata.get("chat") is not True:
            continue
        size = metadata.get("size_bytes")
        loaded = metadata.get("loaded")
        capabilities = metadata.get("capabilities")
        models[name] = ChatModel(
            name,
            size if type(size) is int and size >= 0 else None,
            loaded if isinstance(loaded, bool) else None,
            tuple(capabilities)
            if isinstance(capabilities, list)
            and all(isinstance(value, str) for value in capabilities)
            else None,
        )
    if not models:
        raise CatalogError("No chat models available at this endpoint.")
    return sorted(models.values(), key=lambda model: (model.id.casefold(), model.id))


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _read_response(response: httpx.Response) -> list[ChatModel]:
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise CatalogError("Invalid response from /v1/models.") from exc
    return parse_models(payload)


def fetch_models(base_url: str, api_key: str | None) -> list[ChatModel]:
    try:
        return _read_response(
            httpx.get(f"{base_url}/v1/models", headers=_headers(api_key), timeout=5.0)
        )
    except httpx.HTTPError as exc:
        raise CatalogError(
            "Could not list models. Check the endpoint, server and API key."
        ) from exc


async def fetch_models_async(base_url: str, api_key: str | None) -> list[ChatModel]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/v1/models", headers=_headers(api_key))
            return _read_response(response)
    except httpx.HTTPError as exc:
        raise CatalogError(
            "Could not list models. Check the endpoint, server and API key."
        ) from exc
