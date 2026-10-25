"""The picker trusts server metadata while retaining legacy endpoints."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from drove.chat_models import (
    CatalogError,
    ChatModel,
    fetch_models,
    fetch_models_async,
    parse_models,
)


def test_parse_filters_speech_and_broken_metadata_without_guessing() -> None:
    assert parse_models(
        {
            "data": [
                {"id": "legacy"},
                {"id": "speech", "drove": {"chat": False}},
                {"id": "broken", "drove": {"chat": None}},
                {"id": "bad", "drove": []},
                {
                    "id": "valid",
                    "drove": {
                        "chat": True,
                        "size_bytes": 12,
                        "loaded": True,
                        "capabilities": ["text", "vision"],
                    },
                },
                {"id": 3},
                None,
                [],
                {"id": ""},
                {"id": "legacy"},
            ]
        }
    ) == [ChatModel("legacy"), ChatModel("valid", 12, True, ("text", "vision"))]


@pytest.mark.parametrize("payload", [None, [], {}, {"data": None}, {"data": [None]}])
def test_bad_catalog_has_normal_error(payload: object) -> None:
    with pytest.raises(CatalogError):
        parse_models(payload)


def test_optional_fields_are_defensive() -> None:
    assert parse_models(
        {
            "data": [
                {
                    "id": "a",
                    "drove": {
                        "chat": True,
                        "size_bytes": True,
                        "loaded": "yes",
                        "capabilities": [2],
                    },
                }
            ]
        }
    ) == [ChatModel("a")]


def test_sync_fetch_preserves_auth_and_url() -> None:
    response = httpx.Response(
        200,
        json={"data": [{"id": "a"}]},
        request=httpx.Request("GET", "https://example.test/v1/models"),
    )
    with patch("drove.chat_models.httpx.get", return_value=response) as get:
        assert fetch_models("https://example.test", "secret") == [ChatModel("a")]
    get.assert_called_once_with(
        "https://example.test/v1/models", headers={"Authorization": "Bearer secret"}, timeout=5.0
    )


async def test_async_fetch_preserves_auth_and_url() -> None:
    response = httpx.Response(
        200,
        json={"data": [{"id": "a"}]},
        request=httpx.Request("GET", "https://example.test/v1/models"),
    )
    with patch("drove.chat_models.httpx.AsyncClient") as client:
        get = AsyncMock(return_value=response)
        client.return_value.__aenter__.return_value.get = get
        assert await fetch_models_async("https://example.test", "secret") == [ChatModel("a")]
    get.assert_awaited_once_with(
        "https://example.test/v1/models", headers={"Authorization": "Bearer secret"}
    )


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, text="not json")])
def test_bad_http_response_has_normal_error(response: httpx.Response) -> None:
    response.request = httpx.Request("GET", "https://example.test/v1/models")
    with patch("drove.chat_models.httpx.get", return_value=response), pytest.raises(CatalogError):
        fetch_models("https://example.test", None)
