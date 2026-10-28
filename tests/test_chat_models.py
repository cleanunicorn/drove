"""The picker trusts server metadata while retaining legacy endpoints."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from drove.chat_models import (
    CatalogError,
    ChatModel,
    fetch_models,
    fetch_models_async,
    model_catalog,
    parse_models,
)
from drove.config import Config
from tests.test_gguf import _kv_string, _kv_uint32, write_gguf


@pytest.mark.parametrize("extension", ["safetensors", "bin", "pt"])
@pytest.mark.parametrize("name", ["raw", "org/raw"])
def test_catalog_omits_unservable_weights(tmp_path: Path, extension: str, name: str) -> None:
    config = Config(models_dir=tmp_path)
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / f"model.{extension}").write_bytes(b"weights")
    (tmp_path / "ready.gguf").write_bytes(b"gguf")

    catalog = model_catalog(config, {"ready"})
    assert [row["id"] for row in catalog] == ["ready"]

    # A GGUF placed next to the raw weights (supplied by the user) is servable.
    (folder / "model.gguf").write_bytes(b"gguf")
    served = parse_models({"data": model_catalog(config, set())})
    assert {model.id for model in served} == {"ready", name}


def test_catalog_omits_interrupted_downloads(tmp_path: Path) -> None:
    config = Config(models_dir=tmp_path)
    cache = tmp_path / "org" / "partial" / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "abc.incomplete").write_bytes(b"partial")
    shards = tmp_path / "split"
    shards.mkdir()
    (shards / "split-00001-of-00002.gguf").write_bytes(b"gguf")
    (tmp_path / "ready.gguf").write_bytes(b"gguf")

    assert [row["id"] for row in model_catalog(config, set())] == ["ready"]


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


@pytest.mark.parametrize("value", [True, 0, -5, "4096", 4096.0, None])
def test_invalid_context_length_is_unknown(value: object) -> None:
    [model] = parse_models(
        {"data": [{"id": "a", "drove": {"chat": True, "context_length": value}}]}
    )
    assert model.context_length is None


def test_context_length_round_trips_from_the_catalog(tmp_path: Path) -> None:
    config = Config(models_dir=tmp_path)
    write_gguf(
        tmp_path / "trained.gguf",
        [_kv_string("general.architecture", "qwen3"), _kv_uint32("qwen3.context_length", 32768)],
    )
    write_gguf(tmp_path / "sized.gguf", [_kv_string("general.architecture", "qwen3")])
    (tmp_path / "sized.toml").write_text("ctx_size = 8192\n")
    write_gguf(tmp_path / "unknown.gguf", [_kv_string("general.architecture", "qwen3")])

    catalog = model_catalog(config, set())

    assert {row["id"]: row["drove"]["context_length"] for row in catalog} == {  # type: ignore[index]
        "sized": 8192,
        "trained": 32768,
        "unknown": None,
    }
    assert {model.id: model.context_length for model in parse_models({"data": catalog})} == {
        "sized": 8192,
        "trained": 32768,
        "unknown": None,
    }


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
