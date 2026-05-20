"""Tests for the observe (request/response logging) module."""

from __future__ import annotations

import json
from pathlib import Path

from drove.observe import (
    ObserveContext,
    ObserveRecord,
    list_records,
    list_records_page,
    load_record,
    save_record,
)


def _make_record(
    model: str = "testmodel",
    endpoint: str = "v1/chat/completions",
    record_id: str = "20260408-120000-abcd1234",
) -> ObserveRecord:
    return ObserveRecord(
        id=record_id,
        timestamp="2026-04-08T12:00:00",
        model=model,
        endpoint=endpoint,
        method="POST",
        request_headers={"content-type": "application/json"},
        request_body='{"model": "testmodel", "messages": []}',
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body='{"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}',
        tokens_prompt=10,
        tokens_completion=5,
        tokens_per_second=25.0,
        ttft_seconds=0.1,
        duration_seconds=0.5,
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    record = _make_record()
    path = save_record(tmp_path, record)

    assert path.exists()
    assert path.suffix == ".json"

    loaded = load_record(path)
    assert loaded.id == record.id
    assert loaded.model == record.model
    assert loaded.endpoint == record.endpoint
    assert loaded.method == record.method
    assert loaded.response_status == record.response_status
    assert loaded.tokens_prompt == 10
    assert loaded.tokens_completion == 5
    assert loaded.tokens_per_second == 25.0
    assert loaded.ttft_seconds == 0.1
    assert loaded.duration_seconds == 0.5
    assert loaded.request_body == record.request_body
    assert loaded.response_body == record.response_body


def test_save_creates_model_subdirectory(tmp_path: Path) -> None:
    record = _make_record(model="mymodel")
    path = save_record(tmp_path, record)

    assert path.parent.name == "mymodel"
    assert path.parent.parent == tmp_path


def test_save_unknown_model_uses_underscore_dir(tmp_path: Path) -> None:
    record = _make_record()
    record.model = None
    path = save_record(tmp_path, record)

    assert path.parent.name == "_unknown"


def test_list_records_returns_newest_first(tmp_path: Path) -> None:
    r1 = _make_record(record_id="20260408-100000-aaaa0001")
    r1.timestamp = "2026-04-08T10:00:00"
    save_record(tmp_path, r1)

    r2 = _make_record(record_id="20260408-120000-aaaa0002")
    r2.timestamp = "2026-04-08T12:00:00"
    save_record(tmp_path, r2)

    r3 = _make_record(record_id="20260408-110000-aaaa0003")
    r3.timestamp = "2026-04-08T11:00:00"
    save_record(tmp_path, r3)

    records = list_records(tmp_path)
    assert len(records) == 3
    assert records[0][1].id == r2.id  # newest
    assert records[1][1].id == r3.id
    assert records[2][1].id == r1.id  # oldest


def test_list_records_filters_by_model(tmp_path: Path) -> None:
    r1 = _make_record(model="modelA", record_id="20260408-100000-aaaa0001")
    save_record(tmp_path, r1)

    r2 = _make_record(model="modelB", record_id="20260408-100000-aaaa0002")
    save_record(tmp_path, r2)

    records_a = list_records(tmp_path, model="modelA")
    assert len(records_a) == 1
    assert records_a[0][1].model == "modelA"

    records_all = list_records(tmp_path)
    assert len(records_all) == 2


def test_list_records_page_paginates(tmp_path: Path) -> None:
    for i in range(1, 6):
        save_record(tmp_path, _make_record(record_id=f"20260408-10000{i}-aaaa000{i}"))

    page, total = list_records_page(tmp_path, offset=0, limit=2)
    assert total == 5
    assert [p[1].id for p in page] == [
        "20260408-100005-aaaa0005",
        "20260408-100004-aaaa0004",
    ]

    last, total = list_records_page(tmp_path, offset=4, limit=2)
    assert total == 5
    assert len(last) == 1
    assert last[0][1].id == "20260408-100001-aaaa0001"


def test_list_records_page_offset_past_end(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(record_id="20260408-100001-aaaa0001"))

    page, total = list_records_page(tmp_path, offset=10, limit=2)
    assert total == 1
    assert page == []


def test_list_records_page_nonexistent_dir(tmp_path: Path) -> None:
    page, total = list_records_page(tmp_path / "nonexistent")
    assert page == []
    assert total == 0


def test_list_records_finds_namespaced_models(tmp_path: Path) -> None:
    """Records saved under 'org/repo:quant' should be discoverable.

    After the naming convention switched to ``repo/name:quant``, model names
    contain ``/`` which creates a nested directory. list_records must walk
    two levels deep to find them alongside legacy flat records.
    """
    # Namespaced (new-style)
    r1 = _make_record(
        model="unsloth/Qwen3-8B-GGUF:Q8_0",
        record_id="20260408-100000-nnnn0001",
    )
    save_record(tmp_path, r1)

    # Flat (legacy)
    r2 = _make_record(
        model="legacy-model",
        record_id="20260408-100000-llll0001",
    )
    save_record(tmp_path, r2)

    records = list_records(tmp_path)
    assert len(records) == 2
    models = {rec.model for _, rec in records}
    assert "unsloth/Qwen3-8B-GGUF:Q8_0" in models
    assert "legacy-model" in models

    # Filtering by namespaced name should also work
    filtered = list_records(tmp_path, model="unsloth/Qwen3-8B-GGUF:Q8_0")
    assert len(filtered) == 1
    assert filtered[0][1].model == "unsloth/Qwen3-8B-GGUF:Q8_0"


def test_list_records_empty_dir(tmp_path: Path) -> None:
    assert list_records(tmp_path) == []


def test_list_records_nonexistent_dir(tmp_path: Path) -> None:
    assert list_records(tmp_path / "nonexistent") == []


def test_saved_json_is_valid(tmp_path: Path) -> None:
    record = _make_record()
    path = save_record(tmp_path, record)

    data = json.loads(path.read_text())
    assert data["id"] == record.id
    assert data["model"] == "testmodel"
    assert data["tokens_prompt"] == 10
    assert data["response_status"] == 200


def test_sse_tool_calls_assembled() -> None:
    """SSE streaming responses with tool calls should be fully assembled."""
    tc1 = '{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":""}}'
    tc2 = '{"index":0,"function":{"arguments":"{\\"city\\":"}}'
    tc3 = '{"index":0,"function":{"arguments":" \\"NYC\\"}"}}'
    sse_body = (
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{tc1}]}}}}]}}\n'
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{tc2}]}}}}]}}\n'
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{tc3}]}}}}]}}\n'
        "data: [DONE]\n"
    )
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={},
        request_body=b"{}",
        response_body=sse_body.encode(),
    )
    record = ctx.to_record()

    parsed = json.loads(record.response_body)
    assert "tool_calls" in parsed
    assert len(parsed["tool_calls"]) == 1
    tc = parsed["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"] == "get_weather"
    assert tc["arguments"] == {"city": "NYC"}


def test_observe_context_to_record() -> None:
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={"content-type": "application/json"},
        request_body=b'{"model": "testmodel"}',
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body=b'{"choices": []}',
        ttft_seconds=0.05,
        duration_seconds=1.2,
        tokens_prompt=15,
        tokens_completion=8,
        tokens_per_second=30.0,
    )

    record = ctx.to_record()
    assert record.model == "testmodel"
    assert record.endpoint == "v1/chat/completions"
    assert record.request_body == '{"model": "testmodel"}'
    # response_body is the assembled (pretty-printed) version
    assert '"choices": []' in record.response_body
    # response_body_raw preserves the original
    assert record.response_body_raw == '{"choices": []}'
    assert record.tokens_prompt == 15
    assert record.tokens_per_second == 30.0
    assert record.id  # has an id
    assert len(record.id) > 8  # non-trivial id


def test_sse_response_assembled() -> None:
    """SSE streaming responses should be assembled into readable content."""
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n'
        "data: [DONE]\n"
    )
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={},
        request_body=b"{}",
        response_body=sse_body.encode(),
    )
    record = ctx.to_record()

    # The assembled body should have the concatenated content
    parsed = json.loads(record.response_body)
    assert parsed["content"] == "Hello world"
    assert parsed["usage"]["prompt_tokens"] == 5

    # The raw body preserves the SSE lines
    assert "data: " in record.response_body_raw
    assert record.response_body_raw != record.response_body


def test_observe_context_to_record_none_model() -> None:
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model=None,
        endpoint="health",
        method="GET",
        request_headers={},
        request_body=b"",
    )
    record = ctx.to_record()
    assert record.model is None
