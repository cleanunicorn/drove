"""Tests for the observe (request/response logging) module."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from drove.observe import (
    MAX_BODY_BYTES,
    MAX_RECORD_BYTES,
    REDACTED,
    ObserveContext,
    ObserveRecord,
    find_record_path,
    list_records,
    list_records_page,
    load_record,
    prune_records,
    save_record,
    search_records_page,
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


def test_search_records_page_paginates_matches(tmp_path: Path) -> None:
    for i in range(1, 4):
        record = _make_record(model="alpha", record_id=f"20260408-10000{i}-aaaa000{i}")
        save_record(tmp_path, record)
    save_record(tmp_path, _make_record(model="beta", record_id="20260408-100009-bbbb0001"))

    page, total = search_records_page(tmp_path, "alpha", offset=0, limit=2)
    assert total == 3
    assert [p[1].id for p in page] == [
        "20260408-100003-aaaa0003",
        "20260408-100002-aaaa0002",
    ]

    last, total = search_records_page(tmp_path, "alpha", offset=2, limit=2)
    assert total == 3
    assert len(last) == 1
    assert last[0][1].id == "20260408-100001-aaaa0001"


def test_search_records_page_no_matches(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(record_id="20260408-100001-aaaa0001"))

    page, total = search_records_page(tmp_path, "nomatch")
    assert page == []
    assert total == 0


def test_search_records_page_skips_corrupt_files(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(model="alpha", record_id="20260408-100001-aaaa0001"))
    (tmp_path / "alpha" / "20260408-100002-corrupt0.json").write_text("not json")

    page, total = search_records_page(tmp_path, "alpha")
    assert total == 1
    assert len(page) == 1


@pytest.mark.parametrize("reader", ["list", "page", "search"])
def test_readers_skip_a_record_pruned_after_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    """Retention can unlink a record between enumeration and open: skip it, keep the rest."""
    import drove.observe as observe

    for n in (1, 2, 3):
        save_record(tmp_path, _make_record(record_id=f"20260408-10000{n}-aaaa000{n}"))
    real_load = observe.load_record

    def load_after_prune(path: Path) -> ObserveRecord:
        if path.stem.endswith("0002"):
            path.unlink()  # the hourly prune wins the race
        return real_load(path)

    monkeypatch.setattr(observe, "load_record", load_after_prune)
    if reader == "list":
        ids = [r.id for _, r in list_records(tmp_path)]
    elif reader == "page":
        page, total = list_records_page(tmp_path)
        ids = [r.id for _, r in page]
        assert total == 3
    else:
        page, _ = search_records_page(tmp_path, "testmodel")
        ids = [r.id for _, r in page]
    assert ids == ["20260408-100003-aaaa0003", "20260408-100001-aaaa0001"]


def test_find_record_path_locates_record(tmp_path: Path) -> None:
    record = _make_record(record_id="20260408-100001-aaaa0001")
    saved = save_record(tmp_path, record)

    assert find_record_path(tmp_path, record.id) == saved
    assert find_record_path(tmp_path, record.id, model="testmodel") == saved


def test_find_record_path_namespaced_model(tmp_path: Path) -> None:
    record = _make_record(
        model="unsloth/Qwen3-8B-GGUF:Q8_0",
        record_id="20260408-100000-nnnn0001",
    )
    saved = save_record(tmp_path, record)

    assert find_record_path(tmp_path, record.id) == saved


def test_find_record_path_missing(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(record_id="20260408-100001-aaaa0001"))

    assert find_record_path(tmp_path, "nonexistent") is None
    assert find_record_path(tmp_path / "nonexistent", "whatever") is None


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


# ── model name containment (#110) ────────────────────────────────────────────


@pytest.mark.parametrize("model", ["../escape", "../../x", "/abs/dir", "a/../../b", "a//b", ""])
def test_save_record_rejects_unsafe_model_and_writes_nothing(tmp_path: Path, model: str) -> None:
    observe_dir = tmp_path / "observe"
    observe_dir.mkdir()
    with pytest.raises(ValueError):
        save_record(observe_dir, _make_record(model=model))
    assert sorted(p for p in tmp_path.rglob("*")) == [observe_dir]


def test_save_record_refuses_outward_namespace_symlink(tmp_path: Path) -> None:
    observe_dir = tmp_path / "observe"
    observe_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (observe_dir / "org").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        save_record(observe_dir, _make_record(model="org/repo:Q4_K_M"))
    assert list(outside.iterdir()) == []


def test_save_record_keeps_namespaced_quant_and_unknown_layout(tmp_path: Path) -> None:
    named = save_record(tmp_path, _make_record(model="org/repo:Q4_K_M"))
    assert named == tmp_path / "org" / "repo:Q4_K_M" / f"{named.stem}.json"
    unknown = save_record(tmp_path, _make_record(model=None, record_id="20260408-120001-x"))
    assert unknown.parent == tmp_path / "_unknown"


@pytest.mark.parametrize("model", ["../outside", "/abs", "org"])
def test_filtered_reads_never_leave_observe_dir(tmp_path: Path, model: str) -> None:
    observe_dir = tmp_path / "observe"
    observe_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    save_record(outside, _make_record(model="victim"))
    (observe_dir / "org").symlink_to(outside / "victim", target_is_directory=True)
    assert list_records(observe_dir, model=model) == []
    assert find_record_path(observe_dir, "20260408-120000-abcd1234", model=model) is None


def test_save_record_refuses_outward_symlink_at_record_file(tmp_path: Path) -> None:
    observe_dir = tmp_path / "observe"
    record = _make_record(model="real")
    (observe_dir / "real").mkdir(parents=True)
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("keep")
    (observe_dir / "real" / f"{record.id}.json").symlink_to(sentinel)
    with pytest.raises(ValueError, match="outside"):
        save_record(observe_dir, record)
    assert sentinel.read_text() == "keep"


def test_unfiltered_scans_skip_outward_directories_and_files(tmp_path: Path) -> None:
    observe_dir = tmp_path / "observe"
    observe_dir.mkdir()
    outside = tmp_path / "outside"
    save_record(outside, _make_record(model="victim", record_id="20260408-120000-out00001"))
    (observe_dir / "linked").symlink_to(outside / "victim", target_is_directory=True)
    (observe_dir / "org").mkdir()
    (observe_dir / "org" / "ns").symlink_to(outside / "victim", target_is_directory=True)
    kept = save_record(observe_dir, _make_record(model="local"))
    (kept.parent / "20260408-120000-out00002.json").symlink_to(
        outside / "victim" / "20260408-120000-out00001.json"
    )
    assert [p for p, _ in list_records(observe_dir)] == [kept]
    assert find_record_path(observe_dir, "20260408-120000-out00001") is None
    assert find_record_path(observe_dir, "20260408-120000-out00002") is None


# ── credential redaction (#122) ──────────────────────────────────────────────

_CREDENTIALS = {
    "Authorization": "Bearer sk-secret-1",
    "PROXY-AUTHORIZATION": "Basic c2VjcmV0LTI=",
    "cookie": "session=secret-3",
    "Set-Cookie": "session=secret-4; HttpOnly",
    "X-Api-KEY": "secret-5",
}


def test_to_record_redacts_credential_headers_any_case() -> None:
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={**_CREDENTIALS, "content-type": "application/json"},
        request_body=b"{}",
        response_headers={**_CREDENTIALS, "x-request-id": "abc"},
    )
    record = ctx.to_record()
    for headers in (record.request_headers, record.response_headers):
        for name in _CREDENTIALS:
            assert headers[name] == REDACTED
    assert record.request_headers["content-type"] == "application/json"
    assert record.response_headers["x-request-id"] == "abc"
    # The context's own mappings are left alone; only the stored copy is redacted.
    assert ctx.request_headers["Authorization"] == "Bearer sk-secret-1"


def test_save_record_redacts_direct_callers(tmp_path: Path) -> None:
    record = _make_record()
    record.request_headers = dict(_CREDENTIALS)
    record.response_headers = dict(_CREDENTIALS)
    path = save_record(tmp_path, record)
    text = path.read_text()
    for value in _CREDENTIALS.values():
        assert value not in text
    assert text.count(REDACTED) == 2 * len(_CREDENTIALS)


# ── body caps and binary placeholders (#122) ─────────────────────────────────


def test_oversized_text_body_is_truncated_with_marker() -> None:
    body = b'{"prompt": "' + b"x" * (3 * MAX_BODY_BYTES) + b'"}'
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/completions",
        method="POST",
        request_headers={"content-type": "application/json"},
        request_body=body,
    )
    record = ctx.to_record()
    assert record.request_body is not None
    assert len(record.request_body.encode()) < MAX_BODY_BYTES + 200
    assert "[TRUNCATED" in record.request_body
    assert f"original_bytes={len(body)}" in record.request_body


def test_binary_bodies_become_placeholders_never_hex() -> None:
    upload = b"\xff\xd8\x00" * 1000
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/audio/transcriptions",
        method="POST",
        request_headers={"content-type": "multipart/form-data; boundary=x"},
        # Valid UTF-8 multipart is still omitted: it can carry whole files.
        request_body=b"--x\r\nContent-Disposition: form-data; name=file\r\n\r\nhello",
        request_body_size=123_456,
        response_headers={"content-type": "audio/wav"},
        response_body=upload,
    )
    record = ctx.to_record()
    assert record.request_body is not None
    assert "123456 bytes" in record.request_body
    assert "multipart/form-data" in record.request_body
    assert "hello" not in record.request_body
    assert f"{len(upload)} bytes" in record.response_body
    assert "audio/wav" in record.response_body_raw
    assert upload.hex()[:20] not in record.response_body_raw


def test_invalid_utf8_without_content_type_is_placeholder() -> None:
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/x",
        method="POST",
        request_headers={},
        request_body=b"\xff\xfe\xfd",
    )
    record = ctx.to_record()
    assert record.request_body is not None
    assert record.request_body.startswith("[binary body omitted: 3 bytes")


def test_saved_record_stays_under_cap_with_escaping_heavy_bodies(tmp_path: Path) -> None:
    # Control characters expand six-fold when JSON-escaped.
    body = b"\x01" * MAX_BODY_BYTES
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/completions",
        method="POST",
        request_headers={"content-type": "text/plain", "x-big": "v" * 100_000},
        request_body=body,
        response_headers={"content-type": "text/plain"},
        response_body=body,
    )
    path = save_record(tmp_path, ctx.to_record())
    assert path.stat().st_size <= MAX_RECORD_BYTES
    loaded = load_record(path)
    assert "[TRUNCATED" in loaded.response_body_raw
    assert len(loaded.request_headers["x-big"]) < 2000


# ── private files (#122) ─────────────────────────────────────────────────────


def test_saved_record_is_private_and_never_overwritten(tmp_path: Path) -> None:
    old_umask = os.umask(0)
    try:
        path = save_record(tmp_path, _make_record())
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        save_record(tmp_path, _make_record())


# ── retention (#122) ─────────────────────────────────────────────────────────


def _aged(path: Path, now: float, age_seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}")
    os.utime(path, (now - age_seconds, now - age_seconds))
    return path


def test_prune_records_deletes_only_expired_records(tmp_path: Path) -> None:
    now = time.time()
    day = 86400
    old_flat = _aged(tmp_path / "m" / "old.json", now, 31 * day)
    new_flat = _aged(tmp_path / "m" / "new.json", now, 29 * day)
    edge = _aged(tmp_path / "m" / "edge.json", now, 30 * day)
    old_ns = _aged(tmp_path / "org" / "repo:q4" / "old.json", now, 40 * day)
    new_ns = _aged(tmp_path / "org" / "repo:q4" / "new.json", now, day)
    old_unknown = _aged(tmp_path / "_unknown" / "old.json", now, 90 * day)
    unrelated = _aged(tmp_path / "m" / "notes.txt", now, 90 * day)
    target = _aged(tmp_path.parent / f"{tmp_path.name}-outside.json", now, 90 * day)
    link = tmp_path / "m" / "link.json"
    link.symlink_to(target)

    removed = prune_records(tmp_path, 30, now=now)

    assert removed == 3
    for gone in (old_flat, old_ns, old_unknown):
        assert not gone.exists()
    for kept in (new_flat, edge, new_ns, unrelated, target):
        assert kept.exists()
    assert link.is_symlink()


def test_prune_records_zero_and_missing_dir_are_noops(tmp_path: Path) -> None:
    now = time.time()
    old = _aged(tmp_path / "m" / "old.json", now, 365 * 86400)
    assert prune_records(tmp_path, 0, now=now) == 0
    assert old.exists()
    assert prune_records(tmp_path / "missing", 30, now=now) == 0
    assert not (tmp_path / "missing").exists()


# ── records from older versions (#122) ───────────────────────────────────────


def test_legacy_record_without_new_fields_still_loads(tmp_path: Path) -> None:
    legacy = {
        "id": "20250101-000000-deadbeef",
        "timestamp": "2025-01-01T00:00:00",
        "model": "testmodel",
        "endpoint": "v1/audio/transcriptions",
        "method": "POST",
        "request_headers": {"authorization": "Bearer old"},
        "request_body": "fffe00",
        "response_status": 200,
        "response_headers": {},
        "response_body": "{}",
    }
    path = tmp_path / "testmodel" / f"{legacy['id']}.json"
    path.parent.mkdir()
    path.write_text(json.dumps(legacy))

    record = load_record(path)
    assert record.request_body == "fffe00"
    assert record.response_body_raw == ""
    assert [p for p, _ in list_records(tmp_path)] == [path]


@pytest.mark.parametrize(("flat_age", "nested_age"), [(1, 40), (40, 1), (40, 40)])
def test_prune_records_handles_flat_and_namespaced_records_under_one_parent(
    tmp_path: Path, flat_age: int, nested_age: int
) -> None:
    # "org" and "org/repo:q4" are both valid model names and can coexist.
    now = time.time()
    flat = _aged(tmp_path / "org" / "flat.json", now, flat_age * 86400)
    nested = _aged(tmp_path / "org" / "repo:q4" / "nested.json", now, nested_age * 86400)

    removed = prune_records(tmp_path, 30, now=now)

    assert flat.exists() == (flat_age < 30)
    assert nested.exists() == (nested_age < 30)
    assert removed == (flat_age >= 30) + (nested_age >= 30)


# ── directory swap races (#122 review) ───────────────────────────────────────


def _swap_to_outward_link(directory: Path, outside: Path) -> None:
    """Replace a validated directory with a symlink pointing outside observe_dir."""
    directory.rename(directory.with_name(directory.name + "-moved"))
    directory.symlink_to(outside, target_is_directory=True)


def _outside_victim(tmp_path: Path, now: float) -> Path:
    victim = tmp_path / "outside" / "old.json"
    return _aged(victim, now, 365 * 86400)


def test_prune_never_follows_a_directory_swapped_before_it_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    observe_dir = tmp_path / "observe"
    _aged(observe_dir / "m" / "old.json", now, 365 * 86400)
    victim = _outside_victim(tmp_path, now)
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if path == "m" and dir_fd is not None and not swapped:
            swapped = True
            _swap_to_outward_link(observe_dir / "m", victim.parent)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    prune_records(observe_dir, 30, now=now)

    assert swapped
    assert victim.exists()


def test_prune_never_follows_a_directory_swapped_before_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    observe_dir = tmp_path / "observe"
    _aged(observe_dir / "m" / "old.json", now, 365 * 86400)
    victim = _outside_victim(tmp_path, now)
    real_unlink = os.unlink
    swapped = False

    def racing_unlink(path, *, dir_fd=None):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if not swapped:
            swapped = True
            _swap_to_outward_link(observe_dir / "m", victim.parent)
        return real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", racing_unlink)
    removed = prune_records(observe_dir, 30, now=now)

    assert swapped
    assert victim.exists()
    # The pinned directory (now moved) lost its expired record instead.
    assert removed == 1
    assert not (observe_dir / "m-moved" / "old.json").exists()


def test_save_record_never_writes_through_a_swapped_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observe_dir = tmp_path / "observe"
    (observe_dir / "testmodel").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if path == "testmodel" and dir_fd is not None and not swapped:
            swapped = True
            _swap_to_outward_link(observe_dir / "testmodel", outside)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(OSError):
        save_record(observe_dir, _make_record())

    assert swapped
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("record_id", ["sub/x", "../x", ""])
def test_save_record_rejects_ids_that_are_not_plain_names(tmp_path: Path, record_id: str) -> None:
    with pytest.raises(ValueError):
        save_record(tmp_path, _make_record(record_id=record_id))


def test_saved_record_stays_under_cap_with_oversized_metadata(tmp_path: Path) -> None:
    ctx = ObserveContext(
        timestamp="2026-04-08T12:00:00",
        model="testmodel",
        endpoint="v1/" + "e" * (2 * MAX_RECORD_BYTES),
        method="M" * 100_000,
        request_headers={f"x-h{i}": "\x01" * 5000 for i in range(200)},
        request_body=b"\x01" * MAX_BODY_BYTES,
        response_headers={"content-type": "text/plain"},
        response_body=b"\x01" * MAX_BODY_BYTES,
    )
    path = save_record(tmp_path, ctx.to_record())
    assert path.stat().st_size <= MAX_RECORD_BYTES
    loaded = load_record(path)
    assert loaded.endpoint.startswith("v1/eee")
    assert "[TRUNCATED" in loaded.endpoint
    assert "[TRUNCATED" in loaded.method


def test_record_falls_back_to_metadata_only_and_never_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drove.observe as observe_module

    monkeypatch.setattr(observe_module, "MAX_RECORD_BYTES", 4096)
    record = _make_record()
    record.request_headers = {f"x-h{i}": "v" * 1000 for i in range(20)}
    path = save_record(tmp_path, record)
    assert path.stat().st_size <= 4096
    loaded = load_record(path)
    assert loaded.request_headers == {}
    assert loaded.request_body is not None
    assert loaded.request_body.startswith("[omitted: record exceeded 4096 bytes]")

    monkeypatch.setattr(observe_module, "MAX_RECORD_BYTES", 10)
    with pytest.raises(ValueError, match="cannot fit"):
        save_record(tmp_path, _make_record(record_id="20260408-120000-other000"))
    assert not (tmp_path / "testmodel" / "20260408-120000-other000.json").exists()


# ── capture boundaries and type rules (#153) ─────────────────────────────────


def test_text_cut_inside_a_character_at_the_capture_limit_stays_text() -> None:
    from drove.observe import render_body

    body = ("a" * (MAX_BODY_BYTES - 1) + "é" + "tail").encode()  # é straddles the limit
    captured = body[:MAX_BODY_BYTES]

    rendered = render_body(captured, len(body), "text/plain")

    assert not rendered.startswith("[binary body omitted")
    assert rendered.startswith("a" * 100)
    assert f"original_bytes={len(body)}" in rendered


def test_invalid_utf8_before_the_limit_is_binary() -> None:
    from drove.observe import render_body

    body = b"a" * 100 + b"\xff" + b"a" * 100
    assert render_body(body, len(body), "") == (
        f"[binary body omitted: {len(body)} bytes, content-type: unknown]"
    )


def test_headers_beyond_the_count_cap_are_summarised() -> None:
    from drove.observe import sanitize_headers

    headers = {f"x-h{i}": "v" for i in range(70)}
    out = sanitize_headers(headers)

    assert len(out) == 65
    assert out["[TRUNCATED]"] == "6 more headers omitted"
    assert list(out)[:64] == [f"x-h{i}" for i in range(64)]


@pytest.mark.parametrize(
    ("content_type", "binary"),
    [
        ("", False),
        ("application/json; charset=utf-8", False),
        ("APPLICATION/X-NDJSON", False),
        ("application/x-www-form-urlencoded", False),
        ("application/xml", False),
        ("text/event-stream", False),
        ("application/vnd.api+json", False),
        ("multipart/form-data; boundary=x", True),
        ("application/octet-stream", True),
        ("audio/wav", True),
        ("image/png", True),
        ("application/pdf", True),
    ],
)
def test_text_type_allowlist(content_type: str, binary: bool) -> None:
    from drove.observe import is_binary_content_type

    assert is_binary_content_type(content_type) is binary
