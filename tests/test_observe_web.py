"""Tests for the observe web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from vllama.observe import ObserveRecord, save_record
from vllama.observe_web import create_observe_app


def _make_record(
    model: str = "testmodel",
    record_id: str = "20260408-120000-abcd1234",
) -> ObserveRecord:
    return ObserveRecord(
        id=record_id,
        timestamp="2026-04-08T12:00:00",
        model=model,
        endpoint="v1/chat/completions",
        method="POST",
        request_headers={"content-type": "application/json"},
        request_body='{"model": "testmodel", "messages": []}',
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body='{"content": "Hello"}',
        response_body_raw='data: {"choices":[{"delta":{"content":"Hello"}}]}',
        tokens_prompt=10,
        tokens_completion=5,
        tokens_per_second=25.0,
        ttft_seconds=0.1,
        duration_seconds=0.5,
    )


def test_index_returns_html(tmp_path: Path) -> None:
    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "vllama observe" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_api_records_empty(tmp_path: Path) -> None:
    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_records_lists_records(tmp_path: Path) -> None:
    record = _make_record()
    save_record(tmp_path, record)

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == record.id
    assert data[0]["model"] == "testmodel"
    assert data[0]["tokens_prompt"] == 10


def test_api_records_filters_by_model(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(model="modelA", record_id="20260408-100000-aaaa0001"))
    save_record(tmp_path, _make_record(model="modelB", record_id="20260408-100000-aaaa0002"))

    app = create_observe_app(tmp_path, model="modelA")
    with TestClient(app) as client:
        resp = client.get("/api/records")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["model"] == "modelA"


def test_api_record_detail(tmp_path: Path) -> None:
    record = _make_record()
    save_record(tmp_path, record)

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get(f"/api/records/{record.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == record.id
    assert data["request_body"] == '{"model": "testmodel", "messages": []}'
    assert data["response_body"] == '{"content": "Hello"}'
    assert data["response_body_raw"] is not None


def test_api_record_not_found(tmp_path: Path) -> None:
    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records/nonexistent")
    assert resp.status_code == 404
