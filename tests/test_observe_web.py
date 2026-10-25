"""Tests for the observe web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from drove.observe import ObserveRecord, save_record
from drove.observe_web import create_observe_app


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
    assert "drove observe" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_api_records_empty(tmp_path: Path) -> None:
    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


def test_api_records_lists_records(tmp_path: Path) -> None:
    record = _make_record()
    save_record(tmp_path, record)

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records")
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == record.id
    assert data["items"][0]["model"] == "testmodel"
    assert data["items"][0]["tokens_prompt"] == 10


def test_api_records_filters_by_model(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(model="modelA", record_id="20260408-100000-aaaa0001"))
    save_record(tmp_path, _make_record(model="modelB", record_id="20260408-100000-aaaa0002"))

    app = create_observe_app(tmp_path, model="modelA")
    with TestClient(app) as client:
        resp = client.get("/api/records")
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["model"] == "modelA"


def test_api_records_paginates(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(record_id="20260408-100000-aaaa0001"))
    save_record(tmp_path, _make_record(record_id="20260408-100100-aaaa0002"))
    save_record(tmp_path, _make_record(record_id="20260408-100200-aaaa0003"))

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records?limit=2")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 2

        resp2 = client.get("/api/records?offset=2&limit=2")
        data2 = resp2.json()
        assert data2["total"] == 3
        assert len(data2["items"]) == 1


def test_api_records_search_paginates(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(model="alpha", record_id="20260408-100000-aaaa0001"))
    save_record(tmp_path, _make_record(model="alpha", record_id="20260408-100100-aaaa0002"))
    save_record(tmp_path, _make_record(model="beta", record_id="20260408-100200-aaaa0003"))

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records?search=alpha&limit=1")
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 1
    assert data["items"][0]["model"] == "alpha"


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


def test_api_record_detail_corrupt_file_is_404(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record())
    (tmp_path / "testmodel" / "20260408-130000-corrupt0.json").write_text("not json")

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records/20260408-130000-corrupt0")
    assert resp.status_code == 404


def test_api_records_search_offset(tmp_path: Path) -> None:
    save_record(tmp_path, _make_record(model="alpha", record_id="20260408-100000-aaaa0001"))
    save_record(tmp_path, _make_record(model="alpha", record_id="20260408-100100-aaaa0002"))
    save_record(tmp_path, _make_record(model="beta", record_id="20260408-100200-aaaa0003"))

    app = create_observe_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/records?search=alpha&offset=1&limit=1")
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == "20260408-100000-aaaa0001"


def test_script_assets_are_served_locally(tmp_path: Path) -> None:
    """A fresh browser must not need a CDN to render logged Markdown."""
    from html.parser import HTMLParser
    from urllib.parse import urlsplit

    class Scripts(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.sources: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "script":
                source = dict(attrs).get("src")
                if source:
                    self.sources.append(source)

    with TestClient(create_observe_app(tmp_path)) as client:
        scripts = Scripts()
        scripts.feed(client.get("/").text)
        assert len(scripts.sources) == 3
        assert scripts.sources[-1] == "static/app.js"
        for source in scripts.sources:
            assert not urlsplit(source).scheme
            assert not urlsplit(source).netloc
            response = client.get("/" + source)
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
            assert response.content


def test_vendored_markdown_assets_match_provenance(tmp_path: Path) -> None:
    """Pinned distribution files and complete licenses travel with the UI."""
    import hashlib

    with TestClient(create_observe_app(tmp_path)) as client:
        response = client.get("/static/vendor/manifest.json")
        assert response.status_code == 200
        manifest = response.json()
        assert {library["name"] for library in manifest} == {"marked", "DOMPurify"}
        scripts = client.get("/").text
        for library in manifest:
            assert library["version"]
            assert library["source"].startswith("https://registry.npmjs.org/")
            assert library["integrity"].startswith("sha512-")
            assert library["license"]
            for asset in library["files"]:
                served = client.get("/static/vendor/" + asset["path"])
                assert served.status_code == 200
                assert hashlib.sha256(served.content).hexdigest() == asset["sha256"]
                assert len(served.content) > 100
                if asset["path"].endswith(".js"):
                    assert scripts.index(asset["path"]) < scripts.index("static/app.js")
