"""Request/response observation logging."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ObserveRecord:
    """A single logged API request/response pair."""

    id: str
    timestamp: str
    model: str | None
    endpoint: str
    method: str

    # Request
    request_headers: dict[str, str]
    request_body: str | None

    # Response
    response_status: int
    response_headers: dict[str, str]
    response_body: str  # assembled/readable version
    response_body_raw: str = ""  # original raw response (SSE lines etc.)

    # Metrics
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_per_second: float | None = None
    ttft_seconds: float | None = None
    duration_seconds: float = 0.0


def _try_parse_json(s: str) -> Any:
    """Parse a JSON string, returning the parsed object or the original string."""
    try:
        return json.loads(s)
    except json.JSONDecodeError, ValueError:
        return s


def _assemble_response(raw: str) -> str:
    """Assemble a readable response from raw response data.

    For SSE streaming responses, concatenates the content/reasoning deltas into
    a clean text output.  For plain JSON responses, returns them as-is.
    """
    # If it parses as plain JSON, return it directly (non-streaming response)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError, ValueError:
        pass

    # Try to parse as SSE stream and reassemble
    has_sse = False
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, str]] = {}  # index -> {id, name, arguments}
    model: str | None = None
    usage: dict[str, Any] | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data: ") or stripped == "data: [DONE]":
            continue
        try:
            obj = json.loads(stripped[6:])
            if not isinstance(obj, dict):
                continue
            has_sse = True

            if not model and obj.get("model"):
                model = obj["model"]
            if obj.get("usage"):
                usage = obj["usage"]

            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning_parts.append(delta["reasoning_content"])

                # Accumulate tool call chunks by index
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.get("id"):
                        tool_calls[idx]["id"] = tc["id"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        tool_calls[idx]["name"] = func["name"]
                    if func.get("arguments"):
                        tool_calls[idx]["arguments"] += func["arguments"]
        except json.JSONDecodeError, ValueError:
            continue

    if not has_sse:
        # Not SSE and not valid JSON — return as-is
        return raw

    # Build assembled output
    assembled: dict[str, Any] = {}
    if model:
        assembled["model"] = model
    if reasoning_parts:
        assembled["reasoning"] = "".join(reasoning_parts)
    if content_parts:
        assembled["content"] = "".join(content_parts)
    if tool_calls:
        assembled["tool_calls"] = [
            {
                "id": tc["id"],
                "function": tc["name"],
                "arguments": _try_parse_json(tc["arguments"]),
            }
            for tc in sorted(tool_calls.values(), key=lambda t: t["id"])
        ]
    if usage:
        assembled["usage"] = usage

    return json.dumps(assembled, indent=2, ensure_ascii=False)


@dataclass
class ObserveContext:
    """Accumulates data through a request lifecycle for observe logging."""

    timestamp: str
    model: str | None
    endpoint: str
    method: str
    request_headers: dict[str, str]
    request_body: bytes

    # Populated after response completes
    response_status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: bytes = b""
    ttft_seconds: float | None = None
    duration_seconds: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_per_second: float | None = None

    def to_record(self) -> ObserveRecord:
        ts = self.timestamp.replace(":", "").replace("-", "").replace("T", "-")[:15]
        record_id = f"{ts}-{uuid.uuid4().hex[:8]}"

        # Decode bodies to strings
        try:
            req_body = self.request_body.decode("utf-8") if self.request_body else None
        except UnicodeDecodeError:
            req_body = self.request_body.hex() if self.request_body else None

        try:
            resp_body_raw = self.response_body.decode("utf-8", errors="replace")
        except Exception:
            resp_body_raw = self.response_body.hex()

        # Assemble a readable version from the raw response
        resp_body = _assemble_response(resp_body_raw)

        return ObserveRecord(
            id=record_id,
            timestamp=self.timestamp,
            model=self.model,
            endpoint=self.endpoint,
            method=self.method,
            request_headers=self.request_headers,
            request_body=req_body,
            response_status=self.response_status,
            response_headers=self.response_headers,
            response_body=resp_body,
            response_body_raw=resp_body_raw,
            tokens_prompt=self.tokens_prompt,
            tokens_completion=self.tokens_completion,
            tokens_per_second=self.tokens_per_second,
            ttft_seconds=self.ttft_seconds,
            duration_seconds=self.duration_seconds,
        )


def _record_to_dict(record: ObserveRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "timestamp": record.timestamp,
        "model": record.model,
        "endpoint": record.endpoint,
        "method": record.method,
        "request_headers": record.request_headers,
        "request_body": record.request_body,
        "response_status": record.response_status,
        "response_headers": record.response_headers,
        "response_body": record.response_body,
        "response_body_raw": record.response_body_raw,
        "tokens_prompt": record.tokens_prompt,
        "tokens_completion": record.tokens_completion,
        "tokens_per_second": record.tokens_per_second,
        "ttft_seconds": record.ttft_seconds,
        "duration_seconds": record.duration_seconds,
    }


def _record_from_dict(data: dict[str, Any]) -> ObserveRecord:
    return ObserveRecord(
        id=str(data["id"]),
        timestamp=str(data["timestamp"]),
        model=str(data["model"]) if data.get("model") else None,
        endpoint=str(data["endpoint"]),
        method=str(data["method"]),
        request_headers=dict(data.get("request_headers") or {}),
        request_body=str(data["request_body"]) if data.get("request_body") is not None else None,
        response_status=int(data.get("response_status", 0)),
        response_headers=dict(data.get("response_headers") or {}),
        response_body=str(data.get("response_body", "")),
        response_body_raw=str(data.get("response_body_raw", "")),
        tokens_prompt=int(data.get("tokens_prompt", 0)),
        tokens_completion=int(data.get("tokens_completion", 0)),
        tokens_per_second=float(data["tokens_per_second"])
        if data.get("tokens_per_second") is not None
        else None,
        ttft_seconds=float(data["ttft_seconds"]) if data.get("ttft_seconds") is not None else None,
        duration_seconds=float(data.get("duration_seconds", 0.0)),
    )


def record_matches(record: ObserveRecord, query: str) -> bool:
    """Return True if any searchable field of the record contains query (case-insensitive).

    Matches across id, timestamp, model, endpoint, method, status, bodies, headers,
    and metrics so a single search box works as a global filter.
    """
    if not query:
        return True
    q = query.lower()
    parts: list[str] = [
        record.id,
        record.timestamp,
        record.model or "",
        record.endpoint,
        record.method,
        str(record.response_status),
        record.request_body or "",
        record.response_body or "",
        record.response_body_raw or "",
        json.dumps(record.request_headers, ensure_ascii=False),
        json.dumps(record.response_headers, ensure_ascii=False),
        str(record.tokens_prompt),
        str(record.tokens_completion),
        "" if record.tokens_per_second is None else str(record.tokens_per_second),
        "" if record.ttft_seconds is None else str(record.ttft_seconds),
        str(record.duration_seconds),
    ]
    return any(q in p.lower() for p in parts)


def save_record(observe_dir: Path, record: ObserveRecord) -> Path:
    """Write an observe record to disk. Returns the file path."""
    model_dir = observe_dir / (record.model or "_unknown")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{record.id}.json"
    path.write_text(json.dumps(_record_to_dict(record), indent=2, ensure_ascii=False))
    return path


def load_record(path: Path) -> ObserveRecord:
    """Load a single observe record from disk."""
    data = json.loads(path.read_text())
    return _record_from_dict(data)


def list_records(observe_dir: Path, model: str | None = None) -> list[tuple[Path, ObserveRecord]]:
    """List observe records, newest first.

    If model is None, lists across all model subdirectories (including
    namespaced dirs like ``org/repo:quant/``).  Returns (path, record) tuples.
    """
    if not observe_dir.exists():
        return []

    if model:
        dirs = [observe_dir / model]
    else:
        # Walk two levels deep so namespaced models (org/repo:quant) are found
        # alongside legacy flat model dirs.
        dirs = []
        for d in observe_dir.iterdir():
            if not d.is_dir():
                continue
            if any(d.glob("*.json")):
                dirs.append(d)
            else:
                # Treat as namespace — scan one level deeper
                for sub in d.iterdir():
                    if sub.is_dir():
                        dirs.append(sub)

    results: list[tuple[Path, ObserveRecord]] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            try:
                record = load_record(p)
                results.append((p, record))
            except json.JSONDecodeError, KeyError, ValueError:
                continue

    results.sort(key=lambda x: x[1].timestamp, reverse=True)
    return results
