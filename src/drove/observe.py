"""Request/response observation logging."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from drove.model_store import is_within, validate_model_name

logger = logging.getLogger(__name__)

# Credential headers are never written to disk, in either direction.
CREDENTIAL_HEADERS = frozenset(
    ["authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"]
)
REDACTED = "[REDACTED]"

# Each captured body keeps at most this many bytes; a saved record is at most
# MAX_RECORD_BYTES of encoded JSON, whatever the traffic looked like.
MAX_BODY_BYTES = 256 * 1024
MAX_RECORD_BYTES = 1024 * 1024
_MAX_HEADERS = 64
_MAX_HEADER_CHARS = 1024

_TEXT_TYPES = frozenset(
    [
        "application/json",
        "application/x-ndjson",
        "application/x-www-form-urlencoded",
        "application/xml",
    ]
)


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


def _truncation_marker(original_bytes: int) -> str:
    return f"\n[TRUNCATED; original_bytes={original_bytes}]"


def _truncate_text(text: str, max_bytes: int, original_bytes: int) -> str:
    """Cut text to at most max_bytes of UTF-8 (on a character boundary) plus a marker."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes and original_bytes <= len(encoded):
        return text
    cut = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return cut + _truncation_marker(original_bytes)


def _content_type(headers: dict[str, str]) -> str:
    for name, value in headers.items():
        if name.lower() == "content-type":
            return value
    return ""


def _is_text_type(content_type: str) -> bool:
    mime = content_type.split(";", 1)[0].strip().lower()
    return mime.startswith("text/") or mime.endswith("+json") or mime in _TEXT_TYPES


def _decode_prefix(data: bytes) -> str:
    """Decode a captured prefix whose last character may have been cut mid-sequence."""
    return data[:MAX_BODY_BYTES].decode("utf-8", errors="ignore")


def _is_binary(data: bytes, content_type: str) -> bool:
    """Whether a body must be stored as a placeholder rather than as text.

    A declared non-text type (multipart, audio, image, octet-stream…) is binary
    even when its bytes happen to decode, since uploads can carry whole files.
    Text, or an undeclared type, is binary only when it is not valid UTF-8.
    """
    if is_binary_content_type(content_type):
        return True
    prefix = data[:MAX_BODY_BYTES]
    try:
        prefix.decode("utf-8")
    except UnicodeDecodeError as e:
        # A capture cut inside a multi-byte character is still text.
        return not (e.reason == "unexpected end of data" and e.start >= len(prefix) - 3)
    return False


def is_binary_content_type(content_type: str) -> bool:
    """Whether a declared content type is stored as a placeholder, never as text."""
    return bool(content_type) and not _is_text_type(content_type)


def render_body(data: bytes, total_bytes: int, content_type: str) -> str:
    """Render a captured body for storage: bounded text, or a binary placeholder."""
    if _is_binary(data, content_type):
        kind = content_type or "unknown"
        return f"[binary body omitted: {total_bytes} bytes, content-type: {kind}]"
    return _truncate_text(_decode_prefix(data), MAX_BODY_BYTES, total_bytes)


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Copy headers for storage: credentials redacted, sizes bounded."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        if len(out) >= _MAX_HEADERS:
            out["[TRUNCATED]"] = f"{len(headers) - _MAX_HEADERS} more headers omitted"
            break
        name = name[:_MAX_HEADER_CHARS]
        if name.lower() in CREDENTIAL_HEADERS:
            out[name] = REDACTED
        elif len(value) > _MAX_HEADER_CHARS:
            out[name] = value[:_MAX_HEADER_CHARS] + _truncation_marker(len(value.encode()))
        else:
            out[name] = value
    return out


@dataclass
class ObserveContext:
    """Accumulates data through a request lifecycle for observe logging."""

    timestamp: str
    model: str | None
    endpoint: str
    method: str
    request_headers: dict[str, str]
    request_body: bytes  # at most MAX_BODY_BYTES are rendered
    request_body_size: int | None = None  # original length; None = len(request_body)

    # Populated after response completes
    response_status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: bytes = b""  # at most MAX_BODY_BYTES are rendered
    response_body_size: int | None = None  # original length; None = len(response_body)
    ttft_seconds: float | None = None
    duration_seconds: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_per_second: float | None = None

    def to_record(self) -> ObserveRecord:
        ts = self.timestamp.replace(":", "").replace("-", "").replace("T", "-")[:15]
        record_id = f"{ts}-{uuid.uuid4().hex[:8]}"

        req_size = (
            len(self.request_body) if self.request_body_size is None else (self.request_body_size)
        )
        req_body = (
            render_body(self.request_body, req_size, _content_type(self.request_headers))
            if req_size
            else None
        )

        resp_size = (
            len(self.response_body)
            if self.response_body_size is None
            else (self.response_body_size)
        )
        resp_type = _content_type(self.response_headers)
        resp_body_raw = render_body(self.response_body, resp_size, resp_type)
        if _is_binary(self.response_body, resp_type):
            resp_body = resp_body_raw
        else:
            # Assemble a readable version from the (possibly truncated) raw response
            resp_body = _assemble_response(_decode_prefix(self.response_body))
            if resp_size > len(self.response_body) or len(self.response_body) > MAX_BODY_BYTES:
                resp_body = _truncate_text(resp_body, MAX_BODY_BYTES, resp_size)

        return ObserveRecord(
            id=record_id,
            timestamp=self.timestamp,
            model=self.model,
            endpoint=self.endpoint,
            method=self.method,
            request_headers=sanitize_headers(self.request_headers),
            request_body=req_body,
            response_status=self.response_status,
            response_headers=sanitize_headers(self.response_headers),
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
    """Write an observe record to disk. Returns the file path.

    Raises ValueError, writing nothing, when the record's model is not a
    valid model name or its directory would resolve outside *observe_dir*.
    """
    if record.model is None:
        model_dir = observe_dir / "_unknown"
    else:
        model_dir = observe_dir / validate_model_name(record.model)
    if not is_within(observe_dir, model_dir):
        raise ValueError(f"Observe directory for '{record.model}' is outside {observe_dir}.")
    path = model_dir / f"{record.id}.json"
    if not is_within(observe_dir, path):
        raise ValueError(f"Observe record {record.id} would be written outside {observe_dir}.")
    model_dir.mkdir(parents=True, exist_ok=True)
    data = _encode_record(record)
    # O_EXCL never follows or reuses an existing file; 0600 applies from creation.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _encode_record(record: ObserveRecord) -> bytes:
    """Encode a record as JSON of at most MAX_RECORD_BYTES, credentials redacted.

    Bodies are shortened (with a truncation marker) until the encoded document
    fits, so escaping and pretty-printing cannot push a record past the cap.
    """
    data = _record_to_dict(record)
    data["request_headers"] = sanitize_headers(record.request_headers)
    data["response_headers"] = sanitize_headers(record.response_headers)
    body_fields = ("request_body", "response_body", "response_body_raw")
    budget = MAX_BODY_BYTES
    while True:
        encoded = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        if len(encoded) <= MAX_RECORD_BYTES or budget < 64:
            return encoded
        budget //= 2
        for name in body_fields:
            value = data[name]
            if isinstance(value, str) and len(value.encode("utf-8")) > budget:
                data[name] = _truncate_text(value, budget, len(value.encode("utf-8")))


def load_record(path: Path) -> ObserveRecord:
    """Load a single observe record from disk."""
    data = json.loads(path.read_text())
    return _record_from_dict(data)


def _check_cancelled(stop: Event | None) -> None:
    if stop is not None and stop.is_set():
        raise CancelledError("Record scan stopped")


def _record_dirs(observe_dir: Path, model: str | None, stop: Event | None = None) -> list[Path]:
    """Resolve the directories that hold record JSON files.

    If model is None, walks two levels deep so namespaced models
    (``org/repo:quant``) are found alongside legacy flat model dirs.
    """
    _check_cancelled(stop)
    if model:
        # A filter is a lookup key, never a path: an unsafe or escaping one
        # matches nothing rather than reading outside observe_dir.
        try:
            model_dir = observe_dir / validate_model_name(model)
        except ValueError:
            return []
        return [model_dir] if is_within(observe_dir, model_dir) else []

    dirs: list[Path] = []
    for d in observe_dir.iterdir():
        _check_cancelled(stop)
        if not d.is_dir() or not is_within(observe_dir, d):
            continue
        has_records = False
        for entry in d.iterdir():
            _check_cancelled(stop)
            if entry.match("*.json"):
                has_records = True
                break
        if has_records:
            dirs.append(d)
        else:
            # Treat as namespace — scan one level deeper
            for sub in d.iterdir():
                _check_cancelled(stop)
                if sub.is_dir() and is_within(observe_dir, sub):
                    dirs.append(sub)
    return dirs


def _record_paths(observe_dir: Path, model: str | None, stop: Event | None = None) -> list[Path]:
    """All record file paths, newest first.

    Sorted by filename, which begins with the record id's timestamp prefix
    (``YYYYMMDD-HHMMSS``), so the order matches a timestamp sort without
    reading any files.
    """
    _check_cancelled(stop)
    if not observe_dir.exists():
        return []

    paths: list[Path] = []
    for d in _record_dirs(observe_dir, model, stop):
        _check_cancelled(stop)
        if not d.exists():
            continue
        for entry in d.iterdir():
            _check_cancelled(stop)
            # The directory is already contained; only a symlinked file can escape.
            if entry.match("*.json") and not (
                entry.is_symlink() and not is_within(observe_dir, entry)
            ):
                paths.append(entry)
    _check_cancelled(stop)
    paths.sort(key=lambda p: p.name, reverse=True)
    _check_cancelled(stop)
    return paths


def list_records(observe_dir: Path, model: str | None = None) -> list[tuple[Path, ObserveRecord]]:
    """List observe records, newest first.

    If model is None, lists across all model subdirectories (including
    namespaced dirs like ``org/repo:quant/``).  Returns (path, record) tuples.
    """
    results: list[tuple[Path, ObserveRecord]] = []
    for p in _record_paths(observe_dir, model):
        try:
            results.append((p, load_record(p)))
        except json.JSONDecodeError, KeyError, ValueError:
            continue
    return results


def list_records_page(
    observe_dir: Path,
    model: str | None = None,
    offset: int = 0,
    limit: int = 100,
    *,
    stop: Event | None = None,
) -> tuple[list[tuple[Path, ObserveRecord]], int]:
    """Return one page of records (newest first) plus the total record count.

    Only the files within the requested window are read from disk; the rest
    are ordered by filename alone, so large logs do not pay the cost of
    parsing every record on each call.  Files that fail to load are skipped
    in the page but still counted in the total. An optional stop event raises
    CancelledError between filesystem operations when the caller shuts down.
    """
    paths = _record_paths(observe_dir, model, stop)
    total = len(paths)
    page: list[tuple[Path, ObserveRecord]] = []
    for p in paths[offset : offset + limit]:
        _check_cancelled(stop)
        try:
            page.append((p, load_record(p)))
        except json.JSONDecodeError, KeyError, ValueError:
            continue
    return page, total


def search_records_page(
    observe_dir: Path,
    query: str,
    model: str | None = None,
    offset: int = 0,
    limit: int = 100,
    *,
    stop: Event | None = None,
) -> tuple[list[tuple[Path, ObserveRecord]], int]:
    """Return one page of records matching query (newest first) plus the total match count.

    Search has to parse every record to evaluate the match, but only the
    matches inside the requested window are kept, so memory stays bounded by
    the page size rather than the full log. An optional stop event raises
    CancelledError between filesystem operations when the caller shuts down.
    """
    page: list[tuple[Path, ObserveRecord]] = []
    matched = 0
    for p in _record_paths(observe_dir, model, stop):
        _check_cancelled(stop)
        try:
            record = load_record(p)
        except json.JSONDecodeError, KeyError, ValueError:
            continue
        if not record_matches(record, query):
            continue
        if offset <= matched < offset + limit:
            page.append((p, record))
        matched += 1
    return page, matched


def find_record_path(observe_dir: Path, record_id: str, model: str | None = None) -> Path | None:
    """Locate a record file by id without parsing any records.

    Record ids double as filename stems, so the lookup is a direct filename
    match across the record directories.
    """
    if not observe_dir.exists():
        return None
    name = f"{record_id}.json"
    for d in _record_dirs(observe_dir, model):
        candidate = d / name
        if candidate.is_file() and is_within(observe_dir, candidate):
            return candidate
    return None


def prune_records(observe_dir: Path, max_age_days: int, now: float | None = None) -> int:
    """Delete record files last modified more than max_age_days ago.

    Age is the file's modification time, so no record is parsed. Only regular
    ``*.json`` files in the record directories are considered: symlinks,
    other files and directories are left alone, and a missing observe_dir is
    not created. Zero (or less) disables pruning. Returns the number deleted.
    """
    if max_age_days <= 0 or not observe_dir.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_days * 86400
    removed = 0
    for d in _record_dirs(observe_dir, None):
        try:
            with os.scandir(d) as entries:
                for entry in entries:
                    if entry.name.endswith(".json") and _prune_entry(entry, cutoff):
                        removed += 1
        except OSError as e:
            logger.warning("Observe retention could not read %s: %s", d, e)
    return removed


def _prune_entry(entry: os.DirEntry[str], cutoff: float) -> bool:
    """Delete one expired regular record file; return whether it was deleted."""
    try:
        if not entry.is_file(follow_symlinks=False):
            return False
        if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
            return False
        os.unlink(entry.path)
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("Observe retention could not delete %s: %s", entry.path, e)
        return False
    return True
