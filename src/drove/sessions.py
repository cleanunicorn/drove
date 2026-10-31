"""Chat session persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from drove.model_store import is_within, validate_model_name
from drove.tool_policy import cap_tool_messages

logger = logging.getLogger(__name__)

_SLUG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Generated ids are timestamps with optional suffixes; anything else could
# name a path outside the model's session directory.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

# Every role drove has ever saved; chat replay renders user/assistant and
# turns the role into a CSS class, so any other string would crash it.
_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})

Warn = Callable[[str], None]


def session_endpoint(base_url: str) -> str:
    """Normalize endpoint identity without storing URL credentials or query secrets."""
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Chat endpoint must be an HTTP or HTTPS URL with a host.")
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, host, path, "", ""))


class Session:
    def __init__(
        self,
        model: str,
        session_id: str,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.model = model
        self.id = session_id
        self.messages = messages
        self.system_prompt = system_prompt
        self.endpoint = session_endpoint(endpoint) if endpoint is not None else None
        now = datetime.now().isoformat(timespec="seconds")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @property
    def title(self) -> str:
        """First user message, truncated, as a human-readable title."""
        for m in self.messages:
            if m["role"] == "user":
                text: str = m["content"].replace("\n", " ").strip()
                return text[:60] + ("…" if len(text) > 60 else "")
        return "(empty)"

    @property
    def message_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "system_prompt": self.system_prompt,
            "endpoint": self.endpoint,
            # Tool results are capped here too, so a saved session never holds a larger one.
            "messages": cap_tool_messages(self.messages),
        }


def new_session(
    model: str, system_prompt: str | None = None, *, endpoint: str | None = None
) -> Session:
    # Microseconds keep ids in chronological order; the random suffix keeps two
    # chats started at the same instant (e.g. in two windows) in separate files.
    session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{uuid4().hex[:8]}"
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    return Session(
        model=model,
        session_id=session_id,
        messages=messages,
        system_prompt=system_prompt,
        endpoint=endpoint,
    )


def _session_dir(sessions_dir: Path, model: str) -> Path:
    """Return the directory holding *model*'s sessions, always inside *sessions_dir*.

    Valid model names keep their readable ``org/name`` layout.  Any other label
    (remote endpoints may name models anything) maps to one flat ``_``-prefixed
    directory with a digest suffix, which no valid name can collide with.
    """
    try:
        directory = sessions_dir / validate_model_name(model)
    except ValueError:
        directory = None
    if directory is None or not is_within(sessions_dir, directory):
        safe = _SLUG_UNSAFE_RE.sub("_", model).strip("._")[:64] or "model"
        digest = hashlib.sha256(model.encode()).hexdigest()[:12]
        directory = sessions_dir / f"_{safe}-{digest}"
        if not is_within(sessions_dir, directory):
            raise ValueError(f"Session directory for '{model}' is outside {sessions_dir}.")
    return directory


def _validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.fullmatch(session_id) or ".." in session_id:
        raise ValueError(f"Invalid session id {session_id!r}.")
    return session_id


def _session_path(sessions_dir: Path, model: str, session_id: str) -> Path:
    directory = _session_dir(sessions_dir, model)
    path = directory / f"{_validate_session_id(session_id)}.json"
    if not is_within(directory, path):
        raise ValueError(f"Session file {path} is outside {directory}.")
    return path


def save_session(sessions_dir: Path, session: Session) -> None:
    """Write *session* atomically: readers see the old file or the new one, never a mix."""
    path = _session_path(sessions_dir, session.model, session.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    session.updated_at = datetime.now().isoformat(timespec="seconds")
    text = json.dumps(session.to_dict(), indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string")
    return value


def _check_message(message: Any) -> None:
    """Reject a message the chat TUI could not replay (see ``ChatApp._replay_history``)."""
    role = message.get("role") if isinstance(message, dict) else None
    if not isinstance(role, str) or role not in _MESSAGE_ROLES:
        raise ValueError(f"each message needs a 'role' from {sorted(_MESSAGE_ROLES)}")
    if role == "user" and not isinstance(message.get("content"), str):
        raise ValueError("user messages need text 'content'")
    if role == "tool" and not isinstance(message.get("tool_call_id"), str):
        raise ValueError("tool messages need a string 'tool_call_id'")
    _optional_str(message, "content")
    _optional_str(message, "thinking")
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ValueError("'tool_calls' must be a list")
    for call in tool_calls:
        function = call.get("function", {}) if isinstance(call, dict) else None
        if not isinstance(function, dict):
            raise ValueError("each tool call must be an object with an object 'function'")
        _optional_str(call, "id")
        # Replay defaults a missing name/arguments to "" but renders a present one
        # as text, so an explicit null is as unusable as a number here.
        for key in ("name", "arguments"):
            if key in function and not isinstance(function[key], str):
                raise ValueError(f"tool call '{key}' must be a string")


def load_session(path: Path) -> Session:
    """Load one session file; raise ``ValueError`` or ``OSError`` if it is unusable."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    model, session_id, messages = data.get("model"), data.get("id"), data.get("messages")
    if not isinstance(model, str) or not isinstance(session_id, str):
        raise ValueError("'model' and 'id' must be strings")
    _validate_session_id(session_id)
    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list")
    for message in messages:
        _check_message(message)
    return Session(
        model=model,
        session_id=session_id,
        messages=messages,
        system_prompt=_optional_str(data, "system_prompt"),
        created_at=_optional_str(data, "created_at"),
        updated_at=_optional_str(data, "updated_at"),
        endpoint=_optional_str(data, "endpoint"),
    )


def _iter_sessions(
    sessions_dir: Path,
    model: str,
    warn: Warn | None,
    endpoint: str | None = None,
    include_legacy: bool = False,
) -> Iterator[Session]:
    """Yield a model's readable sessions newest first, skipping unusable files."""
    d = _session_dir(sessions_dir, model)
    if not d.exists():
        return
    endpoint = session_endpoint(endpoint) if endpoint is not None else None
    for path in sorted(d.glob("*.json"), reverse=True):
        try:
            session = load_session(path)
            if session.model != model:
                raise ValueError("session model does not match its directory")
            if endpoint is None or session.endpoint == endpoint:
                yield session
            elif include_legacy and session.endpoint is None:
                yield session
        except (OSError, ValueError) as e:
            (warn or logger.warning)(f"Skipping unreadable session file {path}: {e}")


def list_sessions(
    sessions_dir: Path,
    model: str,
    *,
    endpoint: str | None = None,
    include_legacy: bool = False,
    warn: Warn | None = None,
) -> list[Session]:
    """Return a model's sessions, optionally scoped to an endpoint, newest first.

    Unassigned legacy sessions are opt-in when filtering by endpoint. Omitting
    endpoint retains unfiltered access for storage inspection and migration.
    """
    return list(_iter_sessions(sessions_dir, model, warn, endpoint, include_legacy))


def latest_session(
    sessions_dir: Path, model: str, *, endpoint: str | None = None, warn: Warn | None = None
) -> Session | None:
    """Return the newest readable session for *model*, or None; *warn* as in list_sessions."""
    return next(_iter_sessions(sessions_dir, model, warn, endpoint), None)
