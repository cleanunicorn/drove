"""Chat session persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class Session:
    def __init__(
        self,
        model: str,
        session_id: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.model = model
        self.id = session_id
        self.messages = messages
        self.system_prompt = system_prompt
        now = datetime.now().isoformat(timespec="seconds")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @property
    def title(self) -> str:
        """First user message, truncated, as a human-readable title."""
        for m in self.messages:
            if m["role"] == "user":
                text = m["content"].replace("\n", " ").strip()
                return text[:60] + ("…" if len(text) > 60 else "")
        return "(empty)"

    @property
    def message_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }


def new_session(model: str, system_prompt: str | None = None) -> Session:
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    return Session(
        model=model, session_id=session_id, messages=messages, system_prompt=system_prompt
    )


def _session_dir(sessions_dir: Path, model: str) -> Path:
    return sessions_dir / model


def _session_path(sessions_dir: Path, model: str, session_id: str) -> Path:
    return _session_dir(sessions_dir, model) / f"{session_id}.json"


def save_session(sessions_dir: Path, session: Session) -> None:
    d = _session_dir(sessions_dir, session.model)
    d.mkdir(parents=True, exist_ok=True)
    session.updated_at = datetime.now().isoformat(timespec="seconds")
    _session_path(sessions_dir, session.model, session.id).write_text(
        json.dumps(session.to_dict(), indent=2, ensure_ascii=False)
    )


def load_session(path: Path) -> Session:
    data = json.loads(path.read_text())
    return Session(
        model=data["model"],
        session_id=data["id"],
        messages=data["messages"],
        system_prompt=data.get("system_prompt"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def list_sessions(sessions_dir: Path, model: str) -> list[Session]:
    """Return sessions for a model, newest first."""
    d = _session_dir(sessions_dir, model)
    if not d.exists():
        return []
    return [load_session(p) for p in sorted(d.glob("*.json"), reverse=True)]


def latest_session(sessions_dir: Path, model: str) -> Session | None:
    sessions = list_sessions(sessions_dir, model)
    return sessions[0] if sessions else None
