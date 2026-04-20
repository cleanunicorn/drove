"""Base types and registry for the agent tool system."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vllama.agents.bash_procs import BgProcs
    from vllama.agents.subagent import SubagentRunner

TierValue = Literal["read", "mutate", "exec"]


@dataclass
class ToolResult:
    """Result of a tool invocation, as passed back to the model."""

    content: str
    error: bool = False
    truncated: bool = False
    meta: dict[str, Any] | None = None


@dataclass(kw_only=True)
class ToolContext:
    """Runtime context passed to every tool handler."""

    cwd: Path
    cap_bytes: int
    cap_bytes_bash: int
    bg_procs: BgProcs | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    subagent_runner: SubagentRunner | None = None
    depth: int = 0


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    """Tool metadata + handler."""

    name: str
    definition: dict[str, Any]
    tier: TierValue
    handler: ToolHandler


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    """Register a tool spec. Replaces any existing spec with the same name."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def all_specs() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """Test-only: drop all registrations."""
    _REGISTRY.clear()
