"""Permission policy + prompt hook types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Tier(StrEnum):
    READ = "read"
    MUTATE = "mutate"
    EXEC = "exec"


class Decision(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"
    DENY = "deny"


PromptDecision = Literal["allow", "session_allow", "deny_continue", "deny_abort"]
PROMPT_DECISIONS: tuple[PromptDecision, ...] = (
    "allow",
    "session_allow",
    "deny_continue",
    "deny_abort",
)

# Signature: (tool_name, tool_args) -> await decision
PromptHook = Callable[[str, dict[str, object]], Awaitable[PromptDecision]]


class AbortTurn(Exception):
    """Raised from ToolRuntime.dispatch when the user picks Deny & Abort.

    Callers (the TUI turn loop) catch this and cancel the whole turn.
    """


_TIER_DEFAULTS: dict[Tier, Decision] = {
    Tier.READ: Decision.AUTO,
    Tier.MUTATE: Decision.PROMPT,
    Tier.EXEC: Decision.PROMPT,
}


@dataclass
class Policy:
    """Per-tool permission decision resolver."""

    overrides: dict[str, Decision] = field(default_factory=dict)
    trust_all: bool = False

    def decide(self, tool_name: str, tier: Tier) -> Decision:
        if self.trust_all:
            return Decision.AUTO
        if tool_name in self.overrides:
            return self.overrides[tool_name]
        return _TIER_DEFAULTS[tier]

    @classmethod
    def trust_mode(cls) -> Policy:
        """All tools auto-approve. Used in tests and when agents.enabled=false."""
        return cls(trust_all=True)

    @classmethod
    def from_config(cls, overrides: Mapping[str, str]) -> Policy:
        """Build a Policy from a config dict of {tool_name: 'auto'|'prompt'|'deny'}."""
        parsed: dict[str, Decision] = {}
        for name, value in overrides.items():
            try:
                parsed[name] = Decision(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid permission decision for {name!r}: {value!r}."
                    f" Expected one of {[d.value for d in Decision]}."
                ) from e
        return cls(overrides=parsed)
