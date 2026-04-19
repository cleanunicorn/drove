"""Permission policy scaffold. Phase 1 ships with Policy.trust_mode() as the default."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Tier(StrEnum):
    READ = "read"
    MUTATE = "mutate"
    EXEC = "exec"


class Decision(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"
    DENY = "deny"


_TIER_DEFAULTS: dict[Tier, Decision] = {
    Tier.READ: Decision.AUTO,
    Tier.MUTATE: Decision.PROMPT,
    Tier.EXEC: Decision.PROMPT,
}


@dataclass
class Policy:
    """Per-tool permission decision resolver.

    If ``trust_all`` is True, every tool returns ``Decision.AUTO``
    regardless of overrides or tier.
    """

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
        """All tools auto-approve. Used in Phase 1 until PromptHook lands."""
        return cls(trust_all=True)
