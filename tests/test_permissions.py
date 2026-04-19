"""Tests for the permissions scaffold."""

from __future__ import annotations

from vllama.agents.permissions import Decision, Policy, Tier


def test_tier_defaults_read_auto() -> None:
    p = Policy(overrides={})
    assert p.decide("read_file", Tier.READ) is Decision.AUTO


def test_tier_defaults_mutate_prompt() -> None:
    p = Policy(overrides={})
    assert p.decide("write_file", Tier.MUTATE) is Decision.PROMPT


def test_tier_defaults_exec_prompt() -> None:
    p = Policy(overrides={})
    assert p.decide("bash", Tier.EXEC) is Decision.PROMPT


def test_override_beats_tier_default() -> None:
    p = Policy(overrides={"write_file": Decision.AUTO})
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO


def test_trust_mode() -> None:
    p = Policy.trust_mode()
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO
    assert p.decide("bash", Tier.EXEC) is Decision.AUTO
    assert p.decide("read_file", Tier.READ) is Decision.AUTO
