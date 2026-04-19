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


def test_from_config_builds_policy() -> None:
    from vllama.agents.permissions import Decision, Policy, Tier

    p = Policy.from_config({"write_file": "auto", "bash": "deny"})
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO
    assert p.decide("bash", Tier.EXEC) is Decision.DENY
    # Unmentioned tool falls back to tier default.
    assert p.decide("read_file", Tier.READ) is Decision.AUTO


def test_from_config_invalid_value_raises() -> None:
    import pytest

    from vllama.agents.permissions import Policy

    with pytest.raises(ValueError):
        Policy.from_config({"bash": "bogus"})


def test_abort_turn_is_exception() -> None:
    from vllama.agents.permissions import AbortTurn

    e = AbortTurn()
    assert isinstance(e, Exception)


def test_prompt_decision_literal_values() -> None:
    """PromptDecision values are well-defined and re-exported."""
    from vllama.agents.permissions import PROMPT_DECISIONS

    assert set(PROMPT_DECISIONS) == {
        "allow",
        "session_allow",
        "deny_continue",
        "deny_abort",
    }
