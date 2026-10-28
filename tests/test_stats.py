"""Tests for ProxyStats sample bounds and averages."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from drove.stats import ProxyStats


def test_samples_stay_bounded_and_averages_match_lifetime() -> None:
    stats = ProxyStats()
    tokens = [(i % 97) + 1 for i in range(10_000)]
    durations = [((i % 13) + 1) * 0.1 for i in range(10_000)]
    ttfts = [((i % 7) + 1) * 0.05 for i in range(10_000)]

    for t, d, f in zip(tokens, durations, ttfts, strict=True):
        stats.record_completion_speed(t, d)
        stats.record_ttft(f)

    # Memory stays bounded: no per-request sample collection is retained.
    collections = [v for v in vars(stats).values() if isinstance(v, Collection)]
    assert all(len(v) <= 2 for v in collections if not isinstance(v, str))
    assert stats.avg_tokens_per_second == pytest.approx(sum(tokens) / sum(durations))
    assert stats.avg_ttft == pytest.approx(sum(ttfts) / len(ttfts))
    assert stats.last_tokens_per_second == pytest.approx(tokens[-1] / durations[-1])
    assert stats.last_ttft == pytest.approx(ttfts[-1])


def test_empty_and_invalid_samples() -> None:
    stats = ProxyStats()
    assert stats.avg_tokens_per_second is None
    assert stats.last_tokens_per_second is None
    assert stats.avg_ttft is None
    assert stats.last_ttft is None

    stats.record_completion_speed(0, 1.0)
    stats.record_completion_speed(5, 0.0)
    stats.record_ttft(-1.0)

    assert stats.avg_tokens_per_second is None
    assert stats.avg_ttft is None
