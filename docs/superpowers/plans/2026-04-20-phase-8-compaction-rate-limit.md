# Phase 8 — History Compaction + Per-Model Rate Limit — Plan

**Goal:** Prevent context blow-up by summarizing older history once it crosses a threshold; honor per-model rate limits with auto-retry on HTTP 429.

**Architecture:**
- `compaction.py` — `async maybe_compact(history, ctx_size, llm_call, config) -> list[dict]`. Estimate tokens via `len(json.dumps(msg))//4`. If above threshold, LLM-summarize `history[1:-keep_tail]` into one system-note message, keep head (original system prompt at index 0) + summary + tail.
- `rate_limit.py` — `RateLimitedClient` wraps `httpx.AsyncClient.post` with per-model token buckets (per-minute and per-hour), configurable `base_delay_ms`, retry on 429 honoring `Retry-After`, exponential fallback if no header.
- Config: `CompactionConfig` and `RateLimitSettings` per-model dict.
- TUI: runs compaction before each main stream call (Task 4 later — for Phase 8 MVP, module + tests suffice; integration can be one-line hookup).

---

## Task 1: Config — compaction + rate_limit

- `CompactionConfig { enabled=True, threshold=0.7, keep_tail_messages=6 }`.
- `RateLimitSettings { base_delay_ms=0, requests_per_minute=None, requests_per_hour=None, max_retries=5, retry_max_backoff_s=60 }`.
- `AgentsConfig.compaction: CompactionConfig`, `rate_limit: dict[str, RateLimitSettings]`.

## Task 2: compaction.py

- `estimate_tokens(messages) -> int` = sum of `len(json.dumps(m))//4`.
- `maybe_compact` — no-op if disabled or below threshold; else LLM call to summarize head; replace with `{"role": "system", "content": "[Earlier conversation summary]\n<summary>"}` at index 1 (preserving index 0 system prompt). Fail-open: on LLM error, return original history unchanged.
- Tests: no-op path, summarize path (mocked llm), fail-open on exception, keep_tail preserved, disabled config.

## Task 3: rate_limit.py

- `TokenBucket` — minute + hour windows. `acquire()` awaits capacity.
- `RateLimitedClient` — `post(url, json, model=...)`. Applies base_delay, calls bucket.acquire, posts, on 429 reads Retry-After, sleeps, retries; exponential fallback on missing header. Raises `RateLimitExceeded` after `max_retries`.
- Tests: bucket enforcement (mocked time), base_delay, 429 with Retry-After, 429 without (expo fallback), per-model isolation, exhaustion raises.

## Task 4: Full verify
