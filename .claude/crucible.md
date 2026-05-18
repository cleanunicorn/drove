## 2025-03-24 - Avoid AsyncMock for aiter_raw

**Pattern:** Using `AsyncMock(return_value=aiter([chunks]))` for `httpx.Response.aiter_raw` causes `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited` because `aiter_raw` is expected to return an async generator directly, not a coroutine.

**Fix:** Use a lambda that returns the async generator directly: `fake_response.aiter_raw = lambda: aiter([chunks])`.

**Lesson:** In `drove`, where proxying streams is central, mocking `aiter_raw` correctly is crucial for clean test output and avoiding false positives in flake detection.
