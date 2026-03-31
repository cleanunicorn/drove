"""Runtime statistics for the vllama proxy."""

from __future__ import annotations

import time


class ProxyStats:
    """Counters for proxy traffic and token usage."""

    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.request_count: int = 0
        self.active_requests: int = 0
        self.error_count: int = 0
        self.tokens_prompt: int = 0
        self.tokens_completion: int = 0
        self._completion_speeds: list[tuple[int, float]] = []  # (tokens, duration_secs)
        self._ttft_samples: list[float] = []  # time-to-first-token in seconds

    def request_started(self) -> None:
        self.request_count += 1
        self.active_requests += 1

    def request_finished(self) -> None:
        self.active_requests = max(0, self.active_requests - 1)

    def request_error(self) -> None:
        self.error_count += 1

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.tokens_prompt += prompt
        self.tokens_completion += completion

    def record_completion_speed(self, completion_tokens: int, duration_seconds: float) -> None:
        if completion_tokens > 0 and duration_seconds > 0:
            self._completion_speeds.append((completion_tokens, duration_seconds))

    def record_ttft(self, seconds: float) -> None:
        if seconds >= 0:
            self._ttft_samples.append(seconds)

    @property
    def last_tokens_per_second(self) -> float | None:
        if not self._completion_speeds:
            return None
        tokens, duration = self._completion_speeds[-1]
        return tokens / duration if duration > 0 else None

    @property
    def avg_tokens_per_second(self) -> float | None:
        if not self._completion_speeds:
            return None
        total_tokens = sum(t for t, _ in self._completion_speeds)
        total_duration = sum(d for _, d in self._completion_speeds)
        return total_tokens / total_duration if total_duration > 0 else None

    @property
    def last_ttft(self) -> float | None:
        return self._ttft_samples[-1] if self._ttft_samples else None

    @property
    def avg_ttft(self) -> float | None:
        if not self._ttft_samples:
            return None
        return sum(self._ttft_samples) / len(self._ttft_samples)
