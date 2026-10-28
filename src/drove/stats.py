"""Runtime statistics for the drove proxy."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

# Recent samples kept for the "last" values; averages use lifetime running sums,
# so memory and /status cost stay constant however many requests are served.
MAX_SAMPLES = 1024


@dataclass
class AudioStats:
    """Completed audio requests and their audio and processing durations."""

    requests: int = 0
    audio_seconds: float = 0.0
    processing_seconds: float = 0.0


class ProxyStats:
    """Counters for proxy traffic and token usage."""

    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.request_count: int = 0
        self.active_requests: int = 0
        self.error_count: int = 0
        self.transcriptions = AudioStats()
        self.speech = AudioStats()
        self.tokens_prompt: int = 0
        self.tokens_completion: int = 0
        # (tokens, duration_secs)
        self._completion_speeds: deque[tuple[int, float]] = deque(maxlen=MAX_SAMPLES)
        self._ttft_samples: deque[float] = deque(maxlen=MAX_SAMPLES)  # time-to-first-token (s)
        self._speed_tokens: int = 0
        self._speed_seconds: float = 0.0
        self._ttft_sum: float = 0.0
        self._ttft_count: int = 0

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
            self._speed_tokens += completion_tokens
            self._speed_seconds += duration_seconds

    def record_ttft(self, seconds: float) -> None:
        if seconds >= 0:
            self._ttft_samples.append(seconds)
            self._ttft_sum += seconds
            self._ttft_count += 1

    @property
    def last_tokens_per_second(self) -> float | None:
        if not self._completion_speeds:
            return None
        tokens, duration = self._completion_speeds[-1]
        return tokens / duration if duration > 0 else None

    @property
    def avg_tokens_per_second(self) -> float | None:
        if self._speed_seconds <= 0:
            return None
        return self._speed_tokens / self._speed_seconds

    @property
    def last_ttft(self) -> float | None:
        return self._ttft_samples[-1] if self._ttft_samples else None

    @property
    def avg_ttft(self) -> float | None:
        if not self._ttft_count:
            return None
        return self._ttft_sum / self._ttft_count
