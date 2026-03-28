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
