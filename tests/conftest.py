from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest


@pytest.fixture
def aiter() -> Callable[[list[bytes]], AsyncIterator[bytes]]:
    """Helper to create an async iterator from a list of items."""

    async def _aiter(items: list[bytes]) -> AsyncIterator[bytes]:
        for item in items:
            yield item

    return _aiter
