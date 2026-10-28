"""Keep Python logging off the terminal while a Textual app owns it."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from textual.logging import TextualHandler


@contextmanager
def textual_logging() -> Iterator[None]:
    """Route root log records to the running Textual app instead of the terminal.

    The root callback's stderr handler holds the real terminal stream, so records such as httpx's
    "HTTP Request" lines would be drawn over the TUI. For the duration of the block the root
    logger's handlers are swapped for a single TextualHandler (visible in Textual devtools, silent
    when no app is active); the originals are restored, unclosed and in order, afterwards. The
    root level is left alone, so `drove -v` still lets DEBUG records through.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    handler = TextualHandler(stderr=False, stdout=False)
    handler.setFormatter(next((h.formatter for h in saved if h.formatter), None))
    for original in saved:
        root.removeHandler(original)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        for original in saved:
            root.addHandler(original)
