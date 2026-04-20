# Phase 6 — Web Fetch — Implementation Plan

**Goal:** Add `web_fetch` tool for HTTP GET + HTML → plain-text extraction.

**Architecture:** `httpx.AsyncClient` GET with 10s timeout and 15MB content cap. Extract main content via `readability-lxml`, convert to text via `html2text`. On HTML-like content types only; other types returned as raw decoded text with a note.

**Tech Stack:** Python 3.14, httpx (existing), `readability-lxml` + `html2text` (new deps).

---

## Task 1: Deps + web_fetch + tests

**Files:**
- Modify: `pyproject.toml` (add `readability-lxml`, `html2text`)
- Create: `src/vllama/agents/tools/webfetch.py`
- Modify: `src/vllama/agents/tools/__init__.py` (side-effect import)
- Modify: `tests/test_agents_base.py` (smoke tests expect 11 tools)
- Create: `tests/test_tools_webfetch.py` (httpx.MockTransport-based)

Tests cover: success (HTML → text), non-HTML content (returns as-is), HTTP 4xx/5xx errors, size cap, timeout, missing url arg, invalid URL.
