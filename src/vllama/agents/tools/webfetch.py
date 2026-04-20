"""web_fetch tool: HTTP GET + HTML → plain-text extraction."""

from __future__ import annotations

from typing import Any

import httpx

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFAULT_TIMEOUT_S = 10.0
_MAX_CONTENT_BYTES = 15_000_000  # 15 MB

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "HTTP GET a URL and return the main content as readable text."
            " HTML pages are parsed with readability and converted to plain text."
            " Non-HTML content-types are returned as decoded bytes with a note."
            " 10s timeout. 15MB content cap."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute HTTP/HTTPS URL."},
                "prompt": {
                    "type": "string",
                    "description": "Optional hint about what to extract (unused in MVP).",
                },
            },
            "required": ["url"],
        },
    },
}


def _extract_html(html: str) -> str:
    """Extract main text from HTML via readability + html2text. Fails open to raw."""
    try:
        from readability import Document
    except ImportError:
        return html

    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        title = doc.short_title() or ""
    except Exception:  # noqa: BLE001 — readability parses imperfectly; fall back
        summary_html = html
        title = ""

    try:
        import html2text

        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # don't wrap
        h2t.ignore_images = True
        text = h2t.handle(summary_html)
    except Exception:  # noqa: BLE001 — html2text rare failure
        text = summary_html

    text = text.strip()
    if title and title not in text:
        text = f"# {title}\n\n{text}"
    return text


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    url = args.get("url")
    if not isinstance(url, str) or not url:
        return ToolResult(content="Error: 'url' argument is required", error=True)
    if not (url.startswith("http://") or url.startswith("https://")):
        return ToolResult(content=f"Error: url must be http(s): {url}", error=True)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=_DEFAULT_TIMEOUT_S)
    except httpx.TimeoutException:
        return ToolResult(
            content=f"Error: request timed out after {_DEFAULT_TIMEOUT_S}s", error=True
        )
    except httpx.RequestError as e:
        return ToolResult(content=f"Error: connection failure: {e}", error=True)

    body = resp.content
    if len(body) > _MAX_CONTENT_BYTES:
        return ToolResult(
            content=(
                f"Error: response too large ({len(body)} bytes; max {_MAX_CONTENT_BYTES})"
            ),
            error=True,
        )

    ctype = resp.headers.get("content-type", "").lower()
    sniff = body.lstrip()[:20].lower()
    is_html = "html" in ctype or sniff.startswith((b"<!doctype html", b"<html"))

    if resp.is_error:
        snippet = body.decode("utf-8", errors="replace")[:500]
        return ToolResult(
            content=(
                f"Error: HTTP {resp.status_code} from {url}."
                f" Body snippet:\n{snippet}"
            ),
            error=True,
        )

    if is_html:
        text = _extract_html(body.decode("utf-8", errors="replace"))
    else:
        raw = body.decode("utf-8", errors="replace")
        text = f"[non-HTML content-type: {ctype or 'unknown'}]\n{raw}"

    return ToolResult(content=text, meta={"url": url, "status": resp.status_code})


register(
    ToolSpec(
        name="web_fetch",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
