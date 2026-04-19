"""read_file tool: read text files, paged by line range."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a text file. For large files, use offset and limit to"
            " read a specific line range (0-based). Binary files are rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path or path relative to the working directory.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line to start from (0-based, default 0).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of lines to read.",
                },
            },
            "required": ["path"],
        },
    },
}


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ToolResult(content="Error: 'path' argument is required", error=True)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ctx.cwd / path

    if not path.exists():
        return ToolResult(content=f"Error: file not found: {path}", error=True)
    if not path.is_file():
        return ToolResult(content=f"Error: not a file: {path}", error=True)

    try:
        with path.open("rb") as fh:
            sample = fh.read(8192)
        if _is_binary(sample):
            return ToolResult(content=f"Error: file appears to be binary: {path}", error=True)
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(content=f"Error reading file: {e}", error=True)

    lines = text.splitlines(keepends=True)
    offset_raw = args.get("offset", 0)
    limit_raw = args.get("limit")

    offset = offset_raw if isinstance(offset_raw, int) and offset_raw >= 0 else 0
    limit = limit_raw if isinstance(limit_raw, int) and limit_raw >= 0 else None

    if limit is not None:
        lines = lines[offset : offset + limit]
    elif offset:
        lines = lines[offset:]

    return ToolResult(content="".join(lines))


register(
    ToolSpec(
        name="read_file",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
