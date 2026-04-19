"""write_file tool: create/overwrite a file, creating parent dirs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write content to a file on disk, overwriting if it exists. Creates parent"
            " directories automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path or path relative to the working directory.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
}


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ToolResult(content="Error: 'path' argument is required", error=True)

    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult(content="Error: 'content' argument is required (string)", error=True)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ctx.cwd / path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return ToolResult(content=f"Error writing file: {e}", error=True)

    return ToolResult(content=f"Wrote {len(content.encode('utf-8'))} bytes to {path}")


register(
    ToolSpec(
        name="write_file",
        definition=_DEFINITION,
        tier="mutate",
        handler=_handler,
    )
)
