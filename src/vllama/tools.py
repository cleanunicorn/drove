"""Tool definitions and execution for TUI function calling."""

from __future__ import annotations

import json
from pathlib import Path

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file from disk."
                " For large files, use offset and limit to read a range of lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line to start from (0-based, default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read (default: all).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file on disk. Creates parent directories automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"Error: invalid JSON arguments: {arguments}"

    if name == "read_file":
        return _read_file(args)
    elif name == "write_file":
        return _write_file(args)
    else:
        return f"Error: unknown tool '{name}'"


def _read_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    if not path.exists():
        return f"Error: file not found: {path}"
    if not path.is_file():
        return f"Error: not a file: {path}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Error reading file: {e}"

    lines = text.splitlines(keepends=True)
    offset = args.get("offset", 0)
    limit = args.get("limit")
    if limit is not None:
        lines = lines[offset : offset + limit]
    elif offset:
        lines = lines[offset:]

    return "".join(lines)


def _write_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    content = args.get("content", "")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Error writing file: {e}"
    return f"Successfully wrote {len(content)} bytes to {path}"
