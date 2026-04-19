"""list_dir tool: ls-like directory listing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFAULT_MAX_ENTRIES = 500

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": (
            "List the contents of a directory. Entries are one per line. Directories"
            " are marked with a trailing '/'. Defaults to the working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to the working directory.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, descend into subdirectories. Default false.",
                },
                "max_entries": {
                    "type": "integer",
                    "description": f"Cap on entries returned (default {_DEFAULT_MAX_ENTRIES}).",
                },
            },
        },
    },
}


def _format_entry(root: Path, entry: Path) -> str:
    rel = entry.relative_to(root).as_posix()
    return rel + "/" if entry.is_dir() else rel


def _iter_entries(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in root.rglob("*"))
    return sorted(root.iterdir())


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = args.get("path")
    if raw_path is None:
        path = ctx.cwd
    else:
        if not isinstance(raw_path, str):
            return ToolResult(content="Error: 'path' must be a string", error=True)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = ctx.cwd / path

    if not path.exists():
        return ToolResult(content=f"Error: directory not found: {path}", error=True)
    if not path.is_dir():
        return ToolResult(content=f"Error: not a directory: {path}", error=True)

    recursive_raw = args.get("recursive", False)
    recursive = bool(recursive_raw)

    max_entries_raw = args.get("max_entries", _DEFAULT_MAX_ENTRIES)
    max_entries = (
        max_entries_raw
        if isinstance(max_entries_raw, int) and max_entries_raw > 0
        else _DEFAULT_MAX_ENTRIES
    )

    entries = _iter_entries(path, recursive)
    total = len(entries)
    entries = entries[:max_entries]

    lines = [_format_entry(path, p) for p in entries]
    if total > max_entries:
        lines.append(f"[truncated: showing {max_entries} of {total} entries]")

    return ToolResult(content="\n".join(lines))


register(
    ToolSpec(
        name="list_dir",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
