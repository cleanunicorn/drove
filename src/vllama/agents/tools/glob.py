"""glob_files tool: pattern match against the working directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_CAP = 1000

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "glob_files",
        "description": (
            "Find files by glob pattern relative to the working directory."
            " Use '**' for recursive match. Results sorted by mtime descending."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. 'src/**/*.py'.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional override for the search root.",
                },
            },
            "required": ["pattern"],
        },
    },
}


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult(content="Error: 'pattern' argument is required", error=True)

    root_raw = args.get("cwd")
    if isinstance(root_raw, str) and root_raw:
        root = Path(root_raw).expanduser()
        if not root.is_absolute():
            root = ctx.cwd / root
    else:
        root = ctx.cwd

    if not root.exists() or not root.is_dir():
        return ToolResult(content=f"Error: search root not a directory: {root}", error=True)

    try:
        matches = [p for p in root.glob(pattern) if p.is_file()]
    except OSError as e:
        return ToolResult(content=f"Error during glob: {e}", error=True)

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    total = len(matches)
    matches = matches[:_CAP]

    lines = [p.relative_to(root).as_posix() for p in matches]
    if total > _CAP:
        lines.append(f"[truncated: showing {_CAP} of {total} matches]")

    return ToolResult(content="\n".join(lines))


register(
    ToolSpec(
        name="glob_files",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
