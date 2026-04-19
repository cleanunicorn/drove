"""apply_patch tool: apply a unified-diff patch to a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unidiff import PatchSet

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Apply a unified-diff patch to a file. The diff is validated hunk-by-hunk"
            " against the current file content before writing. On any context mismatch"
            " the file is left unchanged and an error is returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file the diff targets.",
                },
                "diff": {
                    "type": "string",
                    "description": (
                        "Unified-diff text. File headers optional; only hunks are used."
                    ),
                },
            },
            "required": ["path", "diff"],
        },
    },
}


def _apply_hunks(old_text: str, patchset: PatchSet) -> tuple[str | None, str]:
    """Apply hunks to old_text. Returns (new_text, error_msg).

    Comparison is done without trailing newlines (unidiff strips them when the
    input diff used lineterm="").
    """
    if not patchset:
        return None, "diff contained no patchable files"

    patched_file = patchset[0]
    if not patched_file:
        return None, "diff contained no hunks"

    old_lines = old_text.splitlines()
    ends_with_newline = old_text.endswith("\n")

    new_lines: list[str] = []
    cursor = 0

    for hunk_idx, hunk in enumerate(patched_file, start=1):
        # source_start is 1-based line number in the *old* file.
        source_start = hunk.source_start - 1
        if source_start < cursor:
            return None, f"hunk {hunk_idx}: overlapping or out-of-order context"
        new_lines.extend(old_lines[cursor:source_start])
        cursor = source_start

        for line in hunk:
            val = line.value.rstrip("\n")
            if line.is_context or line.is_removed:
                if cursor >= len(old_lines):
                    return None, (f"hunk {hunk_idx}: ran off end of file at line {cursor + 1}")
                actual = old_lines[cursor]
                if val != actual:
                    return None, (
                        f"hunk {hunk_idx}: context mismatch at line {cursor + 1}."
                        f" Expected: {val!r}, got: {actual!r}"
                    )
                if line.is_context:
                    new_lines.append(actual)
                cursor += 1
            elif line.is_added:
                new_lines.append(val)
            else:
                return None, f"hunk {hunk_idx}: unknown diff line type"

    new_lines.extend(old_lines[cursor:])
    new_text = "\n".join(new_lines)
    if ends_with_newline:
        new_text += "\n"
    return new_text, ""


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ToolResult(content="Error: 'path' argument is required", error=True)
    diff_text = args.get("diff")
    if not isinstance(diff_text, str) or not diff_text:
        return ToolResult(content="Error: 'diff' argument is required", error=True)

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ctx.cwd / path

    if not path.exists():
        return ToolResult(content=f"Error: file not found: {path}", error=True)
    if not path.is_file():
        return ToolResult(content=f"Error: not a file: {path}", error=True)

    try:
        patchset = PatchSet(diff_text)
    except Exception as e:  # noqa: BLE001 — unidiff raises several subclasses
        return ToolResult(content=f"Error: invalid diff: {e}", error=True)

    try:
        original = path.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(content=f"Error reading file: {e}", error=True)

    new_text, err = _apply_hunks(original, patchset)
    if new_text is None:
        return ToolResult(
            content=f"Error: patch failed — {err}. Re-read the file and rebuild the diff.",
            error=True,
        )

    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return ToolResult(content=f"Error writing file: {e}", error=True)

    hunk_count = sum(1 for _ in patchset[0])
    return ToolResult(content=f"Applied {hunk_count} hunk(s) to {path}")


register(
    ToolSpec(
        name="apply_patch",
        definition=_DEFINITION,
        tier="mutate",
        handler=_handler,
    )
)
