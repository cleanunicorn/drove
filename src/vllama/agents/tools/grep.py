"""grep tool: content search via ripgrep if available, Python fallback otherwise."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shutil
from pathlib import Path
from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFAULT_MAX_MATCHES = 200

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search file contents by regex. Returns matches as 'path:line:content'."
            " Uses ripgrep if installed, otherwise a pure-Python fallback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to match against file contents.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search. Defaults to cwd.",
                },
                "glob": {
                    "type": "string",
                    "description": "Filename glob filter, e.g. '*.py'. Default: no filter.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context before/after each match (default 0).",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Ignore case (default false).",
                },
                "max_matches": {
                    "type": "integer",
                    "description": f"Cap on match lines returned (default {_DEFAULT_MAX_MATCHES}).",
                },
            },
            "required": ["pattern"],
        },
    },
}


def _iter_candidate_files(root: Path, glob: str | None) -> list[Path]:
    if root.is_file():
        if glob is None or fnmatch.fnmatch(root.name, glob):
            return [root]
        return []
    results: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if glob is not None and not fnmatch.fnmatch(p.name, glob):
            continue
        results.append(p)
    return results


def _is_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _python_search(
    pattern: str,
    root: Path,
    glob: str | None,
    case_insensitive: bool,
    context_lines: int,
) -> list[str]:
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return [f"__ERROR__: invalid regex: {e}"]

    lines_out: list[str] = []
    for path in _iter_candidate_files(root, glob):
        try:
            with path.open("rb") as fh:
                sample = fh.read(8192)
            if _is_binary(sample):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        all_lines = text.splitlines()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()

        for idx, line in enumerate(all_lines):
            if regex.search(line):
                lo = max(0, idx - context_lines)
                hi = min(len(all_lines), idx + context_lines + 1)
                for i in range(lo, hi):
                    lines_out.append(f"{rel}:{i + 1}:{all_lines[i]}")
    return lines_out


async def _run_ripgrep(
    pattern: str,
    root: Path,
    glob: str | None,
    case_insensitive: bool,
    context_lines: int,
) -> list[str]:
    args: list[str] = ["rg", "--line-number", "--no-heading", "--color=never"]
    if case_insensitive:
        args.append("-i")
    if context_lines > 0:
        args.extend(["-C", str(context_lines)])
    if glob is not None:
        args.extend(["-g", glob])
    args.extend(["-e", pattern, str(root)])

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode not in (0, 1):  # 1 = no matches, not an error
        return [f"__ERROR__: rg exited {proc.returncode}"]
    out_text = stdout.decode("utf-8", errors="replace")
    # Convert absolute paths to relative (rg outputs absolute because we gave it abs root).
    rel_lines: list[str] = []
    root_prefix = str(root).rstrip("/") + "/"
    for raw in out_text.splitlines():
        if raw.startswith(root_prefix):
            raw = raw[len(root_prefix) :]
        rel_lines.append(raw)
    return rel_lines


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult(content="Error: 'pattern' argument is required", error=True)

    raw_path = args.get("path")
    if isinstance(raw_path, str) and raw_path:
        root = Path(raw_path).expanduser()
        if not root.is_absolute():
            root = ctx.cwd / root
    else:
        root = ctx.cwd

    if not root.exists():
        return ToolResult(content=f"Error: path not found: {root}", error=True)

    glob_raw = args.get("glob")
    glob_filter = glob_raw if isinstance(glob_raw, str) and glob_raw else None

    ci_raw = args.get("case_insensitive", False)
    case_insensitive = bool(ci_raw)

    cl_raw = args.get("context_lines", 0)
    context_lines = cl_raw if isinstance(cl_raw, int) and cl_raw >= 0 else 0

    max_raw = args.get("max_matches", _DEFAULT_MAX_MATCHES)
    max_matches = max_raw if isinstance(max_raw, int) and max_raw > 0 else _DEFAULT_MAX_MATCHES

    if shutil.which("rg"):
        lines = await _run_ripgrep(pattern, root, glob_filter, case_insensitive, context_lines)
    else:
        lines = _python_search(pattern, root, glob_filter, case_insensitive, context_lines)

    if lines and lines[0].startswith("__ERROR__"):
        return ToolResult(content="Error: " + lines[0][len("__ERROR__: ") :], error=True)

    total = len(lines)
    lines = lines[:max_matches]
    if total > max_matches:
        lines.append(f"[truncated: showing {max_matches} of {total} match lines]")

    return ToolResult(content="\n".join(lines))


register(
    ToolSpec(
        name="grep",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
