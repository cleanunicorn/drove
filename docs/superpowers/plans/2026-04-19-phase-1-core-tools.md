# Phase 1 — Core File/Code Tools + Runtime Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing 2-tool `src/vllama/tools.py` with a new `src/vllama/agents/` package that ships 6 file/code tools (`read_file`, `write_file`, `apply_patch`, `list_dir`, `glob_files`, `grep`) behind a clean registry + async runtime with output capping. No permission prompts yet (trust mode default).

**Architecture:** Tool-per-file under `src/vllama/agents/tools/`. Each module registers a `ToolSpec` (definition + async handler) at import time. `ToolRuntime.dispatch` looks up, parses args, runs handler, applies output cap. A minimal `Policy` scaffold exists but always returns `AUTO` in this phase. TUI (`tui.py`) imports the new registry and calls `await runtime.dispatch(name, arguments_json)` in the existing tool-call loop. Legacy `tools.py` deleted at the end.

**Tech Stack:** Python 3.14, async/await, `pathlib`, stdlib `re`, `subprocess`, `shutil.which` for ripgrep detection, `unidiff` (new dep) for patch parsing, `pytest` + `pytest-asyncio` (existing, `asyncio_mode=auto`). mypy strict, ruff line-length 100.

---

## File Structure

```
src/vllama/agents/
    __init__.py                 # empty
    tools/
        __init__.py             # imports all tool modules so registry populates
        _base.py                # ToolResult, ToolSpec, ToolContext, register, all_specs, get_spec
        read.py                 # read_file
        write.py                # write_file
        edit.py                 # apply_patch
        list.py                 # list_dir
        glob.py                 # glob_files
        grep.py                 # grep (Python + ripgrep fallback)
    permissions.py              # Policy scaffold (all-auto in Phase 1)
    runtime.py                  # ToolRuntime.dispatch

tests/
    test_agents_base.py         # registry + types
    test_tools_read.py
    test_tools_write.py
    test_tools_edit.py
    test_tools_list.py
    test_tools_glob.py
    test_tools_grep.py
    test_permissions.py
    test_runtime.py
```

`src/vllama/tools.py` deleted in Task 12. `src/vllama/tui.py` modified in Task 11 (import swap + await).

---

## Task 1: Package scaffolding + base types

**Files:**
- Create: `src/vllama/agents/__init__.py`
- Create: `src/vllama/agents/tools/__init__.py`
- Create: `src/vllama/agents/tools/_base.py`
- Test: `tests/test_agents_base.py`

- [ ] **Step 1: Create empty package inits**

Create both files, each empty (zero bytes):

- `src/vllama/agents/__init__.py`
- `src/vllama/agents/tools/__init__.py`

The `tools/__init__.py` gets its real contents (the registration-side-effect imports) in Task 8, once all six tool modules exist. Keeping it empty now means individual tool modules can be imported one at a time as tests are added in Tasks 2–7 without the package-level import breaking on a missing sibling.

Commands:
```bash
mkdir -p src/vllama/agents/tools
: > src/vllama/agents/__init__.py
: > src/vllama/agents/tools/__init__.py
```

- [ ] **Step 2: Write the failing test for ToolResult + registry**

Create `tests/test_agents_base.py`:

```python
"""Tests for the tool registry and base types."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    all_specs,
    clear_registry,
    get_spec,
    register,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    clear_registry()


def test_tool_result_defaults() -> None:
    r = ToolResult(content="hello")
    assert r.content == "hello"
    assert r.error is False
    assert r.truncated is False
    assert r.meta is None


async def _noop_handler(args: dict, ctx: ToolContext) -> ToolResult:
    return ToolResult(content="")


def test_register_and_get_spec() -> None:
    spec = ToolSpec(
        name="demo",
        definition={"type": "function", "function": {"name": "demo"}},
        tier="read",
        handler=_noop_handler,
    )
    register(spec)
    assert get_spec("demo") is spec
    assert spec in all_specs()


def test_get_spec_missing_returns_none() -> None:
    assert get_spec("missing") is None


def test_register_replaces_same_name() -> None:
    s1 = ToolSpec(name="demo", definition={}, tier="read", handler=_noop_handler)
    s2 = ToolSpec(name="demo", definition={}, tier="mutate", handler=_noop_handler)
    register(s1)
    register(s2)
    assert get_spec("demo") is s2
    assert len(all_specs()) == 1


def test_tool_context_shape(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
    assert ctx.cwd == tmp_path
    assert ctx.cap_bytes == 8192
    assert ctx.cap_bytes_bash == 32768
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllama.agents'` (or the tools._base submodule).

- [ ] **Step 4: Create `_base.py`**

Create `src/vllama/agents/tools/_base.py`:

```python
"""Base types and registry for the agent tool system."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TierValue = Literal["read", "mutate", "exec"]


@dataclass
class ToolResult:
    """Result of a tool invocation, as passed back to the model."""

    content: str
    error: bool = False
    truncated: bool = False
    meta: dict[str, Any] | None = None


@dataclass
class ToolContext:
    """Runtime context passed to every tool handler."""

    cwd: Path
    cap_bytes: int
    cap_bytes_bash: int


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    """Tool metadata + handler."""

    name: str
    definition: dict[str, Any]
    tier: TierValue
    handler: ToolHandler


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    """Register a tool spec. Replaces any existing spec with the same name."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def all_specs() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def clear_registry() -> None:
    """Test-only: drop all registrations."""
    _REGISTRY.clear()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Type-check and lint**

Run: `uv run mypy src/vllama/agents/` — expect no errors.
Run: `uv run ruff check src/vllama/agents/ tests/test_agents_base.py` — expect no errors.

- [ ] **Step 7: Commit**

```bash
git add src/vllama/agents/__init__.py src/vllama/agents/tools/_base.py tests/test_agents_base.py
git commit -m "feat(agents): add base types and tool registry for Phase 1 scaffolding"
```

---

## Task 2: `read_file` tool

**Files:**
- Create: `src/vllama/agents/tools/read.py`
- Test: `tests/test_tools_read.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools_read.py`:

```python
"""Tests for the read_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    # Import triggers registration.
    import vllama.agents.tools.read  # noqa: F401


async def test_read_file_full(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p)}, ctx)
    assert result.error is False
    assert result.content == "hello\nworld\n"


async def test_read_file_offset_limit(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "offset": 1, "limit": 2}, ctx)
    assert result.content == "b\nc\n"


async def test_read_file_offset_only(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "x.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "offset": 1}, ctx)
    assert result.content == "b\nc\n"


async def test_read_file_not_found(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "nope.txt")}, ctx)
    assert result.error is True
    assert "not found" in result.content.lower()


async def test_read_file_not_a_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path)}, ctx)
    assert result.error is True
    assert "not a file" in result.content.lower()


async def test_read_file_binary_rejected(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "bin"
    p.write_bytes(b"\x00\x01\x02\x03" * 16)
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": str(p)}, ctx)
    assert result.error is True
    assert "binary" in result.content.lower()


async def test_read_file_relative_to_cwd(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "rel.txt").write_text("content", encoding="utf-8")
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({"path": "rel.txt"}, ctx)
    assert result.error is False
    assert result.content == "content"


async def test_read_file_missing_path_arg(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("read_file")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "path" in result.content.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_read.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllama.agents.tools.read'`.

- [ ] **Step 3: Implement `read_file`**

Create `src/vllama/agents/tools/read.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools_read.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Lint and type-check**

Run: `uv run mypy src/vllama/agents/tools/read.py`
Run: `uv run ruff check src/vllama/agents/tools/read.py tests/test_tools_read.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/read.py tests/test_tools_read.py
git commit -m "feat(agents): add read_file tool with paging and binary detection"
```

---

## Task 3: `write_file` tool

**Files:**
- Create: `src/vllama/agents/tools/write.py`
- Test: `tests/test_tools_write.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools_write.py`:

```python
"""Tests for the write_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.write  # noqa: F401


async def test_write_new_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.txt"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "hello"}, ctx)
    assert result.error is False
    assert p.read_text(encoding="utf-8") == "hello"


async def test_write_overwrite(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.txt"
    p.write_text("old", encoding="utf-8")
    spec = get_spec("write_file")
    assert spec is not None
    await spec.handler({"path": str(p), "content": "new"}, ctx)
    assert p.read_text(encoding="utf-8") == "new"


async def test_write_creates_parents(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "nested" / "deep" / "a.txt"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "hi"}, ctx)
    assert result.error is False
    assert p.read_text(encoding="utf-8") == "hi"


async def test_write_relative(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    await spec.handler({"path": "a.txt", "content": "rel"}, ctx)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "rel"


async def test_write_missing_path(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"content": "x"}, ctx)
    assert result.error is True
    assert "path" in result.content.lower()


async def test_write_missing_content(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "a")}, ctx)
    assert result.error is True
    assert "content" in result.content.lower()


async def test_write_reports_bytes(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a"
    spec = get_spec("write_file")
    assert spec is not None
    result = await spec.handler({"path": str(p), "content": "12345"}, ctx)
    assert "5" in result.content
    assert str(p) in result.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_write.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllama.agents.tools.write'`.

- [ ] **Step 3: Implement `write_file`**

Create `src/vllama/agents/tools/write.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_write.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/tools/write.py`
Run: `uv run ruff check src/vllama/agents/tools/write.py tests/test_tools_write.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/write.py tests/test_tools_write.py
git commit -m "feat(agents): add write_file tool with parent-dir creation"
```

---

## Task 4: `list_dir` tool

**Files:**
- Create: `src/vllama/agents/tools/list.py`
- Test: `tests/test_tools_list.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools_list.py`:

```python
"""Tests for the list_dir tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.list  # noqa: F401


async def test_list_flat(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path)}, ctx)
    assert result.error is False
    lines = result.content.strip().split("\n")
    assert "a.txt" in lines
    assert "sub/" in lines


async def test_list_recursive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler(
        {"path": str(tmp_path), "recursive": True}, ctx
    )
    assert "a.txt" in result.content
    assert "sub/" in result.content
    assert "sub/b.txt" in result.content


async def test_list_max_entries(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler(
        {"path": str(tmp_path), "max_entries": 5}, ctx
    )
    lines = [ln for ln in result.content.split("\n") if ln.strip()]
    # includes truncation marker line as a non-path line
    file_lines = [ln for ln in lines if ln.endswith(".txt")]
    assert len(file_lines) == 5
    assert "truncated" in result.content.lower()


async def test_list_path_not_found(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(tmp_path / "nope")}, ctx)
    assert result.error is True


async def test_list_not_a_dir(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({"path": str(f)}, ctx)
    assert result.error is True


async def test_list_default_cwd(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    spec = get_spec("list_dir")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert "a.txt" in result.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_list.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `list_dir`**

Create `src/vllama/agents/tools/list.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_list.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/tools/list.py`
Run: `uv run ruff check src/vllama/agents/tools/list.py tests/test_tools_list.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/list.py tests/test_tools_list.py
git commit -m "feat(agents): add list_dir tool with optional recursion and entry cap"
```

---

## Task 5: `glob_files` tool

**Files:**
- Create: `src/vllama/agents/tools/glob.py`
- Test: `tests/test_tools_glob.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tools_glob.py`:

```python
"""Tests for the glob_files tool."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.glob  # noqa: F401


async def test_glob_flat(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    assert result.error is False
    assert "a.py" in result.content
    assert "b.txt" not in result.content


async def test_glob_recursive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "top.py").write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "**/*.py"}, ctx)
    assert "sub/a.py" in result.content.replace("\\", "/")
    assert "top.py" in result.content


async def test_glob_sorts_by_mtime_desc(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    old = tmp_path / "old.py"
    old.write_text("x", encoding="utf-8")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    new = tmp_path / "new.py"
    new.write_text("y", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    lines = [ln for ln in result.content.split("\n") if ln.strip()]
    assert lines[0].endswith("new.py")
    assert lines[1].endswith("old.py")


async def test_glob_cap(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    for i in range(1100):
        (tmp_path / f"f{i:04d}.py").write_text("x", encoding="utf-8")
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.py"}, ctx)
    lines = [ln for ln in result.content.split("\n") if ln.strip() and ln.endswith(".py")]
    assert len(lines) == 1000
    assert "truncated" in result.content.lower()


async def test_glob_missing_pattern(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "pattern" in result.content.lower()


async def test_glob_no_matches(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("glob_files")
    assert spec is not None
    result = await spec.handler({"pattern": "*.nothere"}, ctx)
    assert result.error is False
    assert result.content == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_glob.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `glob_files`**

Create `src/vllama/agents/tools/glob.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_glob.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/tools/glob.py`
Run: `uv run ruff check src/vllama/agents/tools/glob.py tests/test_tools_glob.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/glob.py tests/test_tools_glob.py
git commit -m "feat(agents): add glob_files tool with mtime sort and 1000-match cap"
```

---

## Task 6: `grep` tool

Two-part task: implement Python fallback first (testable without `rg`), then add ripgrep path behind a detection check.

**Files:**
- Create: `src/vllama/agents/tools/grep.py`
- Test: `tests/test_tools_grep.py`

- [ ] **Step 1: Write the failing tests (all scenarios, Python-path assumed)**

Create `tests/test_tools_grep.py`:

```python
"""Tests for the grep tool."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture(autouse=True)
def _force_python_impl(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default test config: pretend rg is not installed so we hit the Python path."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    yield


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.grep  # noqa: F401


async def test_grep_basic_match(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("hello\nworld\nhello again\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello"}, ctx)
    assert result.error is False
    assert "a.py:1:hello" in result.content
    assert "a.py:3:hello again" in result.content


async def test_grep_regex(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("foo1\nbar\nfoo42\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": r"foo\d+"}, ctx)
    assert "foo1" in result.content
    assert "foo42" in result.content
    assert "bar" not in result.content


async def test_grep_case_insensitive(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("Hello\nHELLO\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello", "case_insensitive": True}, ctx)
    assert "Hello" in result.content
    assert "HELLO" in result.content


async def test_grep_glob_filter(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("match\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("match\n", encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "match", "glob": "*.py"}, ctx)
    assert "a.py" in result.content
    assert "b.txt" not in result.content


async def test_grep_context_lines(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text(
        "line1\nline2\ntarget\nline4\nline5\n", encoding="utf-8"
    )
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler(
        {"pattern": "target", "context_lines": 1}, ctx
    )
    assert "line2" in result.content
    assert "target" in result.content
    assert "line4" in result.content
    assert "line1" not in result.content
    assert "line5" not in result.content


async def test_grep_max_matches(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    (tmp_path / "a.py").write_text("m\n" * 500, encoding="utf-8")
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "m", "max_matches": 10}, ctx)
    match_lines = [ln for ln in result.content.split("\n") if "a.py:" in ln]
    assert len(match_lines) == 10
    assert "truncated" in result.content.lower()


async def test_grep_missing_pattern(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({}, ctx)
    assert result.error is True
    assert "pattern" in result.content.lower()


async def test_grep_uses_rg_when_present(
    tmp_path: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ripgrep is on PATH, grep shells out to it. We assert the code path ran."""
    _load()
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")

    import vllama.agents.tools.grep as grep_mod

    called = {"rg": False}

    async def fake_rg(
        pattern: str,
        root: Path,
        glob: str | None,
        case_insensitive: bool,
        context_lines: int,
    ) -> list[str]:
        called["rg"] = True
        return [f"{root}/a.py:1:hello"]

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/rg" if name == "rg" else None)
    monkeypatch.setattr(grep_mod, "_run_ripgrep", fake_rg)

    spec = get_spec("grep")
    assert spec is not None
    result = await spec.handler({"pattern": "hello"}, ctx)
    assert called["rg"] is True
    assert "hello" in result.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_grep.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `grep` (Python path + ripgrep shell-out)**

Create `src/vllama/agents/tools/grep.py`:

```python
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
            raw = raw[len(root_prefix):]
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
        return ToolResult(content="Error: " + lines[0][len("__ERROR__: "):], error=True)

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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools_grep.py -v`
Expected: all 8 tests pass. The `test_grep_uses_rg_when_present` test forces `rg` on PATH and verifies `_run_ripgrep` is called.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/tools/grep.py`
Run: `uv run ruff check src/vllama/agents/tools/grep.py tests/test_tools_grep.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/grep.py tests/test_tools_grep.py
git commit -m "feat(agents): add grep tool with ripgrep detection and Python fallback"
```

---

## Task 7: `apply_patch` tool

**Files:**
- Modify: `pyproject.toml` (add `unidiff` dep)
- Create: `src/vllama/agents/tools/edit.py`
- Test: `tests/test_tools_edit.py`

- [ ] **Step 1: Add `unidiff` dependency**

Run:
```bash
uv add unidiff
```

Verify `pyproject.toml` now lists `unidiff>=0.7` (or similar) under `[project.dependencies]`. Run `uv sync` to install it.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tools_edit.py`:

```python
"""Tests for the apply_patch tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry, get_spec


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)


def _load() -> None:
    import vllama.agents.tools.edit  # noqa: F401


def _make_diff(old: str, new: str, path: str) -> str:
    """Build a unified diff from two full-file strings (without trailing-newline concerns)."""
    import difflib

    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff) + "\n"


async def test_apply_patch_success(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    diff = _make_diff("x = 1\ny = 2\nz = 3\n", "x = 1\ny = 20\nz = 3\n", "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is False, result.content
    assert p.read_text(encoding="utf-8") == "x = 1\ny = 20\nz = 3\n"


async def test_apply_patch_context_mismatch(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("actual\ncontent\n", encoding="utf-8")
    diff = _make_diff("different\ncontent\n", "different\nnew\n", "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is True
    assert "hunk" in result.content.lower() or "mismatch" in result.content.lower()
    # File left untouched.
    assert p.read_text(encoding="utf-8") == "actual\ncontent\n"


async def test_apply_patch_missing_file(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    spec = get_spec("apply_patch")
    assert spec is not None
    diff = _make_diff("a\n", "b\n", "missing.py")
    result = await spec.handler(
        {"path": str(tmp_path / "missing.py"), "diff": diff}, ctx
    )
    assert result.error is True
    assert "not found" in result.content.lower()


async def test_apply_patch_missing_args(ctx: ToolContext) -> None:
    _load()
    spec = get_spec("apply_patch")
    assert spec is not None
    r1 = await spec.handler({"diff": "x"}, ctx)
    r2 = await spec.handler({"path": "/tmp/x"}, ctx)
    assert r1.error is True
    assert r2.error is True


async def test_apply_patch_malformed_diff(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    p.write_text("x\n", encoding="utf-8")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": "not a diff"}, ctx)
    assert result.error is True


async def test_apply_patch_multi_hunk(tmp_path: Path, ctx: ToolContext) -> None:
    _load()
    p = tmp_path / "a.py"
    original = "a\nb\nc\nd\ne\nf\ng\nh\n"
    modified = "a\nB\nc\nd\ne\nf\nG\nh\n"
    p.write_text(original, encoding="utf-8")
    diff = _make_diff(original, modified, "a.py")
    spec = get_spec("apply_patch")
    assert spec is not None
    result = await spec.handler({"path": str(p), "diff": diff}, ctx)
    assert result.error is False, result.content
    assert p.read_text(encoding="utf-8") == modified
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools_edit.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement `apply_patch`**

Create `src/vllama/agents/tools/edit.py`:

```python
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
                    return None, (
                        f"hunk {hunk_idx}: ran off end of file at line {cursor + 1}"
                    )
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tools_edit.py -v`
Expected: all 6 tests pass.

- [ ] **Step 6: Lint + mypy**

Run: `uv run mypy src/vllama/agents/tools/edit.py`
Run: `uv run ruff check src/vllama/agents/tools/edit.py tests/test_tools_edit.py`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/vllama/agents/tools/edit.py tests/test_tools_edit.py
git commit -m "feat(agents): add apply_patch tool (unified-diff) with hunk-level validation"
```

---

## Task 8: Tool package index

**Files:**
- Create: `src/vllama/agents/tools/__init__.py`

Now that all tool modules exist, create the package index that imports all of them to populate the registry.

- [ ] **Step 1: Create the index**

Create `src/vllama/agents/tools/__init__.py`:

```python
"""Tool registry. Importing this package populates all tool specs."""

# Import for registration side-effects.
from vllama.agents.tools import edit as _edit  # noqa: F401
from vllama.agents.tools import glob as _glob  # noqa: F401
from vllama.agents.tools import grep as _grep  # noqa: F401
from vllama.agents.tools import list as _list  # noqa: F401
from vllama.agents.tools import read as _read  # noqa: F401
from vllama.agents.tools import write as _write  # noqa: F401

from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    all_specs,
    clear_registry,
    get_spec,
    register,
)

__all__ = [
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "all_specs",
    "clear_registry",
    "get_spec",
    "register",
]
```

- [ ] **Step 2: Add a smoke test**

Append to `tests/test_agents_base.py` (add these two tests at the bottom, keeping the existing `_reset_registry` autouse fixture):

```python
def test_package_import_registers_all_six_tools() -> None:
    # Re-populate by importing the package.
    import importlib

    import vllama.agents.tools as pkg

    importlib.reload(pkg)  # re-runs side-effect imports

    names = {s.name for s in all_specs()}
    assert names == {
        "read_file",
        "write_file",
        "apply_patch",
        "list_dir",
        "glob_files",
        "grep",
    }


def test_all_definitions_match_openai_shape() -> None:
    import importlib

    import vllama.agents.tools as pkg

    importlib.reload(pkg)
    for spec in all_specs():
        assert spec.definition.get("type") == "function"
        fn = spec.definition.get("function")
        assert isinstance(fn, dict)
        assert fn.get("name") == spec.name
        assert "description" in fn
        assert "parameters" in fn
```

- [ ] **Step 3: Run the smoke tests**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: 7 tests pass (original 5 + 2 new).

- [ ] **Step 4: Commit**

```bash
git add src/vllama/agents/tools/__init__.py tests/test_agents_base.py
git commit -m "feat(agents): add tools package index and registry smoke tests"
```

---

## Task 9: `permissions.py` scaffold

**Files:**
- Create: `src/vllama/agents/permissions.py`
- Test: `tests/test_permissions.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_permissions.py`:

```python
"""Tests for the permissions scaffold."""

from __future__ import annotations

from vllama.agents.permissions import Decision, Policy, Tier


def test_tier_defaults_read_auto() -> None:
    p = Policy(overrides={})
    assert p.decide("read_file", Tier.READ) is Decision.AUTO


def test_tier_defaults_mutate_prompt() -> None:
    p = Policy(overrides={})
    assert p.decide("write_file", Tier.MUTATE) is Decision.PROMPT


def test_tier_defaults_exec_prompt() -> None:
    p = Policy(overrides={})
    assert p.decide("bash", Tier.EXEC) is Decision.PROMPT


def test_override_beats_tier_default() -> None:
    p = Policy(overrides={"write_file": Decision.AUTO})
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO


def test_trust_mode() -> None:
    p = Policy.trust_mode()
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO
    assert p.decide("bash", Tier.EXEC) is Decision.AUTO
    assert p.decide("read_file", Tier.READ) is Decision.AUTO
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_permissions.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement permissions scaffold**

Create `src/vllama/agents/permissions.py`:

```python
"""Permission policy scaffold. Phase 1 ships with Policy.trust_mode() as the default."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Tier(StrEnum):
    READ = "read"
    MUTATE = "mutate"
    EXEC = "exec"


class Decision(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"
    DENY = "deny"


_TIER_DEFAULTS: dict[Tier, Decision] = {
    Tier.READ: Decision.AUTO,
    Tier.MUTATE: Decision.PROMPT,
    Tier.EXEC: Decision.PROMPT,
}


@dataclass
class Policy:
    """Per-tool permission decision resolver.

    If ``trust_all`` is True, every tool returns ``Decision.AUTO``
    regardless of overrides or tier.
    """

    overrides: dict[str, Decision] = field(default_factory=dict)
    trust_all: bool = False

    def decide(self, tool_name: str, tier: Tier) -> Decision:
        if self.trust_all:
            return Decision.AUTO
        if tool_name in self.overrides:
            return self.overrides[tool_name]
        return _TIER_DEFAULTS[tier]

    @classmethod
    def trust_mode(cls) -> "Policy":
        """All tools auto-approve. Used in Phase 1 until PromptHook lands."""
        return cls(trust_all=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_permissions.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/permissions.py`
Run: `uv run ruff check src/vllama/agents/permissions.py tests/test_permissions.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/permissions.py tests/test_permissions.py
git commit -m "feat(agents): add permission Policy scaffold (trust mode default)"
```

---

## Task 10: `runtime.py` — dispatch + output cap

**Files:**
- Create: `src/vllama/agents/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runtime.py`:

```python
"""Tests for ToolRuntime dispatch + output cap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    clear_registry,
    register,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=32, cap_bytes_bash=128)


async def _echo_handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content=str(args.get("text", "")))


def _reg(name: str, tier: str = "read") -> None:
    register(
        ToolSpec(
            name=name,
            definition={
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            },
            tier=tier,  # type: ignore[arg-type]
            handler=_echo_handler,
        )
    )


async def test_dispatch_unknown_tool(ctx: ToolContext) -> None:
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("nope", "{}")
    assert r.error is True
    assert "unknown" in r.content.lower()


async def test_dispatch_invalid_json(ctx: ToolContext) -> None:
    _reg("echo")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("echo", "{not json")
    assert r.error is True
    assert "json" in r.content.lower()


async def test_dispatch_success(ctx: ToolContext) -> None:
    _reg("echo")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert r.error is False
    assert r.content == "hi"


async def test_dispatch_caps_output(ctx: ToolContext) -> None:
    _reg("echo")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    # ctx.cap_bytes = 32 from fixture.
    payload = "A" * 200
    r = await rt.dispatch("echo", '{"text": "' + payload + '"}')
    assert r.truncated is True
    assert r.content.startswith("A" * 32)
    assert "truncated" in r.content.lower()
    assert "200" in r.content  # total bytes reported


async def test_dispatch_caps_bash_separately(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8, cap_bytes_bash=64)
    _reg("bash", tier="exec")
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("bash", '{"text": "' + "x" * 50 + '"}')
    # Bash cap is 64, payload 50 — below cap, no truncate.
    assert r.truncated is False


async def test_dispatch_handler_exception_becomes_error_result(ctx: ToolContext) -> None:
    async def boom(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")

    register(
        ToolSpec(
            name="boom",
            definition={"type": "function", "function": {"name": "boom"}},
            tier="read",
            handler=boom,
        )
    )
    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx)
    r = await rt.dispatch("boom", "{}")
    assert r.error is True
    assert "kaboom" in r.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `runtime.py`**

Create `src/vllama/agents/runtime.py`:

```python
"""ToolRuntime: dispatches tool calls with permission check and output cap."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from vllama.agents.permissions import Decision, Policy, Tier
from vllama.agents.tools._base import ToolContext, ToolResult, get_spec


@dataclass
class ToolRuntime:
    policy: Policy
    ctx: ToolContext

    async def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        spec = get_spec(name)
        if spec is None:
            return ToolResult(content=f"Error: unknown tool '{name}'", error=True)

        try:
            args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as e:
            return ToolResult(content=f"Error: invalid JSON arguments: {e}", error=True)
        if not isinstance(args, dict):
            return ToolResult(
                content="Error: arguments must decode to a JSON object", error=True
            )

        tier = Tier(spec.tier)
        decision = self.policy.decide(name, tier)
        if decision is Decision.DENY:
            return ToolResult(content=f"Error: tool '{name}' denied by policy", error=True)
        # Decision.PROMPT handled in later phases (PromptHook). In Phase 1 we rely on
        # trust-mode or AUTO policies; treat PROMPT as AUTO for now.

        try:
            result = await spec.handler(args, self.ctx)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"Error in '{name}': {e}", error=True)

        cap = self.ctx.cap_bytes_bash if name.startswith("bash") else self.ctx.cap_bytes
        if len(result.content) > cap:
            total = len(result.content)
            result.content = (
                result.content[:cap]
                + f"\n[truncated at {cap} bytes of {total} total."
                + f" Re-call with offset={cap} to continue.]"
            )
            result.truncated = True
        return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: all 6 tests pass.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/runtime.py`
Run: `uv run ruff check src/vllama/agents/runtime.py tests/test_runtime.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/runtime.py tests/test_runtime.py
git commit -m "feat(agents): add ToolRuntime with dispatch, permission wrap, and output cap"
```

---

## Task 11: TUI integration

**Files:**
- Modify: `src/vllama/tui.py` (import swap + await dispatch)

This task replaces the legacy `tools.py` usage with the new registry + runtime. No new tests — relies on manual smoke test + existing TUI tests (none currently exist for tool flow).

- [ ] **Step 1: Read the existing tool-call site**

Open `src/vllama/tui.py` and locate:
- Line 33: `from vllama.tools import TOOL_DEFINITIONS, execute_tool`
- Around L678: `result = execute_tool(tc["name"], tc["arguments"])`
- Around L605–710: the tool-call accumulation + execution loop in `_send_message`.

The migration preserves the existing streaming + tool-call-accumulation logic; only the imports and the single `execute_tool` call need to change.

- [ ] **Step 2: Swap the import**

Modify `src/vllama/tui.py` around line 33:

```python
# Before:
from vllama.tools import TOOL_DEFINITIONS, execute_tool

# After:
from vllama.agents.permissions import Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools import ToolContext, all_specs
```

- [ ] **Step 3: Add runtime construction in `ChatApp.__init__`**

`Path` is already imported at the top of `tui.py` (`from pathlib import Path`). In `ChatApp.__init__` (around L310–338), after the existing attribute setup and before the method ends, add:

```python
self._tool_ctx = ToolContext(
    cwd=Path.cwd(),
    cap_bytes=8192,
    cap_bytes_bash=32768,
)
self._runtime = ToolRuntime(policy=Policy.trust_mode(), ctx=self._tool_ctx)
```

- [ ] **Step 4: Replace `TOOL_DEFINITIONS` references with `[s.definition for s in all_specs()]`**

Search the file for `TOOL_DEFINITIONS`:
```bash
grep -n "TOOL_DEFINITIONS" src/vllama/tui.py
```

At each reference site, replace with:
```python
[s.definition for s in all_specs()]
```

(Typically this is passed into the chat request body's `tools` field.)

- [ ] **Step 5: Replace `execute_tool` with `await self._runtime.dispatch`**

Around L678 in the tool-call execution loop, change:

```python
# Before:
result = execute_tool(tc["name"], tc["arguments"])
```

to:

```python
# After:
tool_result = await self._runtime.dispatch(tc["name"], tc["arguments"])
result = tool_result.content
```

Note: the surrounding code (`_send_message`) is already `async def`, so `await` is valid here.

If the code uses `result` directly (as a string) anywhere after, this drop-in works. If it uses `result.error` or other fields, also thread `tool_result` through.

Re-read the block around L678 and adjust consumers to work with the string `result` as they did before.

- [ ] **Step 6: Verify the app still launches**

Run: `uv run vllama chat --model <any-available-model>` (or whatever the chat launch command is — check `src/vllama/cli/main.py` for the exact invocation).

Expected: TUI opens without import errors. Type a message (even if no model is loaded, the app should render). Exit with the existing bind.

If the app errors on startup, read the traceback and fix the import/constructor mismatch.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -x`
Expected: all tests pass (old + new). Existing `tests/test_*` files should still work because we haven't deleted `tools.py` yet.

- [ ] **Step 8: Lint + mypy on the modified file**

Run: `uv run mypy src/vllama/tui.py`
Run: `uv run ruff check src/vllama/tui.py`
Expected: no errors. (Existing ignored errors unrelated to this change can stay.)

- [ ] **Step 9: Commit**

```bash
git add src/vllama/tui.py
git commit -m "feat(tui): wire ChatApp to new agents runtime + registry"
```

---

## Task 12: Remove legacy `tools.py`

**Files:**
- Delete: `src/vllama/tools.py`

- [ ] **Step 1: Confirm nothing else imports `vllama.tools`**

Run: `grep -rn "from vllama.tools\|import vllama.tools" src/ tests/`
Expected: no output (all imports migrated in Task 11).

If any references remain, update them the same way as Task 11 before deleting.

- [ ] **Step 2: Delete the file**

```bash
rm src/vllama/tools.py
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -x`
Expected: all tests pass.

- [ ] **Step 4: Final lint + mypy check on the whole source tree**

Run: `uv run ruff check src/ tests/`
Run: `uv run mypy src/`
Expected: no errors introduced by this plan. Pre-existing errors (unrelated) may remain; do not fix them here.

- [ ] **Step 5: Commit**

```bash
git add -u src/vllama/tools.py
git commit -m "refactor(agents): remove legacy tools.py (superseded by agents package)"
```

---

## Phase 1 Acceptance Criteria

- [ ] All 6 tools register on package import (`import vllama.agents.tools`).
- [ ] `ToolRuntime.dispatch` handles unknown tool, invalid JSON args, handler exceptions, and output cap correctly.
- [ ] TUI chat issues tool calls through the runtime, not the legacy dispatcher.
- [ ] Full test suite passes (`uv run pytest`).
- [ ] Ripgrep path works when `rg` is on `$PATH`; Python fallback works when it isn't.
- [ ] `apply_patch` fails safely on context mismatch (file untouched).
- [ ] `src/vllama/tools.py` is deleted.
- [ ] No mypy or ruff errors introduced.
