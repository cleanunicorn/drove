# Phase 5 — Todo Write + Evaluator Todos Signal — Implementation Plan

> **For agentic workers:** Execute task-by-task; each step is a small TDD cycle.

**Goal:** Add `todo_write` tool so the model can track multi-step plans; thread todos into the evaluator so incomplete items bias toward "not done"; render todos in the TUI under their tool bubble.

**Architecture:**
- `ToolContext.todos: list[dict]` — in-memory list, mutated by `todo_write`, read by evaluator. Default empty.
- `todo_write` tool (tier=read) — replaces `ctx.todos` with the supplied list after validation.
- `check_done` evaluator gains `todos` kwarg; any non-completed todo flips the long-reply short-circuit off and nudges the model to continue.
- TUI turn loop passes `self._tool_ctx.todos` into `check_done`.

**Tech Stack:** Python 3.14, mypy strict, ruff (E/F/I/UP, 100).

---

## Task 1: `ToolContext.todos` field

- Modify: `src/vllama/agents/tools/_base.py` (add `todos: list[dict[str, Any]] = field(default_factory=list)`).
- Modify: `tests/conftest.py` (ctx fixture continues to default-init; todos accessed as mutable list).
- Append to `tests/test_agents_base.py`: test that `ctx.todos == []` by default, is mutable, and survives repeated handler calls.

## Task 2: `todo_write` tool

- Create: `src/vllama/agents/tools/todo.py` — registers `todo_write`, tier=read. Args: `todos: list[{id, content, status}]`. Validates each entry: `id` (str), `content` (str), `status in {"pending","in_progress","completed"}`. Replaces `ctx.todos` in place (`ctx.todos[:] = new_list`). Returns rendered checklist.
- Modify: `src/vllama/agents/tools/__init__.py` (add side-effect import).
- Modify: `tests/test_agents_base.py` smoke tests to expect 10 tools (adding `todo_write`).
- Create: `tests/test_tools_todo.py` — 6 tests: replace, validation errors per field, returns checklist, empty input clears, status values.

## Task 3: Evaluator todos signal

- Modify: `src/vllama/agents/evaluator.py` — add `todos: list[dict[str, Any]] | None = None` kwarg to `check_done`. If any entry has status != "completed":
  - Skip the long-reply short-circuit (force the LLM call so model has a chance to nudge).
  - Include todos in the prompt (full JSON, truncated at 2000 chars).
- Modify: `tests/test_evaluator.py` — add 3 tests: pending todos bypass long-reply skip; completed-only todos behave as before; todos appear in the prompt.

## Task 4: Turn-loop passes todos to evaluator

- Modify: `src/vllama/tui.py` — `_send_message` calls `check_done(history=..., todos=self._tool_ctx.todos, llm_call=..., config=...)`.

## Task 5: TUI renders todo checklist

- Modify: `src/vllama/tui.py` — when the `todo_write` tool result comes back, render a static checklist widget below the tool-call bubble. Simple: the tool's `ToolResult.content` is already a rendered checklist; the existing `append_tool_call` collapsible displays it. Plus: add a `/todos` slash command that prints the current list.
- Create: `tests/test_tui_todos_command.py` — 2 tests for `render_todos_summary(todos) -> str`.
