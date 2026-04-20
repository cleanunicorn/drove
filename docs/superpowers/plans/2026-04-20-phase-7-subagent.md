# Phase 7 — Subagent `task` Tool — Implementation Plan

**Goal:** Add a `task` tool so the model can delegate a sub-task to a fresh, bounded conversation. Subagent runs a headless turn loop (non-streaming) with the same model and a configurable tool subset, depth-capped.

**Architecture:**
- `subagent.py` — `SubagentRunner` dataclass wrapping the state a subagent needs: `base_url`, `model`, `api_key`, `runtime` (for tool dispatch), `max_iterations`, `depth_cap`.
  - `async run(description: str, prompt: str, allowed_tools: list[str] | None, depth: int) -> str` — builds fresh history, loops up to `max_iterations` non-streaming LLM calls + tool dispatches. Returns final assistant text or a short error marker.
  - Uses `call_chat_json` (non-streaming) for each iteration. Parses `tool_calls` from the response; dispatches via `runtime`.
  - Passes a filtered `runtime` view: tools not in `allowed_tools` are treated as unknown.
- `ToolContext` gets `subagent_runner: SubagentRunner | None = None` and `depth: int = 0`.
- `task` tool (tier=exec) in `tools/task.py`: validates `ctx.depth < depth_cap`, calls `subagent_runner.run(..., depth=ctx.depth + 1)`.
- `ChatApp` constructs the `SubagentRunner` alongside the main runtime.

**Tech Stack:** Python 3.14, httpx (non-streaming). mypy strict. ruff (E/F/I/UP, 100).

---

## Task 1: SubagentRunner

- Create `src/vllama/agents/subagent.py` with `SubagentRunner` + `run()`.
- Create `tests/test_subagent.py` — drive with scripted httpx.MockTransport responses covering: single-iter final reply; one tool call then done; depth exceeded raises (separately — this is checked in `task` tool); max_iter cap; allowed_tools filter.

## Task 2: ToolContext fields

- Add `subagent_runner: SubagentRunner | None = None` and `depth: int = 0` to ToolContext (kw_only, defaults). Use `TYPE_CHECKING` import.

## Task 3: `task` tool

- Create `src/vllama/agents/tools/task.py`. Args: `description` (str), `prompt` (str), `allowed_tools` (list[str] | None). Validates types. Checks `ctx.subagent_runner is not None` and `ctx.depth < runner.depth_cap`. Calls `await runner.run(description, prompt, allowed_tools, depth=ctx.depth + 1)`. Returns the subagent's final text as ToolResult.
- Register in package __init__.
- Update smoke test (12 tools).
- Create `tests/test_tools_task.py`.

## Task 4: TUI wiring

- ChatApp.__init__ constructs `SubagentRunner(base_url, model, api_key, runtime=self._runtime, max_iterations=cfg.agents.max_iterations, depth_cap=cfg.agents.subagent_depth)`.
- Thread it into `ToolContext.subagent_runner`.
- Add `subagent_depth: int = 3` to AgentsConfig.

## Task 5: Config addition + final tests

- Add `subagent_depth: int = 3` to `AgentsConfig`.
- Append config tests.
- Full suite green.
