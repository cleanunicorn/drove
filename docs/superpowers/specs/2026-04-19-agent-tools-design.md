# Agent Tools & Extended Problem Solving — Design Spec

**Date:** 2026-04-19
**Scope:** Expand `vllama` chat TUI from 2 file-I/O tools to a full agentic tool suite with LLM-based tool routing, per-iteration evaluator loop, permission gating, background shell execution, and subagent delegation. Inspired by Claude Code and opencode.

## Goals

1. Give small local models more tools so they can solve multi-step tasks in chat.
2. Guard against small-model misfires with a configurable permission model.
3. Keep context usage in check via LLM-based per-iteration tool routing.
4. Prevent premature termination via a done-evaluator that inspects results and pending todos.
5. Organize tools in a module that can later be reused by the proxy (out of scope now).

## Non-goals

- Proxy-level tool injection (agentic proxy). Tool module designed reusable but only TUI wired this spec.
- MCP server integration.
- Image/PDF/notebook tools.
- Cross-session todo persistence.
- Live streaming of foreground bash output in the TUI.

## Context

### Current state

- `src/vllama/tools.py` (108 lines): two tools — `read_file`, `write_file` — plus `execute_tool(name, arguments) -> str` dispatcher.
- `src/vllama/tui.py` (788 lines): Textual chat TUI. Turn loop already handles streaming, `tool_call` event accumulation, post-call dispatch, and appending tool results. See `_send_message` (L584–713) and `_stream_chat` (L715–768).
- `src/vllama/sessions.py`: chat sessions persisted as JSON per model. Messages, titles, metadata.
- `src/vllama/observe.py`: request/response logger.
- Only one model runs at a time; `ServerManager` swaps.

### Decisions taken during brainstorming

| Decision | Choice |
|---|---|
| Permission model | Configurable per-tool, tier-based default (read=auto, mutate/exec=prompt) |
| Bash scope | Full bash, timeout only, foreground + background |
| Working directory | TUI launch cwd (not configurable per-session) |
| Subagent | Same model, fresh history, depth cap |
| Todos | Ephemeral per-session (in-memory) |
| Web | WebFetch only (no search) |
| Permission UX | Modal block with Allow / Session-allow / Deny&Continue / Deny&Abort |
| Edit format | Unified diff (`apply_patch`) |
| Grep/glob impl | `ripgrep` when present, Python `re` + `pathlib` fallback |
| Output handling | Cap per tool + paging hint |
| Background bash interface | Run flag + `bash_output` + `bash_kill` |
| Integration surface | TUI-only; tools designed reusable |
| Tool routing | Per-iteration LLM router; permissive prompt |
| Done detection | Evaluator LLM call on tool-less response; checks todos |
| Max iterations per turn | 50 |
| Todos in evaluator prompt | Full JSON |

## Architecture

### Module layout

```
src/vllama/agents/
    __init__.py
    tools/
        __init__.py              # imports all tool modules → populates registry
        _base.py                 # ToolResult, ToolSpec, ToolContext, @register
        read.py                  # read_file
        write.py                 # write_file
        edit.py                  # apply_patch
        list.py                  # list_dir
        glob.py                  # glob_files
        grep.py                  # grep
        bash.py                  # bash (fg + bg), bash_output, bash_kill
        webfetch.py              # web_fetch
        todo.py                  # todo_write
        task.py                  # task (subagent)
    router.py                    # LLM tool selector
    evaluator.py                 # LLM done-detector
    compaction.py                # history summarize+trim
    rate_limit.py                # per-model token bucket + 429 retry
    runtime.py                   # ToolRuntime: dispatch + permission + caps
    permissions.py               # Policy, Tier, Decision, PromptHook
    bash_procs.py                # BgProcs singleton manager
    subagent.py                  # Subagent runner (fresh chat loop)
```

`src/vllama/tools.py` deleted after Phase 1 completes migration.

### Core types (`_base.py`)

```python
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Literal

@dataclass
class ToolResult:
    content: str
    error: bool = False
    truncated: bool = False
    meta: dict | None = None  # UI hints (e.g., diff text, shell_id)

@dataclass
class ToolContext:
    cwd: Path
    cap_bytes: int
    cap_bytes_bash: int
    bg_procs: "BgProcs"
    subagent_runner: "SubagentRunner"
    depth: int                       # 0 for top-level; subagent bumps it
    cancel_token: asyncio.Event

@dataclass
class ToolSpec:
    name: str
    definition: dict                 # OpenAI function schema
    tier: Literal["read", "mutate", "exec"]
    handler: Callable[[dict, ToolContext], Awaitable[ToolResult]]

# Registry
_REGISTRY: dict[str, ToolSpec] = {}

def register(spec: ToolSpec) -> None:
    _REGISTRY[spec.name] = spec

def all_specs() -> list[ToolSpec]:
    return list(_REGISTRY.values())
```

### Turn loop (replaces current loop in `tui.py`)

```
append user_msg to history
for iter in range(max_iterations):
    history = await compaction.maybe_compact(history, ctx_size)  # no-op if under threshold
    selected = await router.select(history, all_specs())         # skipped if router.enabled=false
    response = await stream_chat(history, selected)              # existing _stream_chat
    append assistant_msg to history
    if response.tool_calls:
        for tc in response.tool_calls:
            result = await runtime.dispatch(tc, ctx)              # may raise AbortTurn
            append tool_result msg
        continue
    else:
        if evaluator.enabled:
            verdict = await evaluator.check(history, todos)
            if verdict.done:
                break
            append system_nudge(verdict.reason)
            continue
        else:
            break
else:
    surface "Max iterations (50) reached" note; break
```

Each iteration: 1 router call + 1 main call + (1 evaluator call if tool-less). ≤3 inferences per iteration.

### Router (`router.py`)

```python
async def select(
    history: list[dict],
    all_specs: list[ToolSpec],
    client: httpx.AsyncClient,
    model: str,
    config: RouterConfig,
) -> list[ToolSpec]:
    if not config.enabled:
        return all_specs
    cache_key = (hash_history(history), len(all_specs))
    if config.cache_on_unchanged_history and cache_key in _cache:
        return _cache[cache_key]
    prompt = build_router_prompt(history, all_specs, permissive=config.permissive)
    resp = await client.post(
        f"{llama_server_base}/v1/chat/completions",
        json={"model": model, "messages": prompt, "temperature": 0},
    )
    try:
        names = parse_json_array(resp)   # e.g. ["read_file","grep"]
        selected = [s for s in all_specs if s.name in names]
        _cache[cache_key] = selected
        return selected
    except ParseError:
        return all_specs                  # fail-open
```

Router prompt template:

```
System: You are a tool selector. Given the conversation below, return a JSON array of tool names that MIGHT help accomplish the next step. Err on the side of INCLUSION — prefer over-including tools rather than missing a useful one. Only exclude tools that are clearly irrelevant.

Available tools:
- read_file: Read text file contents
- grep: Search file contents by regex
- bash: Run shell command
- ... (compact name + one-line desc for each)

Conversation (last 6 messages):
{last_6_messages}

Return JSON array only, no prose. Example: ["read_file","grep","bash"]
```

### Evaluator (`evaluator.py`)

```python
@dataclass
class Verdict:
    done: bool
    reason: str

async def check(
    history: list[dict],
    todos: list[dict],
    client, model,
    config: EvaluatorConfig,
) -> Verdict:
    if not config.enabled:
        return Verdict(done=True, reason="evaluator disabled")
    prompt = build_evaluator_prompt(
        last_user_msg(history),
        last_assistant_msg(history),
        todos if config.todos_in_prompt == "full" else todo_counts(todos),
    )
    resp = await client.post(...)
    try:
        j = parse_json_object(resp)       # {"done": bool, "reason": str}
        return Verdict(**j)
    except ParseError:
        return Verdict(done=True, reason="evaluator parse failure (fail-safe)")
```

Evaluator prompt:

```
System: Judge whether the user's request has been accomplished. Return JSON {"done": bool, "reason": string}.

Done criteria:
- The user's last request has been fulfilled by the assistant.
- All todos in the todo list are marked "completed" (if any).

Not-done signals:
- Assistant said "I will" / "next I'll" / "let me" without doing it.
- Todos with status "pending" or "in_progress" remain.
- Assistant answered a different question than asked.

User's last request:
{last_user}

Assistant's latest reply:
{last_assistant}

Todos:
{todos_json}

Return JSON only.
```

### Compaction (`compaction.py`)

When history token count exceeds `ctx_size × compaction.threshold` (default 0.7), summarize older messages into one replacement.

```python
async def maybe_compact(
    history: list[dict],
    ctx_size: int,
    client, model,
    config: CompactionConfig,
) -> list[dict]:
    tokens = estimate_tokens(history)
    if tokens < int(ctx_size * config.threshold):
        return history
    keep_tail = config.keep_tail_messages            # default 6
    head = history[:-keep_tail]
    tail = history[-keep_tail:]
    summary = await summarize(head, client, model, config)
    system_summary = {
        "role": "system",
        "content": f"[Earlier conversation summary]\n{summary}",
    }
    return [history[0]] + [system_summary] + tail    # keep original system prompt at [0]
```

Summarization prompt: *"Summarize the conversation below into a concise brief preserving decisions, file paths mentioned, tool results, and open tasks. Omit verbose tool output."*

Called before each main LLM call in the turn loop (not router/evaluator — cheap calls). On summarize failure → keep original history (fail-open), log warning.

### Rate limit (`rate_limit.py`)

Per-model configurable limits. Applies to all LLM calls (main, router, evaluator, compaction, subagent) routed through a shared `RateLimitedClient` wrapper around `httpx.AsyncClient`.

```python
@dataclass
class ModelLimits:
    base_delay_ms: int = 0
    requests_per_minute: int | None = None
    requests_per_hour: int | None = None
    max_retries: int = 5
    retry_max_backoff_s: int = 60

class RateLimitedClient:
    async def post(self, url, json, model: str) -> httpx.Response:
        limits = self._for(model)
        await self._token_bucket(limits).acquire()
        if limits.base_delay_ms:
            await asyncio.sleep(limits.base_delay_ms / 1000)
        for attempt in range(limits.max_retries):
            resp = await self._client.post(url, json=json)
            if resp.status_code != 429:
                return resp
            retry_after = _parse_retry_after(resp)  # honors Retry-After header
            if retry_after is None:
                retry_after = min(2 ** attempt, limits.retry_max_backoff_s)
            _surface_to_tui(f"Rate limited by {model}, retry in {retry_after}s")
            await asyncio.sleep(retry_after)
        raise RateLimitExceeded(model)
```

Token-bucket keyed per model: separate buckets for minute and hour windows. `requests_per_minute=None` disables that window.

`_surface_to_tui`: pushes a status-line note, not a modal; user can cancel turn if too slow.

### Runtime (`runtime.py`)

```python
class ToolRuntime:
    def __init__(self, policy: Policy, ctx: ToolContext, prompt_hook: PromptHook): ...

    async def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        spec = registry.get(name)
        if spec is None:
            return ToolResult(content=f"Error: unknown tool '{name}'", error=True)
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            return ToolResult(content=f"Error: invalid JSON args: {e}", error=True)

        decision = self.policy.decide(spec.name, spec.tier)
        if decision == Decision.DENY:
            return ToolResult(content="Error: tool denied by policy", error=True)
        if decision == Decision.PROMPT and spec.name not in self._session_permits:
            user_choice = await self.prompt_hook(spec.name, args)
            if user_choice == "session_allow":
                self._session_permits.add(spec.name)
            elif user_choice == "deny_continue":
                return ToolResult(content="Error: user denied this tool call", error=True)
            elif user_choice == "deny_abort":
                raise AbortTurn()

        try:
            result = await spec.handler(args, self.ctx)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return ToolResult(content=f"Error in {spec.name}: {e}", error=True)

        cap = self.ctx.cap_bytes_bash if spec.name.startswith("bash") else self.ctx.cap_bytes
        if len(result.content) > cap:
            total = len(result.content)
            result.content = result.content[:cap] + f"\n[truncated at {cap} bytes of {total} total. Re-call with offset={cap} to continue.]"
            result.truncated = True
        return result
```

### Permissions (`permissions.py`)

```python
class Tier(StrEnum):
    READ = "read"
    MUTATE = "mutate"
    EXEC = "exec"

class Decision(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"
    DENY = "deny"

_TIER_DEFAULTS = {
    Tier.READ: Decision.AUTO,
    Tier.MUTATE: Decision.PROMPT,
    Tier.EXEC: Decision.PROMPT,
}

class Policy:
    def __init__(self, overrides: dict[str, Decision]):
        self._overrides = overrides

    def decide(self, tool_name: str, tier: Tier) -> Decision:
        if tool_name in self._overrides:
            return self._overrides[tool_name]
        return _TIER_DEFAULTS[tier]

PromptHook = Callable[[str, dict], Awaitable[Literal["allow","session_allow","deny_continue","deny_abort"]]]
```

### Background procs (`bash_procs.py`)

```python
@dataclass
class BgProcess:
    shell_id: str
    pid: int
    command: str
    proc: asyncio.subprocess.Process
    buffer: bytearray             # ring buffer (bounded)
    exit_code: int | None = None
    exited_at: float | None = None

class BgProcs:
    def __init__(self, buffer_bytes: int): ...
    async def start(self, command: str, cwd: Path) -> str: ...  # returns shell_id
    async def output(self, shell_id: str, offset: int = 0) -> tuple[bytes, int | None]: ...
    async def kill(self, shell_id: str) -> bool: ...
    async def gc_loop(self) -> None:   # clears exited procs older than 10 min
    async def shutdown(self) -> None:  # SIGTERM all, await cleanup
```

Singleton owned by `ChatApp`. Wired into `ToolContext.bg_procs`.

### Subagent (`subagent.py`)

```python
@dataclass
class SubagentConfig:
    model: str
    depth: int                    # current depth of the caller
    depth_cap: int
    allowed_tools: set[str] | None

class SubagentRunner:
    async def run(
        self,
        description: str,
        prompt: str,
        config: SubagentConfig,
    ) -> str:
        if config.depth >= config.depth_cap:
            raise SubagentDepthExceeded(config.depth_cap)
        fresh_history = [
            {"role": "system", "content": subagent_system_prompt(description)},
            {"role": "user", "content": prompt},
        ]
        sub_ctx = ctx.replace(depth=config.depth + 1)
        sub_policy = Policy(overrides=restrict_to(allowed_tools))
        final_content = await run_turn_loop(
            history=fresh_history,
            ctx=sub_ctx,
            policy=sub_policy,
            max_iterations=cfg.max_iterations,
        )
        return final_content
```

Subagent uses the same `ServerManager` / llama-server and the same model. Tool subset enforced by synthesizing a restricted `Policy` that returns `DENY` for disallowed tools.

## Tool Suite (13 tools)

### File / code

**`read_file`** (read):
- Args: `path` (string, required), `offset` (int, default 0), `limit` (int, default all).
- Behavior: reject binary files (null-byte heuristic in first 8KB). Relative paths resolve against `ctx.cwd`. UTF-8 decode with `errors="replace"`.
- Errors: not found, not a file, permission.

**`write_file`** (mutate):
- Args: `path`, `content`. Creates parent dirs.
- Success: `"Wrote N bytes to /abs/path"`.

**`apply_patch`** (mutate):
- Args: `path`, `diff` (unified-diff string).
- Impl: `unidiff` parse + manual apply. Dry-run validate hunks against file; if any context mismatch → error with hunk index + mismatched line.
- Success: `"Applied N hunks to /abs/path"`.

**`list_dir`** (read):
- Args: `path` (default cwd), `recursive` (bool, default false), `max_entries` (default 500).
- Returns one entry per line: `dir/` vs file suffix; sorted alphabetically.

**`glob_files`** (read):
- Args: `pattern` (e.g., `"src/**/*.py"`), `cwd` (override, default ctx.cwd).
- Uses `pathlib.Path.glob` with recursive `**`. Returns paths sorted by mtime desc, capped at 1000.

**`grep`** (read):
- Args: `pattern` (regex), `path` (file or dir, default cwd), `glob` (filter, optional, default `**/*`), `context_lines` (default 0), `case_insensitive` (default false), `max_matches` (default 200).
- Impl: `rg` binary if on `$PATH` (detected once at startup and cached); else Python `re` + walk fallback. Same output format either way.
- Output: `path:line:match_content`. Multiline off. Paged via `offset`.

### Shell

**`bash`** (exec):
- Args: `command`, `timeout_ms` (default 120_000, max 600_000), `run_in_background` (bool, default false), `description` (string, for UI).
- Foreground: `asyncio.create_subprocess_shell`, pipe stdout+stderr, stream chunks to TUI via callback (see TUI section) while accumulating for return. Await with `asyncio.wait_for(timeout)`. Return combined output. Non-zero exit → still returned as content, error=false; model decides.
- Background: delegate to `BgProcs.start(command, ctx.cwd)`. Return `"Background shell started. shell_id=X, pid=Y"`.
- Timeout: `error=True`, content includes partial output + `"Process timed out after N ms and was killed."`

**`bash_output`** (read):
- Args: `shell_id`, `offset` (default 0).
- Polls `BgProcs.output`. Non-blocking. Returns new bytes from offset; includes exit code if exited.

**`bash_kill`** (mutate):
- Args: `shell_id`.
- `SIGTERM` → 2s wait → `SIGKILL` if still alive. Returns `"Killed shell X (exit code N)"`.

### Web

**`web_fetch`** (read):
- Args: `url`. (`prompt` reserved for future summarization; unused in MVP.)
- Impl: `httpx.AsyncClient.get(url, timeout=10, follow_redirects=True)`. Max 15MB pre-strip. Runs `readability-lxml` to extract main content → converts to Markdown-ish text via `html2text` or built-in tag-strip.
- Errors: connect error, HTTP 4xx/5xx (content returned but error=true), content-type not HTML → raw text with warning, size exceeded.

### Planning

**`todo_write`** (read):
- Args: `todos` — list of `{id: str, content: str, status: "pending"|"in_progress"|"completed"}`.
- Replaces current `ctx.todos` state (in-memory dict on `ChatApp`). Returns rendered checklist.
- Tier=read because it only mutates in-memory state, not filesystem.

### Delegation

**`task`** (exec):
- Args: `description` (short), `prompt` (detailed task for subagent), `allowed_tools` (optional subset of tool names).
- Delegates to `SubagentRunner.run`. Depth-capped at 3.
- Returns subagent's final assistant text.
- Errors: depth exceeded, subagent max_iterations reached, subagent error.

## TUI Integration

### Permission modal

`PermissionModal(ModalScreen[Literal["allow","session_allow","deny_continue","deny_abort"]])`:
- Shows: tool name, tier, args (pretty-printed JSON, truncated at 1KB).
- Buttons + keys: `[A]llow`, `[S]ession-allow`, `[D]eny & Continue`, `[X] Deny & Abort`. `Esc` = Deny & Abort.
- Blocks generation; awaited by `prompt_hook`.

### Message bubble

Extend `Message.append_tool_call` in `tui.py`:
- Add status badge: `pending | running | ok | error | denied | killed`.
- `apply_patch` → collapsible with syntax-highlighted diff.
- `bash` fg → live-preview pane shows last N lines (default 10, config `agents.bash.preview_lines`) while running; full output hidden in expandable section. On completion: preview stays, full output available on expand.
- `bash` bg → shows `[background]` tag + `shell_id`. `bash_output` calls render as extensions of the same bubble when `shell_id` matches.
- `todo_write` → renders current todos as checklist below bubble.
- `task` → nested collapsible showing subagent messages (collapsed by default).
- Truncation marker visible when `ToolResult.truncated=True`.

### Slash commands

Add to `_dispatch_command`:
- `/tools` — list active tool specs for current turn.
- `/permits` — show current policy + session overrides.
- `/bg` — list running background procs.
- `/kill <shell_id>` — kill bg proc.
- `/todos` — show current todo list.

### Status line

Extend `_update_status`:
- Mid-turn: `iter 3/50 • bash running (12s) • 2 todos pending`.
- Idle: existing `cwd • model` display.

### Cancellation

Existing `is_generating` reactive. Extend so cancel:
- Cancels in-flight tool task.
- Does not kill bg procs.
- Discards pending router / evaluator calls.

Implementation via `asyncio.TaskGroup` or explicit `asyncio.Task` tracking in `_send_message`.

## Config Schema

### `~/.config/vllama/config.toml` — `[agents]` section

```toml
[agents]
enabled = true
max_iterations = 50
output_cap_bytes = 8192
output_cap_bash = 32768
subagent_depth = 3

[agents.router]
enabled = true
permissive = true
cache_on_unchanged_history = true
heuristic_short_circuit = false   # optional: keyword-based pre-filter skips LLM router
skip_on_first_iteration = true    # on 1st iter of a turn, send all tools (assume user req needs all)

[agents.evaluator]
enabled = true
todos_in_prompt = "full"          # "full" | "counts"
skip_when_no_todos_and_long_reply = true  # if 0 todos and assistant reply > 200 chars, treat done

[agents.compaction]
enabled = true
threshold = 0.7                   # fraction of ctx_size
keep_tail_messages = 6            # messages kept verbatim at end

[agents.bash]
default_timeout_ms = 120000
max_timeout_ms = 600000
background_buffer_bytes = 65536
kill_on_session_end = true
preview_lines = 10                # live preview lines for fg bash

[agents.web]
fetch_timeout_s = 10
fetch_max_bytes = 15000000

# Per-model rate limits. Key = model name (matches what proxy receives).
# Omit a model to use no limits.
[agents.rate_limit."qwen-2.5-coder-7b"]
base_delay_ms = 0
requests_per_minute = 60
requests_per_hour = 1000
max_retries = 5
retry_max_backoff_s = 60

[agents.rate_limit."free-remote-model"]
base_delay_ms = 500
requests_per_minute = 10
requests_per_hour = 100
max_retries = 10
retry_max_backoff_s = 120

[agents.permissions]
# Values: "auto" | "prompt" | "deny"
# Omitted tools fall back to tier defaults (read=auto, mutate=prompt, exec=prompt).
read_file = "auto"
list_dir = "auto"
glob_files = "auto"
grep = "auto"
todo_write = "auto"
web_fetch = "auto"
write_file = "prompt"
apply_patch = "prompt"
bash = "prompt"
bash_output = "auto"
bash_kill = "prompt"
task = "prompt"
```

### Pydantic models

Added to `config.py` as nested `AgentsConfig`, attached to `Settings` as `agents: AgentsConfig`. Env var overrides follow existing `VLLAMA_` prefix with double-underscore nesting: `VLLAMA_AGENTS__MAX_ITERATIONS=100`.

## Error Handling

### Tool-level

Handlers return `ToolResult(error=True, content=...)` rather than raise. Errors are model-visible:

- Arg validation: `"Error: invalid argument 'path' — expected string"`.
- Filesystem: `"Error: file not found: /x/y"`.
- Patch fail: `"Error: patch failed at hunk 2. Context line 'foo' not found. Re-read file and rebuild diff."`
- Bash non-zero: content = output, error=false (model decides).
- Bash timeout: error=true, partial output + timeout message.
- Web: specific HTTP / size / parse errors.
- Subagent: depth exceeded, iterations, internal errors.

### Runtime-level

- `AbortTurn` exception (from Deny & Abort) caught in `_send_message`; TUI posts `"Turn aborted."`.
- Max iterations → loop breaks; TUI posts `"Max iterations (50) reached."`.
- Router/evaluator/compaction parse failure → fail-open (router returns all; evaluator returns done=true; compaction keeps original history).
- llama-server unavailable → existing httpx error path.
- Rate limit hit (HTTP 429) → `RateLimitedClient` honors `Retry-After`, sleeps, retries up to `max_retries`. TUI status-line note during wait. Exhausted retries → `RateLimitExceeded` surfaced as tool-level or turn-level error depending on caller.
- Compaction LLM failure → log warning, continue with uncompacted history (may OOM context; last-resort truncate to `keep_tail_messages` + original system prompt).

### Background

- Zombies: keep buffer + exit code for 10 min after exit, then GC.
- SIGTERM-ignored: escalate to SIGKILL after 2s.
- Double-kill: second call returns `"Error: shell X already terminated"`.

### Logging

Extend `observe.py` to log tool calls with `{turn_id, iter, tool, args, result_bytes, error, duration_ms}`.

## Testing

### Unit (tests/agents/)

- `test_tools_<name>.py` per tool — happy path, arg validation, errors.
- `test_runtime.py` — permission wrap, output cap, paging marker, abort behavior.
- `test_permissions.py` — tier defaults, overrides, session permits.
- `test_bash_procs.py` — bg lifecycle, ring buffer, GC, double-kill.
- `test_router.py` — parse, fallback, caching, prompt assembly.
- `test_evaluator.py` — done/not-done, parse failure, todo integration.
- `test_subagent.py` — fresh history, depth cap, tool subset.
- `test_compaction.py` — below-threshold no-op, over-threshold summarize, keep-tail preserved, summarize failure fail-open.
- `test_rate_limit.py` — token bucket window enforcement, base delay applied, 429 with `Retry-After` retry, exponential fallback when no header, per-model isolation.

### Integration (tests/test_chat_loop.py)

Mocked llama-server. Scripts: multi-tool flow, bash timeout, router selection, evaluator nudge loop.

### TUI

Textual pilot for `PermissionModal`. Other TUI behavior covered by existing tests.

### Fixtures

- `tmp_path` for file tools.
- `FakeLlamaServer` httpx mock returning scripted responses.

## Phased Delivery

Each phase ships independently. `writing-plans` produces a concrete plan per phase as it begins.

**Phase 1 — Core file/code tools + runtime scaffold**
Scope: `_base.py`, registry, `ToolContext`, migrated `read_file`, `write_file`, `apply_patch`, `list_dir`, `glob_files`, `grep`, `runtime.py` with output cap (no permission gating yet — trust mode default), TUI import switch, tests.

**Phase 2 — Permission model + config**
Scope: `Policy` with config-driven overrides, `PermissionModal`, `prompt_hook`, session permits, `/permits` command, tests.

**Phase 3 — Bash + background**
Scope: `bash` (fg), `BgProcs`, `bash_output`, `bash_kill`, `/bg`, `/kill`, tests.

**Phase 4 — Router + evaluator**
Scope: `router.py`, `evaluator.py`, turn-loop rewrite, iteration cap, status line updates, tests with mocked LLM.

**Phase 5 — Planning**
Scope: `todo_write`, todos in evaluator prompt, TUI todo rendering, tests.

**Phase 6 — Web**
Scope: `web_fetch`, `readability-lxml` dep, tests with httpx mock.

**Phase 7 — Subagent**
Scope: `SubagentRunner`, `task` tool, subagent bubble nesting, tests.

**Phase 8 — Compaction + rate limiting**
Scope: `compaction.py` (summarize head + keep tail), token estimator, wire into turn loop; `rate_limit.py` (`RateLimitedClient`, token bucket, 429 retry with `Retry-After`), per-model config, TUI status-line wait note; tests with mocked 429 responses.

## Open / Deferred

- Proxy-level tool injection (agentic-proxy product direction).
- MCP server integration.
- Image / PDF / notebook tools.
- Cross-session / project-level todos.
- WebSearch (any provider).
- Subagent using a *different* model than the parent.
- Exact-string `edit` tool alongside `apply_patch` (may revisit if small-model diff accuracy is poor).
