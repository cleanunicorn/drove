# Phase 2 — Permission Model + Config — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 1's trust-mode `Policy` into a real permission gate. Load per-tool policy from `~/.config/vllama/config.toml`, prompt the user via a Textual modal for tools flagged `prompt`, support session-scoped allow-all, and honor a dedicated "Deny & Abort" path that cancels the whole turn.

**Architecture:**
- Extend `config.py` with an `AgentsConfig` nested model carrying `permissions: dict[str, Literal["auto","prompt","deny"]]`.
- Add `Policy.from_config(config_overrides) -> Policy`.
- Add `PromptHook = Callable[[str, dict[str, Any]], Awaitable[PromptDecision]]` and `AbortTurn` exception in `permissions.py`.
- Thread `prompt_hook` through `ToolContext` (keyword-only, defaults to None). `ToolRuntime.dispatch` calls the hook when `Decision.PROMPT`. Session permits cached on the runtime instance.
- In TUI: `PermissionModal` renders the request. `ChatApp` constructs `Policy.from_config(config.agents.permissions)`, wires a thin async wrapper around the modal as `prompt_hook`, catches `AbortTurn` in `_send_message`.
- Phase 1 review follow-ups folded in at the start: consolidate `_reset` + `ctx` fixtures into `tests/conftest.py`; `ToolContext` → `@dataclass(kw_only=True)`.

**Tech Stack:** Python 3.14, pydantic-settings (existing), Textual `ModalScreen[ReturnT]`, asyncio. mypy strict. ruff line-length 100.

---

## File Structure

```
tests/
    conftest.py                 # NEW: shared fixtures (_reset, ctx factory)

src/vllama/
    config.py                   # MODIFY: add AgentsConfig, attach to Config
    agents/
        _base.py                # MODIFY: ToolContext → kw_only=True
                                # (file lives at src/vllama/agents/tools/_base.py)
        permissions.py          # MODIFY: add AbortTurn, PromptHook, PromptDecision
                                # + Policy.from_config classmethod
        runtime.py              # MODIFY: honor PROMPT via prompt_hook; session_permits
    tui.py                      # MODIFY: import Policy.from_config,
                                # add PermissionModal, wrap as prompt_hook,
                                # handle AbortTurn, /permits slash command
```

New / modified test files:
```
tests/
    conftest.py                          # Task 1
    test_config.py                       # MODIFY: add tests for [agents.permissions]
    test_permissions.py                  # MODIFY: tests for from_config + AbortTurn
    test_runtime.py                      # MODIFY: prompt hook + session permit + abort tests
    test_permission_modal.py             # NEW: Textual pilot for modal
    test_tui_permits_command.py          # NEW: /permits slash command test
```

---

## Task 1: Consolidate shared test fixtures

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_tools_read.py`, `test_tools_write.py`, `test_tools_list.py`, `test_tools_glob.py`, `test_tools_grep.py`, `test_tools_edit.py`, `test_runtime.py`, `test_agents_base.py` (remove duplicate fixtures)

The six tool test files and the runtime test define the same `_reset` (autouse) and `ctx` fixtures. Move them into `tests/conftest.py` so tests can delete the boilerplate. Keep each file's `_load` helper local — it's module-specific.

- [ ] **Step 1: Create `tests/conftest.py`**

Create `tests/conftest.py`:

```python
"""Shared fixtures for the agents test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllama.agents.tools._base import ToolContext, clear_registry


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Clear the global tool registry before every test in tests/."""
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """Default ToolContext for tool handlers: cwd=tmp_path, caps set to real defaults."""
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
```

- [ ] **Step 2: Remove duplicate fixtures from each tool test**

For each of:
- `tests/test_tools_read.py`
- `tests/test_tools_write.py`
- `tests/test_tools_list.py`
- `tests/test_tools_glob.py`
- `tests/test_tools_grep.py`
- `tests/test_tools_edit.py`

Delete the local `_reset` fixture block and the local `ctx` fixture block. Keep the `_load` helper. Tests that accept `ctx: ToolContext` keep their parameter — the fixture now resolves from `conftest.py`.

Example diff for `tests/test_tools_read.py` (remove these lines):
```python
@pytest.fixture(autouse=True)
def _reset() -> None:
    clear_registry()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)
```

Also remove now-unused imports: `from vllama.agents.tools._base import ... clear_registry ...` (but keep `ToolContext` and `get_spec` if still referenced).

Do this for all 6 tool test files and `tests/test_runtime.py`. Do NOT touch `tests/test_agents_base.py` — that one has a different autouse fixture (`_reset_registry`) that can also be removed, but verify by reading first.

- [ ] **Step 3: Reconcile `tests/test_agents_base.py` and `tests/test_permissions.py`**

In `tests/test_agents_base.py`: the local `_reset_registry` fixture duplicates conftest's — delete it. Tests remain otherwise unchanged.

In `tests/test_permissions.py`: no registry state touched, no `ctx` used — leave as-is.

In `tests/test_tools_grep.py`: keeps its own `_force_python_impl` fixture (monkeypatches `shutil.which`) — that's not in conftest. Keep it.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -q`
Expected: 112 tests pass (same count, zero regressions).

- [ ] **Step 5: Lint + mypy**

Run: `uv run ruff check tests/`
Run: `uv run mypy src/vllama/agents/` (shouldn't be affected, but sanity-check).
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_*.py
git commit -m "refactor(tests): hoist shared _reset + ctx fixtures into conftest.py"
```

---

## Task 2: `ToolContext` → kw_only

**Files:**
- Modify: `src/vllama/agents/tools/_base.py`

Phase 1 review flagged: `ToolContext` is a `@dataclass` with positional fields. Every test constructs it via kwargs already, but adding fields later (Task 5 puts `prompt_hook` on the **runtime**, not here — but we might still add context fields in Phase 3+) risks positional drift. Switch to `kw_only=True` now. No new fields in this task.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agents_base.py` (append at bottom):

```python
def test_tool_context_requires_kwargs(tmp_path: Path) -> None:
    """ToolContext construction must be kw-only (protects against positional drift)."""
    import pytest

    from vllama.agents.tools._base import ToolContext

    with pytest.raises(TypeError):
        ToolContext(tmp_path, 8192, 32768)  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents_base.py::test_tool_context_requires_kwargs -v`
Expected: FAIL — positional construction currently works.

- [ ] **Step 3: Update `_base.py`**

In `src/vllama/agents/tools/_base.py`, change only the `ToolContext` decorator line from `@dataclass` to `@dataclass(kw_only=True)`:

```python
@dataclass(kw_only=True)
class ToolContext:
    """Runtime context passed to every tool handler."""

    cwd: Path
    cap_bytes: int
    cap_bytes_bash: int
```

Nothing else in the file changes.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_agents_base.py -v`
Expected: all tests pass including the new one. Existing construction sites already use kwargs (verified in Phase 1 review).

Run: `uv run pytest -q`
Expected: 113 tests pass (was 112 + 1 new).

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/`
Run: `uv run ruff check src/vllama/agents/ tests/test_agents_base.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/tools/_base.py tests/test_agents_base.py
git commit -m "refactor(agents): make ToolContext kw-only to prevent positional drift"
```

---

## Task 3: `AgentsConfig` in `config.py`

**Files:**
- Modify: `src/vllama/config.py`
- Modify: `tests/test_config.py`

Add a nested `AgentsConfig` Pydantic model with a `permissions: dict[str, str]` field. Wire it onto `Config` as `agents: AgentsConfig`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_agents_permissions_default_empty(tmp_path: Path) -> None:
    """Default config has no per-tool permission overrides."""
    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.agents.permissions == {}


def test_agents_permissions_from_toml(tmp_path: Path) -> None:
    import tomli_w

    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps(
            {
                "agents": {
                    "permissions": {
                        "write_file": "auto",
                        "bash": "deny",
                    },
                },
            }
        ).encode()
    )
    cfg = load_config(path)
    assert cfg.agents.permissions == {"write_file": "auto", "bash": "deny"}


def test_agents_permissions_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var overrides TOML for agents.permissions."""
    import tomli_w

    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps({"agents": {"permissions": {"bash": "prompt"}}}).encode()
    )
    monkeypatch.setenv("VLLAMA_AGENTS__PERMISSIONS", '{"bash": "auto"}')
    cfg = load_config(path)
    assert cfg.agents.permissions == {"bash": "auto"}


def test_agents_permissions_invalid_value_rejected(tmp_path: Path) -> None:
    """Unknown decision value raises at load time."""
    import tomli_w

    import pytest

    from vllama.config import load_config

    path = tmp_path / "c.toml"
    path.write_bytes(
        tomli_w.dumps({"agents": {"permissions": {"bash": "ignore"}}}).encode()
    )
    with pytest.raises(Exception):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `cfg.agents` attribute doesn't exist.

- [ ] **Step 3: Update `config.py`**

Modify `src/vllama/config.py`. Add these imports near the top (after existing imports):

```python
from typing import Literal

from pydantic import BaseModel
```

Add the nested `AgentsConfig` class right above `class Config`:

```python
DecisionValue = Literal["auto", "prompt", "deny"]


class AgentsConfig(BaseModel):
    """Configuration for the agents subsystem (Phase 2+)."""

    permissions: dict[str, DecisionValue] = {}
```

Attach it to `Config` by adding this line alongside the other fields:

```python
    agents: AgentsConfig = AgentsConfig()
```

Update `Config.save` to include agents in the TOML output. In the `data` dict, add:

```python
            "agents": self.agents.model_dump(),
```

(Before the closing `}` of the `data` dict.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 8 tests pass (4 existing + 4 new).

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/config.py`
Run: `uv run ruff check src/vllama/config.py tests/test_config.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/config.py tests/test_config.py
git commit -m "feat(config): add [agents.permissions] per-tool decision overrides"
```

---

## Task 4: `AbortTurn`, `PromptHook`, `Policy.from_config`

**Files:**
- Modify: `src/vllama/agents/permissions.py`
- Modify: `tests/test_permissions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_permissions.py`:

```python
def test_from_config_builds_policy() -> None:
    from vllama.agents.permissions import Decision, Policy, Tier

    p = Policy.from_config({"write_file": "auto", "bash": "deny"})
    assert p.decide("write_file", Tier.MUTATE) is Decision.AUTO
    assert p.decide("bash", Tier.EXEC) is Decision.DENY
    # Unmentioned tool falls back to tier default.
    assert p.decide("read_file", Tier.READ) is Decision.AUTO


def test_from_config_invalid_value_raises() -> None:
    import pytest

    from vllama.agents.permissions import Policy

    with pytest.raises(ValueError):
        Policy.from_config({"bash": "bogus"})


def test_abort_turn_is_exception() -> None:
    from vllama.agents.permissions import AbortTurn

    e = AbortTurn()
    assert isinstance(e, Exception)


def test_prompt_decision_literal_values() -> None:
    """PromptDecision values are well-defined and re-exported."""
    from vllama.agents.permissions import PROMPT_DECISIONS

    assert set(PROMPT_DECISIONS) == {
        "allow",
        "session_allow",
        "deny_continue",
        "deny_abort",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_permissions.py -v`
Expected: FAIL — `from_config`, `AbortTurn`, `PROMPT_DECISIONS` don't exist yet.

- [ ] **Step 3: Update `permissions.py`**

Replace the contents of `src/vllama/agents/permissions.py` with:

```python
"""Permission policy + prompt hook types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class Tier(StrEnum):
    READ = "read"
    MUTATE = "mutate"
    EXEC = "exec"


class Decision(StrEnum):
    AUTO = "auto"
    PROMPT = "prompt"
    DENY = "deny"


PromptDecision = Literal["allow", "session_allow", "deny_continue", "deny_abort"]
PROMPT_DECISIONS: tuple[PromptDecision, ...] = (
    "allow",
    "session_allow",
    "deny_continue",
    "deny_abort",
)

# Signature: (tool_name, tool_args) -> await decision
PromptHook = Callable[[str, dict[str, object]], Awaitable[PromptDecision]]


class AbortTurn(Exception):
    """Raised from ToolRuntime.dispatch when the user picks Deny & Abort.

    Callers (the TUI turn loop) catch this and cancel the whole turn.
    """


_TIER_DEFAULTS: dict[Tier, Decision] = {
    Tier.READ: Decision.AUTO,
    Tier.MUTATE: Decision.PROMPT,
    Tier.EXEC: Decision.PROMPT,
}


@dataclass
class Policy:
    """Per-tool permission decision resolver."""

    overrides: dict[str, Decision] = field(default_factory=dict)
    trust_all: bool = False

    def decide(self, tool_name: str, tier: Tier) -> Decision:
        if self.trust_all:
            return Decision.AUTO
        if tool_name in self.overrides:
            return self.overrides[tool_name]
        return _TIER_DEFAULTS[tier]

    @classmethod
    def trust_mode(cls) -> Policy:
        """All tools auto-approve. Used in tests and when agents.enabled=false."""
        return cls(trust_all=True)

    @classmethod
    def from_config(cls, overrides: dict[str, str]) -> Policy:
        """Build a Policy from a config dict of {tool_name: 'auto'|'prompt'|'deny'}."""
        parsed: dict[str, Decision] = {}
        for name, value in overrides.items():
            try:
                parsed[name] = Decision(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid permission decision for {name!r}: {value!r}."
                    f" Expected one of {[d.value for d in Decision]}."
                ) from e
        return cls(overrides=parsed)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_permissions.py -v`
Expected: all 9 tests pass (5 existing + 4 new).

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/permissions.py`
Run: `uv run ruff check src/vllama/agents/permissions.py tests/test_permissions.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/permissions.py tests/test_permissions.py
git commit -m "feat(agents): add AbortTurn, PromptHook, and Policy.from_config"
```

---

## Task 5: `ToolRuntime` honors PROMPT via hook + session permits

**Files:**
- Modify: `src/vllama/agents/runtime.py`
- Modify: `tests/test_runtime.py`

Runtime now asks the hook (if present) when `Decision.PROMPT`. Hook return values:
- `"allow"` → run this invocation.
- `"session_allow"` → run, add tool to `session_permits` (skips future prompts for that tool).
- `"deny_continue"` → return `ToolResult(error=True, content="Error: user denied this tool call")`.
- `"deny_abort"` → raise `AbortTurn`.

If the policy is `PROMPT` but no hook is configured (e.g., agentless tests), treat as a hard error (don't silently auto-approve — that was flagged in the Phase 1 review).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runtime.py` (keep existing tests):

```python
# ── PromptHook / AbortTurn / session-permit tests ──────────────────────────────
async def test_prompt_hook_invoked_on_prompt_tier(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Decision, Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")  # tier mutate → default PROMPT
    called = {"count": 0, "last_args": None}

    async def hook(name: str, args: dict[str, object]) -> str:
        called["count"] += 1
        called["last_args"] = args
        return "allow"

    rt = ToolRuntime(policy=Policy(), ctx=ctx, prompt_hook=hook)
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert called["count"] == 1
    assert called["last_args"] == {"text": "hi"}
    assert r.content == "hi"


async def test_session_allow_skips_future_prompts(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")
    counts = {"n": 0}

    async def hook(name: str, args: dict[str, object]) -> str:
        counts["n"] += 1
        return "session_allow"

    rt = ToolRuntime(policy=Policy(), ctx=ctx, prompt_hook=hook)
    await rt.dispatch("echo", '{"text": "a"}')
    await rt.dispatch("echo", '{"text": "b"}')
    await rt.dispatch("echo", '{"text": "c"}')
    assert counts["n"] == 1  # first prompted, rest auto-approved


async def test_deny_continue_returns_error(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")

    async def hook(name: str, args: dict[str, object]) -> str:
        return "deny_continue"

    rt = ToolRuntime(policy=Policy(), ctx=ctx, prompt_hook=hook)
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert r.error is True
    assert "denied" in r.content.lower()


async def test_deny_abort_raises_abort_turn(ctx: ToolContext) -> None:
    import pytest

    from vllama.agents.permissions import AbortTurn, Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")

    async def hook(name: str, args: dict[str, object]) -> str:
        return "deny_abort"

    rt = ToolRuntime(policy=Policy(), ctx=ctx, prompt_hook=hook)
    with pytest.raises(AbortTurn):
        await rt.dispatch("echo", '{"text": "x"}')


async def test_deny_decision_blocks_without_hook(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Decision, Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="read")
    rt = ToolRuntime(
        policy=Policy(overrides={"echo": Decision.DENY}), ctx=ctx, prompt_hook=None
    )
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert r.error is True
    assert "denied" in r.content.lower()


async def test_prompt_without_hook_is_hard_error(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")  # tier default = PROMPT
    rt = ToolRuntime(policy=Policy(), ctx=ctx, prompt_hook=None)
    r = await rt.dispatch("echo", '{"text": "hi"}')
    assert r.error is True
    assert "no prompt hook" in r.content.lower()


async def test_trust_mode_bypasses_hook(ctx: ToolContext) -> None:
    from vllama.agents.permissions import Policy
    from vllama.agents.runtime import ToolRuntime

    _reg("echo", tier="mutate")
    called = {"n": 0}

    async def hook(name: str, args: dict[str, object]) -> str:
        called["n"] += 1
        return "allow"

    rt = ToolRuntime(policy=Policy.trust_mode(), ctx=ctx, prompt_hook=hook)
    r = await rt.dispatch("echo", '{"text": "ok"}')
    assert called["n"] == 0  # trust mode short-circuits before the hook
    assert r.error is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runtime.py -v`
Expected: new tests FAIL because `ToolRuntime.__init__` doesn't accept `prompt_hook` and the PROMPT branch doesn't invoke hooks.

- [ ] **Step 3: Update `runtime.py`**

Replace `src/vllama/agents/runtime.py` with:

```python
"""ToolRuntime: dispatches tool calls with permission check and output cap."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from vllama.agents.permissions import (
    AbortTurn,
    Decision,
    Policy,
    PromptHook,
    Tier,
)
from vllama.agents.tools._base import ToolContext, ToolResult, get_spec


@dataclass
class ToolRuntime:
    policy: Policy
    ctx: ToolContext
    prompt_hook: PromptHook | None = None
    session_permits: set[str] = field(default_factory=set)

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

        if decision is Decision.PROMPT and name not in self.session_permits:
            if self.prompt_hook is None:
                return ToolResult(
                    content=(
                        f"Error: tool '{name}' requires a prompt but no prompt hook is"
                        f" configured. Set agents.permissions.{name} to 'auto' or 'deny',"
                        f" or wire a hook."
                    ),
                    error=True,
                )
            choice = await self.prompt_hook(name, args)
            if choice == "allow":
                pass
            elif choice == "session_allow":
                self.session_permits.add(name)
            elif choice == "deny_continue":
                return ToolResult(
                    content=f"Error: user denied '{name}' for this call", error=True
                )
            elif choice == "deny_abort":
                raise AbortTurn()
            else:  # defensive — Literal narrowing should prevent this
                return ToolResult(
                    content=f"Error: prompt hook returned unknown decision {choice!r}",
                    error=True,
                )

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
Expected: all tests pass (6 existing + 7 new = 13).

Run the full suite too:
Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/agents/`
Run: `uv run ruff check src/vllama/agents/ tests/`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/agents/runtime.py tests/test_runtime.py
git commit -m "feat(agents): wire prompt_hook + session permits + AbortTurn into runtime"
```

---

## Task 6: `PermissionModal` in TUI

**Files:**
- Modify: `src/vllama/tui.py` (add `PermissionModal` class)
- Create: `tests/test_permission_modal.py`

Create a Textual `ModalScreen[PromptDecision]`. Displays: tool name, tier, pretty-printed args (truncated at 1KB). Four action buttons + keybindings:
- `[A] Allow` — returns `"allow"`, key `a`
- `[S] Session-allow` — returns `"session_allow"`, key `s`
- `[D] Deny & Continue` — returns `"deny_continue"`, key `d`
- `[X] Deny & Abort` — returns `"deny_abort"`, key `x`
- `esc` — returns `"deny_abort"` (safest default)

- [ ] **Step 1: Write the failing test**

Create `tests/test_permission_modal.py`:

```python
"""Textual-pilot tests for PermissionModal."""

from __future__ import annotations

import pytest

from textual.app import App, ComposeResult

from vllama.agents.permissions import PROMPT_DECISIONS
from vllama.tui import PermissionModal


class _Harness(App[str]):
    """Small app that pushes the modal and records the result."""

    def __init__(self, name: str, args: dict[str, object]) -> None:
        super().__init__()
        self._tool_name = name
        self._tool_args = args
        self.result: str | None = None

    def compose(self) -> ComposeResult:
        return
        yield  # pragma: no cover

    async def on_mount(self) -> None:
        def done(choice: str | None) -> None:
            self.result = choice
            self.exit()

        await self.push_screen(
            PermissionModal(name=self._tool_name, args=self._tool_args), done
        )


@pytest.mark.parametrize(
    "key, expected",
    [
        ("a", "allow"),
        ("s", "session_allow"),
        ("d", "deny_continue"),
        ("x", "deny_abort"),
        ("escape", "deny_abort"),
    ],
)
async def test_modal_keys_return_decision(key: str, expected: str) -> None:
    assert expected in PROMPT_DECISIONS
    app = _Harness("write_file", {"path": "/tmp/x", "content": "hi"})
    async with app.run_test() as pilot:
        await pilot.press(key)
        await pilot.pause()
    assert app.result == expected


async def test_modal_renders_name_and_args() -> None:
    app = _Harness("bash", {"command": "ls -la", "run_in_background": False})
    async with app.run_test() as pilot:
        # Modal should be mounted and visible after on_mount
        screen = app.screen
        text = screen.render().__str__()
        assert "bash" in text
        # pretty-printed args contain the command string
        assert "ls -la" in text or "command" in text
        await pilot.press("a")
        await pilot.pause()
    assert app.result == "allow"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_permission_modal.py -v`
Expected: FAIL — `PermissionModal` doesn't exist.

- [ ] **Step 3: Add `PermissionModal` to `tui.py`**

In `src/vllama/tui.py`, near the other modal classes (next to `SessionPicker`), add:

```python
# ── Permission modal ────────────────────────────────────────────────────────────
class PermissionModal(ModalScreen[str]):
    """Modal that asks the user how to handle a prompt-tier tool call.

    Dismisses with one of: "allow" | "session_allow" | "deny_continue" | "deny_abort".
    """

    BINDINGS = [
        Binding("a", "allow", "Allow"),
        Binding("s", "session_allow", "Session-allow"),
        Binding("d", "deny_continue", "Deny & Continue"),
        Binding("x", "deny_abort", "Deny & Abort"),
        Binding("escape", "deny_abort", "Cancel"),
    ]

    def __init__(self, name: str, args: dict[str, object]) -> None:
        super().__init__()
        self._name = name
        self._args = args

    def compose(self) -> ComposeResult:
        import json as _json

        try:
            pretty = _json.dumps(self._args, indent=2, default=str)
        except (TypeError, ValueError):
            pretty = repr(self._args)
        if len(pretty) > 1024:
            pretty = pretty[:1024] + "\n… (truncated)"

        yield Vertical(
            Static(f"Tool call: {self._name}", id="perm-title"),
            Static(pretty, id="perm-args"),
            Horizontal(
                Static("[A]llow  [S]ession-allow  [D]eny&Continue  e[X]it-turn", id="perm-help"),
            ),
            id="perm-modal",
        )

    def action_allow(self) -> None:
        self.dismiss("allow")

    def action_session_allow(self) -> None:
        self.dismiss("session_allow")

    def action_deny_continue(self) -> None:
        self.dismiss("deny_continue")

    def action_deny_abort(self) -> None:
        self.dismiss("deny_abort")
```

Required imports (likely already present; add the missing ones at the top of `tui.py`):
```python
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_permission_modal.py -v`
Expected: 6 tests pass (5 param + 1 render).

- [ ] **Step 5: Lint + mypy**

Run: `uv run mypy src/vllama/tui.py`
Run: `uv run ruff check src/vllama/tui.py tests/test_permission_modal.py`

- [ ] **Step 6: Commit**

```bash
git add src/vllama/tui.py tests/test_permission_modal.py
git commit -m "feat(tui): add PermissionModal for prompt-tier tool calls"
```

---

## Task 7: Wire ChatApp — config-driven Policy + modal as prompt_hook + AbortTurn

**Files:**
- Modify: `src/vllama/tui.py`

Three changes to `ChatApp`:

1. Swap `Policy.trust_mode()` for `Policy.from_config(cfg.agents.permissions)` — use the loaded config.
2. Build an async `prompt_hook` that pushes `PermissionModal` and awaits its dismissal.
3. Catch `AbortTurn` in `_send_message` and post a note to the chat.

- [ ] **Step 1: Read the current `ChatApp.__init__`**

Open `src/vllama/tui.py` around L310–L350. Note how `self._cfg` (or equivalent) is populated — likely by calling `load_config()` in `ChatApp.__init__` or passed in from CLI.

If `ChatApp` does not currently load config, check the CLI entry point (`src/vllama/cli/main.py` → `chat` command) to see how to pass it in.

- [ ] **Step 2: Ensure `ChatApp` has access to the config**

If not already present, modify `ChatApp.__init__` to accept a `config: Config` parameter (or load via `load_config()` if simpler). Store as `self._cfg`.

Then update the CLI entry that constructs `ChatApp(...)` to pass the loaded config.

(Inspect the current code first; if config is already loaded somewhere in `ChatApp`, just reference it in Step 3 below.)

- [ ] **Step 3: Replace runtime construction**

Find the existing block in `ChatApp.__init__`:

```python
self._tool_ctx = ToolContext(
    cwd=Path.cwd(),
    cap_bytes=8192,
    cap_bytes_bash=32768,
)
self._runtime = ToolRuntime(policy=Policy.trust_mode(), ctx=self._tool_ctx)
```

Replace with:

```python
self._tool_ctx = ToolContext(
    cwd=Path.cwd(),
    cap_bytes=8192,
    cap_bytes_bash=32768,
)
self._runtime = ToolRuntime(
    policy=Policy.from_config(self._cfg.agents.permissions),
    ctx=self._tool_ctx,
    prompt_hook=self._prompt_hook,
)
```

- [ ] **Step 4: Add `_prompt_hook` method to `ChatApp`**

Inside the `ChatApp` class, add:

```python
async def _prompt_hook(self, name: str, args: dict[str, object]) -> str:
    """Push PermissionModal and await user's choice."""
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    def _on_close(choice: str | None) -> None:
        if choice is None:
            future.set_result("deny_abort")
        else:
            future.set_result(choice)

    await self.push_screen(PermissionModal(name=name, args=args), _on_close)
    return await future
```

Make sure `asyncio` is imported at the top of `tui.py` (it should be).

- [ ] **Step 5: Catch `AbortTurn` in `_send_message`**

Locate the outer try block (or add one) that wraps the tool-call loop inside `_send_message`. The relevant region is around L605–L710, specifically around the call:

```python
tool_result = await self._runtime.dispatch(tc["name"], tc["arguments"])
```

Wrap the tool-call-loop body in:

```python
try:
    # ... existing loop contents, including await self._runtime.dispatch(...)
except AbortTurn:
    await self._show_note("Turn aborted by user.")
    return
```

Import at top of file:
```python
from vllama.agents.permissions import AbortTurn
```

If a matching `_show_note` helper already exists (check `ChatApp` helpers), use it. If not, use `self.notify("Turn aborted by user.")` as a minimal fallback.

- [ ] **Step 6: Smoke-test imports and full test suite**

Run: `uv run python -c "from vllama.tui import ChatApp; print('ok')"`
Expected: "ok", no import errors.

Run: `uv run pytest -q`
Expected: all tests pass (existing + modal tests).

- [ ] **Step 7: Lint + mypy**

Run: `uv run mypy src/vllama/tui.py`
Run: `uv run ruff check src/vllama/tui.py`

- [ ] **Step 8: Manual smoke**

Launch the TUI with a model that supports tools:
```bash
uv run vllama chat --model <some-model>
```
Type a message that triggers `write_file`; expect a modal to appear. Press `a` → tool runs. Press `x` → modal closes, chat shows "Turn aborted by user."

If no model is available, skip this step and note it in the report.

- [ ] **Step 9: Commit**

```bash
git add src/vllama/tui.py src/vllama/cli/main.py  # if CLI touched
git commit -m "feat(tui): route prompt-tier tools through PermissionModal"
```

---

## Task 8: `/permits` slash command

**Files:**
- Modify: `src/vllama/tui.py` (`ChatApp._dispatch_command`)
- Create: `tests/test_tui_permits_command.py`

`/permits` prints a rendered summary of:
- Tier defaults (read=auto, mutate=prompt, exec=prompt).
- Config overrides (per-tool).
- Active session permits (set on the runtime).

- [ ] **Step 1: Write the failing test**

Create `tests/test_tui_permits_command.py`:

```python
"""Tests for /permits slash command handler logic (extracted, UI-free)."""

from __future__ import annotations

from vllama.agents.permissions import Decision, Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import ToolContext


def _make_runtime() -> ToolRuntime:
    from pathlib import Path

    ctx = ToolContext(cwd=Path("/tmp"), cap_bytes=8192, cap_bytes_bash=32768)
    pol = Policy.from_config({"write_file": "auto", "bash": "deny"})
    return ToolRuntime(policy=pol, ctx=ctx)


def test_permits_summary_includes_overrides_and_session() -> None:
    from vllama.tui import render_permits_summary

    rt = _make_runtime()
    rt.session_permits.add("apply_patch")
    text = render_permits_summary(rt)
    assert "write_file" in text and "auto" in text
    assert "bash" in text and "deny" in text
    assert "apply_patch" in text
    # Tier defaults reminder present:
    assert "read" in text.lower()
    assert "mutate" in text.lower() or "prompt" in text.lower()


def test_permits_summary_no_overrides_no_session() -> None:
    from pathlib import Path

    from vllama.tui import render_permits_summary

    rt = ToolRuntime(
        policy=Policy(),
        ctx=ToolContext(cwd=Path("/tmp"), cap_bytes=8192, cap_bytes_bash=32768),
    )
    text = render_permits_summary(rt)
    assert "no overrides" in text.lower() or "default" in text.lower()
    assert "no session permits" in text.lower() or "none" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tui_permits_command.py -v`
Expected: FAIL — `render_permits_summary` doesn't exist.

- [ ] **Step 3: Add `render_permits_summary` to `tui.py`**

At module level in `src/vllama/tui.py`, add:

```python
def render_permits_summary(runtime: ToolRuntime) -> str:
    """Human-readable summary of runtime permissions state, for /permits."""
    lines: list[str] = []
    lines.append("Tier defaults: read=auto, mutate=prompt, exec=prompt")
    if runtime.policy.trust_all:
        lines.append("Policy: TRUST MODE (all tools auto-approved)")
    elif runtime.policy.overrides:
        lines.append("Config overrides:")
        for name, decision in sorted(runtime.policy.overrides.items()):
            lines.append(f"  {name} = {decision.value}")
    else:
        lines.append("Config overrides: (none; tier defaults apply)")
    if runtime.session_permits:
        lines.append("Session permits: " + ", ".join(sorted(runtime.session_permits)))
    else:
        lines.append("Session permits: (none)")
    return "\n".join(lines)
```

- [ ] **Step 4: Wire `/permits` into `_dispatch_command`**

Locate `ChatApp._dispatch_command` (around L403–L445). Add a new branch before any `else`/fallback:

```python
if text.strip() == "/permits":
    await self._show_note(render_permits_summary(self._runtime))
    return
```

(Use whatever the existing note-posting helper is called. If it's `_show_note`, great. If something else, match it.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tui_permits_command.py -v`
Expected: both tests pass.

Run full suite:
Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Lint + mypy**

Run: `uv run mypy src/vllama/tui.py`
Run: `uv run ruff check src/vllama/tui.py tests/test_tui_permits_command.py`

- [ ] **Step 7: Commit**

```bash
git add src/vllama/tui.py tests/test_tui_permits_command.py
git commit -m "feat(tui): add /permits slash command to inspect current permissions"
```

---

## Task 9: Integration test — full prompt→allow→dispatch loop

**Files:**
- Create: `tests/test_runtime_prompt_integration.py`

End-to-end-ish test: real `ToolRuntime` + real `Policy.from_config` + a stubbed async hook. Uses a registered dummy tool, asserts the hook is called once, then skipped for the same tool after session_allow, and that `AbortTurn` propagates.

(Task 5 already has unit tests; this one ties config + policy + runtime together and serves as the phase's end-to-end check.)

- [ ] **Step 1: Write the test**

Create `tests/test_runtime_prompt_integration.py`:

```python
"""Integration test: config → Policy → ToolRuntime → prompt_hook → dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllama.agents.permissions import AbortTurn, Policy
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import (
    ToolContext,
    ToolResult,
    ToolSpec,
    register,
)


async def _echo(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return ToolResult(content=str(args.get("text", "")))


def _register_dummy(name: str, tier: str = "mutate") -> None:
    register(
        ToolSpec(
            name=name,
            definition={
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            },
            tier=tier,  # type: ignore[arg-type]
            handler=_echo,
        )
    )


async def test_end_to_end_prompt_then_session_allow(tmp_path: Path) -> None:
    _register_dummy("write_file")
    policy = Policy.from_config({"write_file": "prompt"})
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    calls: list[str] = []

    async def hook(name: str, args: dict[str, object]) -> str:
        calls.append(name)
        return "session_allow"

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=hook)
    r1 = await rt.dispatch("write_file", '{"text": "a"}')
    r2 = await rt.dispatch("write_file", '{"text": "b"}')
    r3 = await rt.dispatch("write_file", '{"text": "c"}')
    assert r1.content == "a"
    assert r2.content == "b"
    assert r3.content == "c"
    assert calls == ["write_file"]  # hooked only once


async def test_end_to_end_abort_turn_propagates(tmp_path: Path) -> None:
    _register_dummy("bash", tier="exec")
    policy = Policy()  # exec tier default = PROMPT
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    async def hook(name: str, args: dict[str, object]) -> str:
        return "deny_abort"

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=hook)
    with pytest.raises(AbortTurn):
        await rt.dispatch("bash", '{"command": "rm -rf /"}')


async def test_end_to_end_deny_config_blocks_without_hook(tmp_path: Path) -> None:
    _register_dummy("bash", tier="exec")
    policy = Policy.from_config({"bash": "deny"})
    ctx = ToolContext(cwd=tmp_path, cap_bytes=8192, cap_bytes_bash=32768)

    rt = ToolRuntime(policy=policy, ctx=ctx, prompt_hook=None)
    r = await rt.dispatch("bash", '{"command": "date"}')
    assert r.error is True
    assert "denied" in r.content.lower()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_runtime_prompt_integration.py -v`
Expected: 3 tests pass.

Run the full suite:
Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Lint + mypy**

Run: `uv run ruff check tests/test_runtime_prompt_integration.py`
Run: `uv run mypy src/vllama/agents/`

- [ ] **Step 4: Commit**

```bash
git add tests/test_runtime_prompt_integration.py
git commit -m "test(agents): end-to-end prompt/abort/deny integration"
```

---

## Phase 2 Acceptance Criteria

- [ ] `config.toml` `[agents.permissions]` section drives per-tool policy; env var override works.
- [ ] `Policy.from_config(dict)` builds a policy and raises on invalid values.
- [ ] `AbortTurn` exception exists in `vllama.agents.permissions`.
- [ ] `ToolRuntime.dispatch` invokes `prompt_hook` for `Decision.PROMPT`, caches `session_allow`, returns error on `deny_continue`, raises `AbortTurn` on `deny_abort`.
- [ ] No-hook + PROMPT decision returns a structured error (not silent AUTO).
- [ ] `PermissionModal` dismisses with one of the four decisions; `esc` = `deny_abort`.
- [ ] `ChatApp` constructs its runtime with a config-driven `Policy` + a modal-backed hook.
- [ ] `AbortTurn` caught in `_send_message` surfaces a chat note.
- [ ] `/permits` slash command renders current policy + session permits.
- [ ] Shared `_reset` + `ctx` fixtures live in `tests/conftest.py`; per-tool duplicates removed.
- [ ] `ToolContext` is `kw_only=True`.
- [ ] Full test suite passes (existing + Phase 2 additions).
- [ ] No mypy or ruff errors introduced by this phase.
