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
            return ToolResult(content="Error: arguments must decode to a JSON object", error=True)

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
                return ToolResult(content=f"Error: user denied '{name}' for this call", error=True)
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
