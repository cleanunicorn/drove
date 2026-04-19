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
