"""Headless subagent runner for the `task` tool.

Runs a bounded, non-streaming chat turn loop: non-streaming LLM call,
dispatch any returned tool_calls via a shared ToolRuntime, append tool
results to the fresh history, loop up to max_iterations. Returns the
subagent's final assistant text.

No Textual coupling. No permission prompts (subagent uses the parent
runtime's policy, which will fail open to error results on PROMPT if no
hook is configured — use `allowed_tools` to restrict instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from vllama.agents.llm_call import call_chat_completion, call_chat_json
from vllama.agents.router import select_tools
from vllama.agents.runtime import ToolRuntime
from vllama.agents.tools._base import ToolSpec, all_specs
from vllama.config import RouterConfig


class SubagentDepthExceeded(Exception):
    pass


@dataclass
class SubagentRunner:
    base_url: str
    model: str
    api_key: str | None
    runtime: ToolRuntime
    router_config: RouterConfig = field(default_factory=RouterConfig)
    max_iterations: int = 50
    depth_cap: int = 3

    async def run(
        self,
        *,
        description: str,
        prompt: str,
        allowed_tools: list[str] | None,
        depth: int,
    ) -> str:
        """Run a bounded subagent turn. Returns the final assistant text."""
        if depth >= self.depth_cap:
            raise SubagentDepthExceeded(f"subagent depth cap ({self.depth_cap}) reached")

        system = (
            "You are a focused sub-agent. You were spawned to handle a single"
            " sub-task. Work until done, then reply with a concise result."
            f" Task description: {description}"
        )
        history: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        full_specs = _filter_specs(all_specs(), allowed_tools)

        async with httpx.AsyncClient() as client:

            async def _llm_json_call(messages: list[dict[str, Any]]) -> str:
                return await call_chat_json(
                    client=client,
                    base_url=self.base_url,
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key,
                    temperature=0.0,
                    timeout=30.0,
                )

            for iter_idx in range(1, self.max_iterations + 1):
                # Router — pick a tool subset for this iteration.
                selected = await select_tools(
                    history=history,
                    all_specs=full_specs,
                    llm_call=_llm_json_call,
                    config=self.router_config,
                    iteration=iter_idx,
                )
                tools_payload = [s.definition for s in selected]

                try:
                    message = await call_chat_completion(
                        client=client,
                        base_url=self.base_url,
                        model=self.model,
                        messages=history,
                        api_key=self.api_key,
                        tools=tools_payload or None,
                    )
                except httpx.HTTPError as e:
                    return f"Subagent error: {e}"

                content = message.get("content") or ""
                tool_calls = message.get("tool_calls") or []

                if not tool_calls:
                    return str(content).strip()

                history.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": tool_calls,
                    }
                )

                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "")
                    args_raw = tc.get("function", {}).get("arguments", "{}")
                    result = await self.runtime.dispatch(name, args_raw)
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result.content,
                        }
                    )
            return (
                "Subagent hit max iterations without returning a final answer."
            )


def _filter_specs(
    all_: list[ToolSpec], allowed: list[str] | None
) -> list[ToolSpec]:
    if allowed is None:
        return all_
    allowed_set = set(allowed)
    return [s for s in all_ if s.name in allowed_set]


__all__ = ["SubagentRunner", "SubagentDepthExceeded"]
