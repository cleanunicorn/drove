"""task tool: delegate a sub-task to a fresh, depth-capped subagent."""

from __future__ import annotations

from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "task",
        "description": (
            "Delegate a focused sub-task to a fresh subagent with a fresh history."
            " Use for well-scoped pieces of work you want isolated from the current"
            " conversation (e.g. exploring a file, drafting a commit message)."
            " Returns the subagent's final reply as a single string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short human-readable label for the sub-task.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed instructions for the subagent.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional subset of tool names the subagent may call."
                        " Defaults to all tools."
                    ),
                },
            },
            "required": ["description", "prompt"],
        },
    },
}


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    description = args.get("description")
    prompt = args.get("prompt")
    if not isinstance(description, str) or not description:
        return ToolResult(content="Error: 'description' is required", error=True)
    if not isinstance(prompt, str) or not prompt:
        return ToolResult(content="Error: 'prompt' is required", error=True)

    allowed_raw = args.get("allowed_tools")
    allowed_tools: list[str] | None = None
    if allowed_raw is not None:
        if not isinstance(allowed_raw, list) or not all(
            isinstance(x, str) for x in allowed_raw
        ):
            return ToolResult(
                content="Error: 'allowed_tools' must be a list of strings", error=True
            )
        allowed_tools = [str(x) for x in allowed_raw]

    runner = ctx.subagent_runner
    if runner is None:
        return ToolResult(
            content="Error: subagent runner not available in this context",
            error=True,
        )

    from vllama.agents.subagent import SubagentDepthExceeded  # lazy: breaks cycle

    try:
        reply = await runner.run(
            description=description,
            prompt=prompt,
            allowed_tools=allowed_tools,
            depth=ctx.depth + 1,
        )
    except SubagentDepthExceeded as e:
        return ToolResult(content=f"Error: {e}", error=True)
    return ToolResult(content=reply, meta={"description": description})


register(
    ToolSpec(
        name="task",
        definition=_DEFINITION,
        tier="exec",
        handler=_handler,
    )
)
