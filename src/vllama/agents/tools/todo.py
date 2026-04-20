"""todo_write tool: track a model-maintained plan in-session.

Replaces `ctx.todos` in place so evaluator and TUI see the same list.
Ephemeral — not persisted across sessions (by design, matches Claude Code).
"""

from __future__ import annotations

from typing import Any

from vllama.agents.tools._base import ToolContext, ToolResult, ToolSpec, register

_VALID_STATUSES = {"pending", "in_progress", "completed"}

_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": (
            "Replace the current in-session todo list with a new plan. Use this to"
            " track multi-step work. Each item has id, content, status. Status must"
            " be one of: pending, in_progress, completed. Call again with an updated"
            " list to mark items complete or add steps. Todos are ephemeral — they"
            " live only for the current chat session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Full replacement list of todos.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": sorted(_VALID_STATUSES),
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
}


def _render(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "Todo list cleared."
    lines = ["Todo list:"]
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    for t in todos:
        mark = marks.get(t["status"], "[?]")
        lines.append(f"  {mark} {t['id']}: {t['content']}")
    return "\n".join(lines)


async def _handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw = args.get("todos")
    if not isinstance(raw, list):
        return ToolResult(content="Error: 'todos' argument must be a list", error=True)

    validated: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            return ToolResult(
                content=f"Error: todos[{idx}] must be an object", error=True
            )
        tid = item.get("id")
        content = item.get("content")
        status = item.get("status")
        if not isinstance(tid, str) or not tid:
            return ToolResult(
                content=f"Error: todos[{idx}].id must be a non-empty string",
                error=True,
            )
        if not isinstance(content, str) or not content:
            return ToolResult(
                content=f"Error: todos[{idx}].content must be a non-empty string",
                error=True,
            )
        if not isinstance(status, str) or status not in _VALID_STATUSES:
            return ToolResult(
                content=(
                    f"Error: todos[{idx}].status must be one of"
                    f" {sorted(_VALID_STATUSES)}; got {status!r}"
                ),
                error=True,
            )
        validated.append({"id": tid, "content": content, "status": status})

    ctx.todos[:] = validated
    return ToolResult(content=_render(validated), meta={"todos": validated})


register(
    ToolSpec(
        name="todo_write",
        definition=_DEFINITION,
        tier="read",
        handler=_handler,
    )
)
