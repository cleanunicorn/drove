"""Textual-pilot tests for PermissionModal."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

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

        await self.push_screen(PermissionModal(name=self._tool_name, args=self._tool_args), done)


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
        # Modal should be mounted; allow a tick for layout.
        await pilot.pause()
        modal = app.screen  # the pushed PermissionModal is the active screen
        title = modal.query_one("#perm-title", Static)
        args_widget = modal.query_one("#perm-args", Static)
        assert "bash" in str(title.render())
        args_text = str(args_widget.render())
        assert "ls -la" in args_text or "command" in args_text
        await pilot.press("a")
        await pilot.pause()
    assert app.result == "allow"
