"""Conversation permissions: policy, editor transactions, wiring and lifecycle."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from textual.widgets import Button, Input, Label, OptionList, Select, Switch

from drove import tools
from drove.permissions import PermissionsModal, make_rule
from drove.sessions import new_session
from drove.tool_policy import Grant, PolicyContext, decide, parse_grants
from drove.tools import ToolContext, ToolFailure, execute_tool
from drove.tui import ChatApp, ChatInput, ChatTurn


def app_for(tmp_path, remote=False, config=None):
    return ChatApp(
        "https://remote.test" if remote else "http://localhost:8080", "m", tmp_path, config
    )


def test_editor_scopes(tmp_path):
    assert make_rule("read_file", "sub", tmp_path).directory == str(tmp_path / "sub")
    assert make_rule("write_file", "sub", tmp_path).directory == str(tmp_path / "sub")
    shell = make_rule("shell_execute", "  git status  ", tmp_path)
    assert shell == Grant("shell_execute", directory=str(tmp_path), command="git status")
    assert make_rule("fetch_url", "https://EXAMPLE.com", tmp_path, True) == Grant(
        "fetch_url", origin="https://example.com:443", internal=True
    )
    for tool, scope in [("bad", "x"), ("read_file", " "), ("fetch_url", "ftp://x")]:
        with pytest.raises(ValueError):
            make_rule(tool, scope, tmp_path)


@pytest.mark.parametrize("suffix", ["", " ", "; echo danger", " $(echo danger)", "\necho danger"])
def test_shell_exact_command_and_cwd(tmp_path, suffix):
    rule = make_rule("shell_execute", "git status", tmp_path)
    ctx = PolicyContext(tmp_path, False, frozenset({rule}), managed=True)
    for background in [True, False]:
        args = json.dumps(
            {"command": "git status" + suffix, "background": background, "timeout": 99}
        )
        assert decide("shell_execute", args, ctx).prompt == bool(suffix.strip())
        assert decide(
            "shell_execute",
            args,
            PolicyContext(tmp_path / "other", False, ctx.grants, managed=True),
        ).prompt
        # Config grants remain the old contract for write_file (editor-only), but shell_execute
        # grants from PR #172 are persistent and can be saved.
        parsed = parse_grants([rule.token(), make_rule("write_file", "sub", tmp_path).token()])
        assert len(parsed) == 1  # shell grant parsed, write_file rejected
        assert list(parsed)[0].tool == "shell_execute"


def test_write_directory_and_symlink_scope(tmp_path):
    allowed = tmp_path / "out"
    allowed.mkdir()
    rule = make_rule("write_file", "out", tmp_path)
    ctx = PolicyContext(tmp_path, False, frozenset({rule}), managed=True)
    assert not decide("write_file", '{"path":"out/a"}', ctx).prompt
    assert decide("write_file", '{"path":"out-other/a"}', ctx).prompt
    assert decide("write_file", '{"path":"out/../a"}', ctx).prompt
    link = allowed / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    assert decide("write_file", '{"path":"out/link/a"}', ctx).prompt
    # Execution checks again, even if a symlink changed since authorization.
    result = execute_tool(
        "write_file",
        '{"path":"out/link/nested/a","content":"x"}',
        context=ToolContext(cwd=tmp_path, write_roots=(allowed,)),
    )
    assert isinstance(result, ToolFailure)
    assert not (tmp_path / "nested").exists()
    assert "Successfully" in execute_tool(
        "write_file",
        '{"path":"out/nested/a","content":"x"}',
        context=ToolContext(cwd=tmp_path, write_roots=(allowed,)),
    )


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    "tool,args",
    [
        ("read_file", '{"path":"/outside/a"}'),
        ("list_files", '{"path":"/outside"}'),
        ("search_files", '{"path":"/outside"}'),
        ("write_file", '{"path":"a","content":"x"}'),
        ("shell_execute", '{"command":"anything"}'),
        ("fetch_url", '{"url":"http://127.0.0.1"}'),
    ],
)
def test_allow_all_and_disable_restores_policy(tmp_path, remote, tool, args):
    app = app_for(tmp_path, remote)
    before = app._policy_context()
    app._allow_all = True
    assert not decide(tool, args, app._policy_context()).prompt
    app._allow_all = False
    assert app._policy_context() == before
    assert decide(tool, args, before).prompt


async def test_command_discovery_dispatch_help_cancel(tmp_path, monkeypatch):
    app = app_for(tmp_path)
    send = AsyncMock()
    monkeypatch.setattr(app, "_start_turn", send)
    async with app.run_test(size=(100, 42)) as pilot:
        assert "/permissions" in app._COMMANDS
        notes = AsyncMock()
        monkeypatch.setattr(app, "_show_note", notes)
        await app._dispatch_command("/help")
        assert "/permissions" in notes.call_args.args[0]
        field = app.query_one(ChatInput)
        field.load_text("/per")
        await pilot.press("tab")
        assert field.text == "/permissions"
        before = app._policy_context()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, PermissionsModal)
        app.screen.query_one("#permission-allow-all", Switch).value = True
        await pilot.press("escape")
        await pilot.pause()
        assert app._policy_context() == before
        send.assert_not_called()


async def settle(pilot, condition) -> None:
    """Wait until *condition* holds; a pause alone can return before bubbled messages arrive."""
    async with asyncio.timeout(5):
        while not condition():
            await pilot.pause(0.01)


async def click(pilot, selector: str) -> None:
    """Click a button once it accepts clicks; it ignores them while its press effect shows."""
    button = pilot.app.screen.query_one(selector, Button)
    await settle(pilot, lambda: not button.has_class("-active"))
    await pilot.click(selector)


async def test_modal_rule_crud_and_apply(tmp_path):
    app = app_for(tmp_path)
    async with app.run_test(size=(100, 50)) as pilot:
        await app._dispatch_command("/permissions")
        await settle(
            pilot,
            lambda: (
                isinstance(app.screen, PermissionsModal)
                and bool(app.screen.query("#permission-scope"))
            ),
        )
        modal = app.screen
        assert isinstance(modal, PermissionsModal)
        scope = modal.query_one("#permission-scope", Input)
        rules = modal.query_one("#permission-rules", OptionList)
        modal.query_one("#permission-tool", Select).value = "shell_execute"
        scope.value = "git status"
        await click(pilot, "#permission-save")
        await settle(pilot, lambda: len(modal.rules) == 1)
        assert app._conversation_rules is None
        rules.highlighted = 0
        rules.focus()
        await pilot.press("enter")
        await settle(pilot, lambda: scope.value == "git status")
        scope.value = "git diff"
        await click(pilot, "#permission-save")
        await settle(pilot, lambda: modal.rules[0].command == "git diff")
        assert len(modal.rules) == 1
        rules.highlighted = 0
        rules.focus()
        await pilot.press("enter")
        await settle(pilot, lambda: scope.value == "git diff")
        await click(pilot, "#permission-remove")
        await settle(pilot, lambda: not modal.rules)
        modal.query_one("#permission-tool", Select).value = "write_file"
        scope.value = str(tmp_path)
        await click(pilot, "#permission-save")
        await settle(pilot, lambda: len(modal.rules) == 1)
        modal.query_one("#permission-allow-all", Switch).value = True
        await click(pilot, "#permissions-apply")
        await settle(pilot, lambda: app._allow_all)
        assert len(app._policy_context().grants) == 1
        assert "Allow all" in str(app.query_one("#status-label", Label).render())
        assert app.query_one("#status-bar").has_class("visible")


@pytest.mark.parametrize("command", ["/new", "/clear", "load"])
async def test_permissions_reset_on_session_change(tmp_path, command):
    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        app._allow_all = True
        app._conversation_rules = frozenset({make_rule("shell_execute", "git status", app._cwd)})
        app._call_grant = make_rule("write_file", "out", app._cwd)
        if command == "load":
            await app._load_session(new_session("m", endpoint=app._endpoint))
        else:
            await app._dispatch_command(command)
        await pilot.pause()
        assert not app._allow_all and app._conversation_rules is None and app._call_grant is None


async def test_rules_removal_cancel_and_later_dialog_grants(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    saved = Grant("fetch_url", origin="https://example.com:443")
    config.write_text("allowed_tools = " + json.dumps([saved.token()]))
    before = config.read_bytes()
    app = app_for(tmp_path, config=config)
    async with app.run_test() as pilot:
        await app._dispatch_command("/permissions")
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, PermissionsModal)
        assert "saved default" in modal._row(saved)
        modal.rules.clear()
        await pilot.click("#permissions-cancel")
        assert saved in app._policy_context().grants
        app._apply_permissions((frozenset(), False))
        assert saved not in app._policy_context().grants
        monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value="allow-session"))
        assert await app._tool_allowed("fetch_url", '{"url":"https://other.test"}')
        assert Grant("fetch_url", origin="https://other.test:443") in app._policy_context().grants
        assert config.read_bytes() == before
        await app._dispatch_command("/new")
        assert saved in app._policy_context().grants


async def test_origin_bypass_and_disabled_restores_prompt(tmp_path, monkeypatch):
    app = app_for(tmp_path)
    confirm = AsyncMock(return_value=False)
    monkeypatch.setattr(app, "_confirm", confirm)
    turn = ChatTurn(app._session, [], "x")
    app._allow_all = True
    assert await app._approve_origin(turn, "http://127.0.0.1:80", True)
    confirm.assert_not_called()
    app._allow_all = False
    assert not await app._approve_origin(turn, "http://127.0.0.1:80", True)
    confirm.assert_awaited_once()


async def test_allow_all_execution_redirect_private_and_read_limits(tmp_path, monkeypatch):
    app = app_for(tmp_path, remote=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("line\n" * 300)
    requested = []

    def handle(request):
        requested.append(request)
        if request.headers["host"] == "example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        return httpx.Response(200, text="private response")

    monkeypatch.setattr(tools, "_resolve", lambda host, port: ["93.184.216.34"])
    monkeypatch.setattr(tools, "_fetch_transport", httpx.MockTransport(handle))
    confirm = AsyncMock(return_value=False)
    monkeypatch.setattr(app, "_confirm", confirm)
    turn = ChatTurn(app._session, [], "x")
    app._allow_all = True
    assert await app._tool_allowed("read_file", json.dumps({"path": str(outside)}))
    result = await app._run_tool(turn, "read_file", json.dumps({"path": str(outside)}))
    assert result.count("line\n") == 200 and "offset=200" in result
    assert await app._tool_allowed("fetch_url", '{"url":"https://example.com"}')
    assert (
        await app._run_tool(turn, "fetch_url", '{"url":"https://example.com"}')
        == "private response"
    )
    assert len(requested) == 2 and requested[-1].url.host == "127.0.0.1"
    # Bypass is only approval: URL credentials are still invalid.
    assert isinstance(
        await app._run_tool(turn, "fetch_url", '{"url":"http://u:p@example.com"}'), ToolFailure
    )
    confirm.assert_not_called()
    app._allow_all = False
    app._conversation_rules = frozenset({Grant("fetch_url", origin="https://example.com:443")})
    requested.clear()
    assert isinstance(
        await app._run_tool(turn, "fetch_url", '{"url":"https://example.com"}'), ToolFailure
    )
    assert len(requested) == 1
    confirm.assert_awaited_once()


async def test_remote_rules_are_labelled_inactive_and_editor_blocks_during_generation(tmp_path):
    app = app_for(tmp_path, remote=True)
    rule = make_rule("read_file", str(tmp_path), app._cwd)
    async with app.run_test() as pilot:
        app._apply_permissions((frozenset({rule}), False))
        assert decide(
            "read_file", json.dumps({"path": str(tmp_path / "a")}), app._policy_context()
        ).prompt
        await app._dispatch_command("/permissions")
        await pilot.pause()
        assert "inactive: remote reads always ask" in app.screen._row(rule)
        await pilot.press("escape")
        app.is_generating = True
        await app._dispatch_command("/permissions")
        assert not isinstance(app.screen, PermissionsModal)
        app.is_generating = False


async def test_model_switch_resets_and_session_save_does_not_persist_permissions(
    tmp_path, monkeypatch
):
    from drove.chat_models import ChatModel
    from drove.sessions import load_session

    app = app_for(tmp_path)
    async with app.run_test() as pilot:
        app._apply_permissions(
            (frozenset({make_rule("shell_execute", "git status", app._cwd)}), True)
        )
        await app._dispatch_command("/save")
        # A conversation file contains history, not permission state.
        session = load_session(next(tmp_path.rglob("*.json")))
        assert not hasattr(session, "allow_all")
        monkeypatch.setattr(
            "drove.tui.fetch_models_async",
            AsyncMock(return_value=[ChatModel("m"), ChatModel("other")]),
        )
        monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value="other"))
        app.run_worker(app._dispatch_command("/model"))
        await pilot.pause()
        assert app._model == "other"
        assert not app._allow_all and app._conversation_rules is None


async def test_allow_always_still_joins_editor_override(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("")
    app = app_for(tmp_path, config=config)
    async with app.run_test():
        app._apply_permissions((frozenset(), False))
        monkeypatch.setattr(app, "push_screen_wait", AsyncMock(return_value="allow-always"))
        assert await app._tool_allowed("fetch_url", '{"url":"https://example.com"}')
        grant = Grant("fetch_url", origin="https://example.com:443")
        assert grant in app._policy_context().grants
        assert grant in app._configured_grants()


async def test_inspecting_selected_rule_can_apply_but_changed_scope_requires_save(tmp_path):
    app = app_for(tmp_path)
    rule = make_rule("shell_execute", "git status", app._cwd)
    async with app.run_test(size=(100, 50)) as pilot:
        app._apply_permissions((frozenset({rule}), False))
        await app._dispatch_command("/permissions")
        await pilot.pause()
        modal = app.screen
        options = modal.query_one("#permission-rules", OptionList)
        options.highlighted = 0
        options.focus()
        await pilot.press("enter")
        modal.query_one("#permission-allow-all", Switch).value = True
        await pilot.click("#permissions-apply")
        await pilot.pause()
        assert not isinstance(app.screen, PermissionsModal)
        assert app._allow_all and app._policy_context().grants == frozenset({rule})
        await app._dispatch_command("/permissions")
        await pilot.pause()
        modal = app.screen
        options = modal.query_one("#permission-rules", OptionList)
        options.highlighted = 0
        options.focus()
        await pilot.press("enter")
        modal.query_one("#permission-scope", Input).value = "git diff"
        await pilot.click("#permissions-apply")
        assert isinstance(app.screen, PermissionsModal)
        assert "Save the rule" in str(modal.query_one("#permission-error", Label).render())
        await pilot.press("escape")
        assert app._policy_context().grants == frozenset({rule})
