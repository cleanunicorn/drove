"""CLI entry-point selection and scripted invocation behavior."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from drove.chat_models import CatalogError, ChatModel
from drove.cli.main import app


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(False, True), (True, False), (False, False)])
def test_missing_model_requires_terminal_before_network(stdin_tty: bool, stdout_tty: bool) -> None:
    with (
        patch("sys.stdin.isatty", return_value=stdin_tty),
        patch("sys.stdout.isatty", return_value=stdout_tty),
        patch("drove.chat_models.fetch_models") as fetch,
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(app, ["chat"])
    assert result.exit_code == 1
    assert "drove chat MODEL" in result.output
    fetch.assert_not_called()
    chat.assert_not_called()


def test_explicit_model_keeps_endpoint_key_and_bypasses_picker() -> None:
    with patch("drove.chat_models.fetch_models") as fetch, patch("drove.tui.ChatApp") as chat:
        result = CliRunner().invoke(
            app, ["chat", "specific", "-e", "https://example.test/v1", "-k", "key"]
        )
    assert result.exit_code == 0
    fetch.assert_not_called()
    assert chat.call_args.kwargs["model"] == "specific"
    assert chat.call_args.kwargs["base_url"] == "https://example.test"
    assert chat.call_args.kwargs["api_key"] == "key"
    chat.return_value.run.assert_called_once()


@pytest.mark.parametrize("selected", [None, "chosen"])
def test_interactive_selection_and_clean_cancel(selected: str | None, monkeypatch) -> None:
    # CliRunner replaces streams; provide TTY streams at the selection guard
    # through the module's sys proxy without altering its terminal capture.
    monkeypatch.setattr("drove.cli.main.sys", MagicMock())
    with (
        patch("drove.chat_models.fetch_models", return_value=[ChatModel("chosen")]),
        patch("drove.model_picker.ModelPickerApp") as picker,
        patch("drove.tui.ChatApp") as chat,
    ):
        picker.return_value.run.return_value = selected
        result = CliRunner().invoke(app, ["chat"])
    assert result.exit_code == 0
    if selected is None:
        chat.assert_not_called()
    else:
        assert chat.call_args.kwargs["model"] == selected


def test_catalog_error_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr("drove.cli.main.sys", MagicMock())
    with patch(
        "drove.chat_models.fetch_models", side_effect=CatalogError("No chat models available.")
    ):
        result = CliRunner().invoke(app, ["chat"])
    assert result.exit_code == 1
    assert "No chat models" in result.output


@pytest.mark.parametrize(
    "default,argument,expected,picker_used",
    [
        ("preferred", None, "preferred", False),
        ("preferred", "explicit", "explicit", False),
        (None, None, "picked", True),
    ],
)
def test_chat_model_precedence(tmp_path: Path, default, argument, expected, picker_used) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f'default_model = "{default}"\n' if default else "")
    args = ["-c", str(config), "chat"]
    if argument is not None:
        args.append(argument)
    with (
        patch("drove.cli.main.sys", MagicMock()),
        patch("drove.cli.main._select_model_from_endpoint", return_value="picked") as picker,
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert chat.call_args.kwargs["model"] == expected
        assert picker.called == picker_used
        chat.return_value.run.assert_called_once()


def test_chat_default_endpoint_and_resume(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f'default_model = "remote-id"\nsessions_dir = "{tmp_path}"\n')
    session = Mock()
    with (
        patch("drove.cli.main._select_model_from_endpoint") as picker,
        patch("drove.sessions.latest_session", return_value=session) as latest,
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(
            app,
            [
                "-c",
                str(config),
                "chat",
                "--endpoint",
                "https://example.test/v1/",
                "--api-key",
                "test-key",
                "--resume",
            ],
        )
        assert result.exit_code == 0, result.output
        picker.assert_not_called()
        latest.assert_called_once_with(tmp_path, "remote-id", endpoint="https://example.test")
        assert chat.call_args.kwargs["base_url"] == "https://example.test"
        assert chat.call_args.kwargs["api_key"] == "test-key"
        assert chat.call_args.kwargs["resume_session"] is session


def test_chat_picker_cancellation_without_default(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("")
    with (
        patch("drove.cli.main.sys", MagicMock()),
        patch("drove.cli.main._select_model_from_endpoint", return_value=None),
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(app, ["-c", str(config), "chat"])
        assert result.exit_code == 0
        chat.assert_not_called()


def test_resume_skips_corrupt_newest_session(tmp_path: Path) -> None:
    from drove.sessions import new_session, save_session

    config = tmp_path / "config.toml"
    sessions = tmp_path / "sessions"
    config.write_text(f'default_model = "remote-id"\nsessions_dir = "{sessions}"\n')
    saved = new_session("remote-id", endpoint="https://example.test")
    saved.messages.append({"role": "user", "content": "resume me"})
    save_session(sessions, saved)
    (sessions / "remote-id" / "20990101-000000.json").write_text("{")
    with patch("drove.tui.ChatApp") as chat:
        result = CliRunner().invoke(
            app, ["-c", str(config), "chat", "--endpoint", "https://example.test/v1/", "--resume"]
        )
    assert result.exit_code == 0, result.output
    assert chat.call_args.kwargs["resume_session"].id == saved.id


def test_resume_never_selects_another_endpoint_or_unassigned_history(tmp_path: Path) -> None:
    from drove.sessions import new_session, save_session

    config = tmp_path / "config.toml"
    config.write_text(f'sessions_dir = "{tmp_path / "sessions"}"\n')
    for endpoint in ["https://private.test", None]:
        saved = new_session("shared-model", endpoint=endpoint)
        saved.messages.append({"role": "user", "content": "private conversation"})
        save_session(tmp_path / "sessions", saved)

    with patch("drove.tui.ChatApp") as chat:
        result = CliRunner().invoke(
            app,
            ["-c", str(config), "chat", "shared-model", "-e", "https://other.test", "-r"],
        )
    assert result.exit_code == 1, result.output
    assert "at this endpoint" in result.output
    assert "/sessions" in result.output
    chat.assert_not_called()


def test_resume_rejects_system_override_before_selecting_or_loading() -> None:
    with (
        patch("drove.cli.main._select_model_from_endpoint") as picker,
        patch("drove.sessions.latest_session") as latest,
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(app, ["chat", "--resume", "--system", "override"])
    assert result.exit_code == 2, result.output
    assert "saved system prompt is preserved" in " ".join(result.output.split())
    picker.assert_not_called()
    latest.assert_not_called()
    chat.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://example.test",
        "http://host:bad",
        "https://example.test/a\nb",
        "https://example.test/?",
        "https://example.test/#",
        "https://example.test/?token=secret",
        "https://example.test/#section",
    ],
)
def test_invalid_endpoint_is_reported_before_model_lookup(endpoint: str) -> None:
    with patch("drove.cli.main._select_model_from_endpoint") as picker:
        result = CliRunner().invoke(app, ["chat", "-e", endpoint])
    assert result.exit_code == 2, result.output
    assert "Invalid chat endpoint" in result.output
    assert "secret" not in result.output
    picker.assert_not_called()
