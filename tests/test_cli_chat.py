"""Chat model precedence without starting a real TUI or backend."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from drove.cli.main import app


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
        latest.assert_called_once_with(tmp_path, "remote-id")
        assert chat.call_args.kwargs["base_url"] == "https://example.test"
        assert chat.call_args.kwargs["api_key"] == "test-key"
        assert chat.call_args.kwargs["resume_session"] is session


def test_chat_picker_cancellation_without_default(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("")
    with (
        patch("drove.cli.main._select_model_from_endpoint", return_value=None),
        patch("drove.tui.ChatApp") as chat,
    ):
        result = CliRunner().invoke(app, ["-c", str(config), "chat"])
        assert result.exit_code == 1
        chat.assert_not_called()
