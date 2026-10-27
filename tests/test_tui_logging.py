"""Logging stays off the real terminal while a Textual app runs, and on stderr otherwise."""

import io
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from click.testing import Result
from textual._log import LogGroup
from textual.logging import TextualHandler
from typer.testing import CliRunner

from drove.chat_models import ChatModel, fetch_models_async
from drove.cli.main import app
from drove.cli.tui_logging import textual_logging
from drove.sessions import new_session
from drove.tui import ChatApp

HTTP_LINE = 'HTTP Request: GET http://x.test/v1/models "HTTP/1.1 200 OK"'


@contextmanager
def bare_root() -> Iterator[logging.Logger]:
    """An unconfigured root logger, restored afterwards.

    pytest attaches its capture handlers to root around the test body (after fixtures run), and
    any root handler turns the CLI's basicConfig into a no-op — so clear root inside the body.
    """
    root = logging.getLogger()
    saved, level = root.handlers[:], root.level
    for handler in saved:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        yield root
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved:
            root.addHandler(handler)
        root.setLevel(level)


def invoke(args: list[str]) -> Result:
    """Run the CLI from a bare root; its basicConfig binds to the runner's stderr."""
    with bare_root():
        return CliRunner().invoke(app, args)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    observe = tmp_path / "observe"
    observe.mkdir()
    path = tmp_path / "config.toml"
    path.write_text(
        f'models_dir = "{models}"\nobserve_dir = "{observe}"\n'
        f'sessions_dir = "{tmp_path / "sessions"}"\n'
    )
    for var in ("DROVE_MODELS_DIR", "DROVE_OBSERVE_DIR", "DROVE_SESSIONS_DIR"):
        monkeypatch.delenv(var, raising=False)
    return path


def emitting_run(seen: list[list[logging.Handler]]) -> Callable[..., None]:
    """Stand in for App.run: note the root handlers and log like a live chat would."""

    def run(*args: object, **kwargs: object) -> None:
        seen.append(logging.getLogger().handlers[:])
        logging.getLogger("httpx").info(HTTP_LINE)
        logging.getLogger("drove.test").warning("sentinel warning")
        logging.getLogger("drove.test").debug("sentinel debug")

    return run


def assert_routed_to_textual(seen: list[list[logging.Handler]], output: str) -> None:
    assert len(seen) == 1
    assert len(seen[0]) == 1 and isinstance(seen[0][0], TextualHandler)
    assert "HTTP Request" not in output
    assert "sentinel" not in output


@pytest.mark.parametrize("verbose", [False, True])
def test_chat_keeps_logging_off_the_terminal(cfg: Path, verbose: bool) -> None:
    seen: list[list[logging.Handler]] = []
    args = ["-c", str(cfg), *(["-v"] if verbose else []), "chat", "a", "-e", "http://x.test"]
    with patch.object(ChatApp, "run", emitting_run(seen)):
        result = invoke(args)
    assert result.exit_code == 0, result.output
    assert_routed_to_textual(seen, result.output)


def test_initial_model_picker_keeps_logging_off_the_terminal(
    cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("drove.cli.main.sys", MagicMock())
    seen: list[list[logging.Handler]] = []
    with (
        patch("drove.chat_models.fetch_models", return_value=[ChatModel("a")]),
        patch("drove.model_picker.ModelPickerApp") as picker,
    ):
        picker.return_value.run.side_effect = emitting_run(seen)
        result = invoke(["-c", str(cfg), "chat"])
    assert result.exit_code == 0, result.output
    assert_routed_to_textual(seen, result.output)


def test_observe_tui_keeps_logging_off_the_terminal(cfg: Path) -> None:
    seen: list[list[logging.Handler]] = []
    with patch("drove.observe_tui.ObserveApp.run", emitting_run(seen)):
        result = invoke(["-c", str(cfg), "observe"])
    assert result.exit_code == 0, result.output
    assert_routed_to_textual(seen, result.output)


def test_quant_picker_keeps_logging_off_the_terminal() -> None:
    from drove.cli.models import _prompt_quant_choice

    terminal = io.StringIO()
    seen: list[list[logging.Handler]] = []
    with (
        bare_root() as root,
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("drove.cli.quant_picker.QuantPicker") as picker,
    ):
        root.addHandler(logging.StreamHandler(terminal))
        root.setLevel(logging.INFO)
        picker.return_value.run.side_effect = emitting_run(seen)
        assert _prompt_quant_choice([], "textual-dark") is None
    assert_routed_to_textual(seen, terminal.getvalue())


@pytest.mark.parametrize("command", ["server", "serve"])
@pytest.mark.parametrize("verbose", [False, True])
def test_server_still_logs_to_stderr(cfg: Path, command: str, verbose: bool) -> None:
    seen: list[list[logging.Handler]] = []
    args = ["-c", str(cfg), *(["-v"] if verbose else []), command]
    with (
        patch("drove.proxy.create_app"),
        patch("drove.cli.server.uvicorn.run", side_effect=emitting_run(seen)),
    ):
        result = invoke(args)
    assert result.exit_code == 0, result.output
    assert not any(isinstance(h, TextualHandler) for h in seen[0])
    assert f"INFO httpx: {HTTP_LINE}" in result.stderr
    assert "WARNING drove.test: sentinel warning" in result.stderr
    assert ("DEBUG drove.test: sentinel debug" in result.stderr) is verbose


def _logged(log: Mock) -> str:
    """Text of every record Textual received through its logging group."""
    return "\n".join(
        " ".join(map(str, call.args[3:]))
        for call in log.call_args_list
        if call.args[0] is LogGroup.LOGGING
    )


@pytest.mark.parametrize("verbose", [False, True])
async def test_live_chat_requests_reach_textual_log_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verbose: bool
) -> None:
    """Real httpx requests from a running chat: streamed reply, catalog fetch, resumed session."""

    def reply(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a"}]})
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')

    real_client = httpx.AsyncClient

    def client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(reply))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    session = new_session("a")
    session.messages.append({"role": "user", "content": "earlier"})
    log = Mock()
    # Stand in for connected devtools; _log is bound into app.log at construction.
    monkeypatch.setattr(ChatApp, "_is_devtools_connected", True)
    monkeypatch.setattr(ChatApp, "_log", log)
    chat = ChatApp("http://x.test", "a", tmp_path, None, resume_session=session)
    terminal = io.StringIO()
    with bare_root() as root:
        root.addHandler(logging.StreamHandler(terminal))
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
        with textual_logging():
            async with chat.run_test() as pilot:
                assert chat._history[-1]["content"] == "earlier"
                assert [event async for event in chat._stream_chat(chat._history)]
                assert await fetch_models_async("http://x.test", None) == [ChatModel("a")]
                logging.getLogger("drove.test").debug("sentinel debug")
                await pilot.pause()
    assert terminal.getvalue() == ""
    logged = _logged(log)
    assert "POST http://x.test/v1/chat/completions" in logged
    assert "GET http://x.test/v1/models" in logged
    assert ("sentinel debug" in logged) is verbose


def test_textual_logging_is_silent_without_an_active_app() -> None:
    terminal = io.StringIO()
    with bare_root() as root, patch("sys.stderr", io.StringIO()) as err:
        root.addHandler(logging.StreamHandler(terminal))
        with textual_logging():
            logging.getLogger("drove.test").error("no app yet")
    assert terminal.getvalue() == "" and err.getvalue() == ""


def test_textual_logging_reuses_the_cli_formatter() -> None:
    original = logging.StreamHandler(io.StringIO())
    original.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    with bare_root() as root:
        root.addHandler(original)
        with textual_logging():
            (handler,) = root.handlers
            assert handler.formatter is original.formatter


@pytest.mark.parametrize("exc", [None, RuntimeError, KeyboardInterrupt])
def test_textual_logging_restores_handlers(exc: type[BaseException] | None) -> None:
    first, second = logging.StreamHandler(io.StringIO()), logging.NullHandler()
    with bare_root() as root:
        root.addHandler(first)
        root.addHandler(second)
        root.setLevel(logging.DEBUG)
        try:
            with textual_logging():
                (handler,) = root.handlers
                assert isinstance(handler, TextualHandler)
                if exc is not None:
                    raise exc
        except BaseException as caught:
            assert exc is not None and isinstance(caught, exc)
        assert root.handlers == [first, second]
        assert root.level == logging.DEBUG
    assert not first.stream.closed


def test_textual_logging_with_no_prior_handlers() -> None:
    with bare_root() as root:
        with textual_logging():
            assert len(root.handlers) == 1
        assert root.handlers == []
