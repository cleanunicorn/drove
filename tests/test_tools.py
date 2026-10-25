"""Tests for tool execution: failure marking and streamed shell output."""

from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from drove.tools import ToolFailure, execute_tool


def _python(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _shell(command: str, timeout: float = 10, on_output=None) -> str:
    args = json.dumps({"command": command, "timeout": timeout})
    return execute_tool("shell_execute", args, on_output)


def test_successful_shell_keeps_the_result_format_and_is_not_a_failure() -> None:
    result = _shell("echo out; echo err >&2")

    assert result == "STDOUT:\nout\n\nSTDERR:\nerr\n\nReturn code: 0"
    assert not isinstance(result, ToolFailure)


def test_nonzero_exit_is_a_failure_with_the_same_text() -> None:
    result = _shell("echo partial; exit 3")

    assert isinstance(result, ToolFailure)
    assert result == "STDOUT:\npartial\n\nReturn code: 3"


def test_timeout_reaps_the_command_and_fails() -> None:
    started = time.monotonic()
    result = _shell("sleep 30", timeout=0.3)

    assert isinstance(result, ToolFailure)
    assert result == "Error: command timed out after 0.3 seconds"
    assert time.monotonic() - started < 5


def test_output_streams_before_the_command_exits(tmp_path: Path) -> None:
    release = tmp_path / "release"
    code = (
        "import os, sys, time\n"
        "sys.stdout.write('first'); sys.stdout.flush()\n"
        "sys.stderr.write('warn'); sys.stderr.flush()\n"
        f"while not os.path.exists({str(release)!r}): time.sleep(0.01)\n"
        "sys.stdout.write(' done')\n"
    )
    chunks: list[str] = []

    def on_output(text: str) -> None:
        chunks.append(text)
        if "first" in "".join(chunks) and "warn" in "".join(chunks):
            # The command is still blocked: it only exits once this file exists.
            release.touch()

    result = _shell(_python(code), on_output=on_output)

    assert result == "STDOUT:\nfirst done\nSTDERR:\nwarn\nReturn code: 0"
    assert release.exists()
    assert "".join(chunks).count("first") == 1


def test_multibyte_characters_split_across_reads_decode(tmp_path: Path) -> None:
    # 4095 ASCII bytes push the two-byte "é" across the 4096-byte read boundary.
    code = "import sys; sys.stdout.buffer.write(b'a' * 4095 + 'é\\r\\nz'.encode())"
    chunks: list[str] = []

    result = _shell(_python(code), on_output=chunks.append)

    assert result == f"STDOUT:\n{'a' * 4095}é\nz\nReturn code: 0"
    assert "".join(chunks) == f"{'a' * 4095}é\nz"


def test_large_output_on_both_streams_does_not_deadlock() -> None:
    code = (
        "import sys\n"
        "for _ in range(200): sys.stderr.write('e' * 1000); sys.stdout.write('o' * 1000)"
    )

    result = _shell(_python(code), timeout=20)

    assert result.count("o") >= 200_000
    assert result.count("e") >= 200_000
    assert result.endswith("Return code: 0")


def test_empty_output_without_callback() -> None:
    assert _shell("true") == "STDOUT:\n\nReturn code: 0"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", json.dumps({"path": "/definitely/missing"})),
        ("shell_execute", "{not json"),
        ("shell_execute", json.dumps({"command": ""})),
        ("no_such_tool", "{}"),
    ],
)
def test_tool_errors_are_failures(name: str, arguments: str) -> None:
    result = execute_tool(name, arguments)

    assert isinstance(result, ToolFailure)
    assert result.startswith("Error")


def test_file_content_that_starts_with_error_is_not_a_failure(tmp_path: Path) -> None:
    path = tmp_path / "log.txt"
    path.write_text("Error: this is just file content")

    result = execute_tool("read_file", json.dumps({"path": str(path)}))

    assert result == "Error: this is just file content"
    assert not isinstance(result, ToolFailure)


def test_background_shell_still_returns_a_task_id_immediately() -> None:
    started = time.monotonic()
    reply = execute_tool("shell_execute", json.dumps({"command": "sleep 2", "background": True}))

    assert reply.startswith("Started background task with ID: ")
    assert time.monotonic() - started < 1
