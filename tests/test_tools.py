"""Tests for tool execution: failures, streamed shell output, limits, fetch safety."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import httpx
import pytest

from drove import tools
from drove.tool_policy import TOOL_OUTPUT_LIMIT, TRUNCATION_NOTICE
from drove.tools import ToolContext, ToolFailure, execute_tool


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
    # Below the 1-second floor, so the clamp applies.
    result = _shell("sleep 30", timeout=0.3)

    assert isinstance(result, ToolFailure)
    assert result == "Error: command timed out after 1 seconds"
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

    # Both pipes are drained to the end (the command exits 0), but only a capped
    # share of each stream is kept.
    assert result.endswith("Return code: 0")
    assert result.count("o") >= 30_000
    assert result.count("e") >= 10_000
    assert result.count(TRUNCATION_NOTICE) == 2
    assert len(result.encode()) <= TOOL_OUTPUT_LIMIT


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


# ── Output cap on every tool (AC1) ──────────────────────────────────────────────


@pytest.fixture
def no_background_tasks():
    tools._background_tasks.clear()
    yield tools._background_tasks
    tools._background_tasks.clear()


def _within_cap(result: str) -> bool:
    return len(result.encode("utf-8")) <= TOOL_OUTPUT_LIMIT


def test_every_tool_result_is_capped(tmp_path: Path, no_background_tasks, fake_net) -> None:
    big = tmp_path / "big"
    big.mkdir()
    for i in range(3000):
        (big / f"{'n' * 40}-é-{i}.txt").write_text("needle")
    (tmp_path / "huge.txt").write_text("€" * 200_000)
    fake_net.body = b"x" * 300_000
    args = {
        "read_file": {"path": str(tmp_path / "huge.txt"), "limit": 10},
        "list_files": {"path": str(big)},
        "search_files": {"path": str(big), "content_query": "needle"},
        "shell_execute": {"command": _python("print('ü' * 300_000)")},
        "fetch_url": {"url": "https://example.com/"},
        "no_such_tool": {"padding": "y" * 200_000},
    }

    for name, arguments in args.items():
        result = execute_tool(name, json.dumps(arguments))
        assert _within_cap(result), name
        assert TRUNCATION_NOTICE in result or name == "no_such_tool", name

    assert _within_cap(execute_tool("shell_execute", "{" + "x" * 200_000))


def test_background_task_output_is_capped(no_background_tasks) -> None:
    reply = execute_tool(
        "shell_execute", json.dumps({"command": "yes | head -c 500000", "background": True})
    )
    task_id = reply.rsplit(" ", 1)[-1]
    deadline = time.monotonic() + 10
    while "status: running" in (status := execute_tool("check_background_task", _task(task_id))):
        assert time.monotonic() < deadline
        time.sleep(0.05)

    assert "status: finished" in status
    assert _within_cap(status)
    assert TRUNCATION_NOTICE in status


def _task(task_id: str) -> str:
    return json.dumps({"task_id": task_id})


# ── Shell timeouts and process groups (AC1) ─────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 30),
        (True, 30),
        ("10", 30),
        (float("nan"), 30),
        (float("inf"), 30),
        (-5, 1),
        (0, 1),
        (0.2, 1),
        (1, 1),
        (45, 45),
        (2.5, 2.5),
        (600, 600),
        (10_000, 600),
    ],
)
def test_timeout_is_clamped(value: object, expected: float) -> None:
    assert tools._clamp_timeout(value) == expected


def test_missing_timeout_defaults_to_30_seconds(monkeypatch) -> None:
    seen: list[float] = []
    real = tools._run_streaming

    def spy(command, timeout, on_output):
        seen.append(timeout)
        return real(command, timeout, on_output)

    monkeypatch.setattr(tools, "_run_streaming", spy)
    execute_tool("shell_execute", json.dumps({"command": "true", "timeout": None}))
    execute_tool("shell_execute", json.dumps({"command": "true"}))

    assert seen == [30, 30]


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError, ProcessLookupError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return state != "Z"


def _wait_gone(pid: int, seconds: float = 5) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    # The shell starts a child that starts a grandchild, then everyone sleeps.
    command = f"sh -c 'sleep 60 & echo $! > {pid_file}; wait' & wait"
    pid = None
    try:
        result = _shell(command, timeout=1)
        pid = int(pid_file.read_text())

        assert result == "Error: command timed out after 1 seconds"
        assert _wait_gone(pid)
    finally:
        if pid is not None and _alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_descendants_left_behind_by_a_finished_command_are_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "stray.pid"
    pid = None
    try:
        result = _shell(f"sleep 60 >/dev/null 2>&1 & echo $! > {pid_file}", timeout=10)
        pid = int(pid_file.read_text())

        assert result.endswith("Return code: 0")
        assert _wait_gone(pid)
    finally:
        if pid is not None and _alive(pid):
            os.kill(pid, signal.SIGKILL)


def test_background_tasks_time_out_with_their_group(no_background_tasks) -> None:
    reply = execute_tool(
        "shell_execute", json.dumps({"command": "sleep 30", "background": True, "timeout": 1})
    )
    task_id = reply.rsplit(" ", 1)[-1]
    deadline = time.monotonic() + 5
    while "status: running" in (status := execute_tool("check_background_task", _task(task_id))):
        assert time.monotonic() < deadline
        time.sleep(0.05)

    assert isinstance(status, str)
    assert "status: timed out" in status
    assert "timed out after 1 seconds" in status


def test_running_background_tasks_are_limited(no_background_tasks) -> None:
    start = json.dumps({"command": "sleep 30", "background": True, "timeout": 2})
    replies = [execute_tool("shell_execute", start) for _ in range(tools._MAX_BACKGROUND_RUNNING)]
    refused = execute_tool("shell_execute", start)

    assert all(r.startswith("Started background task with ID: ") for r in replies)
    assert isinstance(refused, ToolFailure)
    assert "already running" in refused


def test_finished_background_tasks_expire_and_are_bounded(no_background_tasks) -> None:
    now = time.monotonic()
    for i in range(tools._MAX_BACKGROUND_FINISHED + 5):
        no_background_tasks[f"t{i}"] = {
            "command": "true",
            "status": "finished",
            "output": "",
            "error": "",
            "finished_at": now - i,  # t0 is the newest
        }
    no_background_tasks["old"] = {
        "command": "true",
        "status": "finished",
        "output": "",
        "error": "",
        "finished_at": now - tools._BACKGROUND_RETENTION_SECONDS - 1,
    }

    expired = execute_tool("check_background_task", _task("old"))

    assert isinstance(expired, ToolFailure) and "not found" in expired
    assert set(no_background_tasks) == {f"t{i}" for i in range(tools._MAX_BACKGROUND_FINISHED)}


# ── read_file limits ────────────────────────────────────────────────────────────


def test_read_file_defaults_to_200_lines_and_says_where_to_continue(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("".join(f"line {i}\n" for i in range(500)))

    first = execute_tool("read_file", json.dumps({"path": str(path)}))
    second = execute_tool("read_file", json.dumps({"path": str(path), "offset": 200}))
    tail = execute_tool("read_file", json.dumps({"path": str(path), "offset": 400, "limit": 500}))

    assert first.startswith("line 0\n") and "line 199\n" in first and "line 200" not in first
    assert first.endswith("[more lines: continue with offset=200]")
    assert second.startswith("line 200\n")
    assert tail.endswith("line 499\n")


def test_read_file_bounds_a_single_huge_line(tmp_path: Path) -> None:
    path = tmp_path / "one-line.txt"
    path.write_text("q" * 5_000_000)

    result = execute_tool("read_file", json.dumps({"path": str(path)}))

    assert _within_cap(result)
    assert result.endswith(TRUNCATION_NOTICE)


@pytest.mark.parametrize(
    "extra", [{"offset": -1}, {"limit": -2}, {"limit": "10"}, {"offset": True}, {"path": 5}]
)
def test_read_file_rejects_bad_arguments(tmp_path: Path, extra: dict) -> None:
    path = tmp_path / "f.txt"
    path.write_text("x")

    result = execute_tool("read_file", json.dumps({"path": str(path), **extra}))

    assert isinstance(result, ToolFailure)


def test_reads_outside_the_approved_roots_fail_at_execution(tmp_path: Path) -> None:
    cwd, outside = tmp_path / "cwd", tmp_path / "out"
    cwd.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (cwd / "link.txt").symlink_to(outside / "secret.txt")
    (cwd / "linkdir").symlink_to(outside)
    context = ToolContext(cwd=cwd.resolve(), read_roots=(cwd.resolve(),))

    via_link = execute_tool("read_file", json.dumps({"path": "link.txt"}), context=context)
    listing = execute_tool("list_files", json.dumps({"path": "linkdir"}), context=context)
    inside = execute_tool("list_files", json.dumps({}), context=context)

    assert isinstance(via_link, ToolFailure) and "outside the approved" in via_link
    assert isinstance(listing, ToolFailure) and "outside the approved" in listing
    assert inside == "link.txt\nlinkdir/"


def test_relative_paths_resolve_against_the_context_cwd(tmp_path: Path) -> None:
    (tmp_path / "here.txt").write_text("found")
    context = ToolContext(cwd=tmp_path, read_roots=(tmp_path.resolve(),))

    assert execute_tool("read_file", json.dumps({"path": "here.txt"}), context=context) == "found"


# ── search_files (AC6) ──────────────────────────────────────────────────────────


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    names: list[str] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        names.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: pytest.fail("read_text"))
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: pytest.fail("read_bytes"))
    return names


def _search(path: Path, context: ToolContext | None = None, **args) -> str:
    return execute_tool("search_files", json.dumps({"path": str(path), **args}), context=context)


def test_search_checks_the_name_before_reading_content(tmp_path: Path, opened) -> None:
    (tmp_path / "match.py").write_text("needle")
    (tmp_path / "other.txt").write_text("needle")
    (tmp_path / "weights.gguf").write_bytes(b"needle" * 1000)
    opened.clear()

    result = _search(tmp_path, name_pattern="*.py", content_query="needle")

    assert result == "match.py"
    assert opened == ["match.py"]


def test_search_skips_binary_oversized_and_undecodable_files(tmp_path: Path, opened) -> None:
    (tmp_path / "text.txt").write_text("has needle")
    (tmp_path / "binary.bin").write_bytes(b"needle\x00\x01")
    (tmp_path / "latin1.txt").write_bytes("needle café".encode("latin-1"))
    with (tmp_path / "big.txt").open("w") as f:
        f.write("needle" + "x" * tools._SEARCH_MAX_FILE_BYTES)
    opened.clear()

    result = _search(tmp_path, content_query="needle")

    assert result.splitlines()[0] == "text.txt"
    assert "[skipped 3 binary, oversized or undecodable files]" in result
    assert "big.txt" not in opened  # the size check comes before any read


def test_name_only_search_lists_binary_files_without_reading_them(tmp_path: Path, opened) -> None:
    (tmp_path / "weights.gguf").write_bytes(b"\x00" * 10)
    opened.clear()

    assert _search(tmp_path, name_pattern="*.gguf") == "weights.gguf"
    assert opened == []


def test_search_skips_content_outside_the_approved_roots(tmp_path: Path) -> None:
    cwd, outside = tmp_path / "cwd", tmp_path / "out"
    cwd.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("needle")
    (cwd / "link.txt").symlink_to(outside / "secret.txt")
    (cwd / "own.txt").write_text("needle")
    context = ToolContext(cwd=cwd.resolve(), read_roots=(cwd.resolve(),))

    result = _search(cwd, context, content_query="needle")

    assert result.splitlines()[0] == "own.txt"
    assert "link.txt" not in result.splitlines()
    assert "outside the approved directories" in result


# ── fetch_url: redirects, SSRF and pinning (AC1) ────────────────────────────────


class FakeNet:
    """A local stand-in for DNS and the network: no request leaves the process."""

    def __init__(self) -> None:
        self.dns: dict[str, list[str]] = {"example.com": ["93.184.216.34"]}
        self.routes: dict[str, httpx.Response] = {}
        self.body = b"hello"
        self.requests: list[httpx.Request] = []
        self.lookups: list[str] = []

    def resolve(self, host: str, port: int) -> list[str]:
        self.lookups.append(host)
        if host not in self.dns:
            raise OSError(f"cannot resolve {host}")
        return self.dns[host]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.headers['host']}{request.url.raw_path.decode()}"
        if key in self.routes:
            return self.routes[key]
        return httpx.Response(200, content=self.body)


@pytest.fixture
def fake_net(monkeypatch) -> FakeNet:
    net = FakeNet()
    monkeypatch.setattr(tools, "_resolve", net.resolve)
    monkeypatch.setattr(tools, "_fetch_transport", httpx.MockTransport(net.handle))
    return net


def _fetch(url: str, context: ToolContext | None = None) -> str:
    return execute_tool("fetch_url", json.dumps({"url": url}), context=context)


def _redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location})


def test_fetch_connects_to_the_vetted_address_keeping_host_and_sni(fake_net) -> None:
    assert _fetch("https://example.com/page?q=1") == "hello"

    [request] = fake_net.requests
    assert request.url.host == "93.184.216.34"
    assert request.url.raw_path == b"/page?q=1"
    assert request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"


def test_fetch_resolves_once_per_hop_so_dns_cannot_rebind(fake_net) -> None:
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    fake_net.resolve = lambda host, port: next(answers)  # type: ignore[method-assign]
    tools._resolve = fake_net.resolve  # the fixture restores it

    assert _fetch("https://example.com/") == "hello"
    assert [r.url.host for r in fake_net.requests] == ["93.184.216.34"]


def test_fetch_follows_up_to_five_redirects(fake_net) -> None:
    for i in range(5):
        fake_net.routes[f"example.com/{i}"] = _redirect(f"/{i + 1}")

    assert _fetch("https://example.com/0") == "hello"
    assert len(fake_net.requests) == 6


def test_fetch_refuses_a_sixth_redirect(fake_net) -> None:
    for i in range(6):
        fake_net.routes[f"example.com/{i}"] = _redirect(f"https://example.com/{i + 1}", 301)

    result = _fetch("https://example.com/0")

    assert isinstance(result, ToolFailure) and "too many redirects" in result
    assert len(fake_net.requests) == 6


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.1.2.3:8080/",
        "http://[::1]/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[::]/",
        "http://0.0.0.0/",
        "http://[::ffff:10.0.0.1]/",
    ],
)
def test_fetch_refuses_internal_addresses(fake_net, url: str) -> None:
    result = _fetch(url)

    assert isinstance(result, ToolFailure) and "internal" in result
    assert fake_net.requests == []


@pytest.mark.parametrize("answers", [["10.0.0.7"], ["93.184.216.34", "127.0.0.1"]])
def test_fetch_refuses_a_name_resolving_to_an_internal_address(fake_net, answers) -> None:
    fake_net.dns["intranet.example"] = answers

    result = _fetch("http://intranet.example/")

    assert isinstance(result, ToolFailure) and "internal" in result
    assert fake_net.requests == []


def test_fetch_refuses_a_redirect_to_an_internal_address(fake_net) -> None:
    fake_net.routes["example.com/"] = _redirect("http://169.254.169.254/latest")

    result = _fetch("https://example.com/")

    assert isinstance(result, ToolFailure) and "internal" in result
    assert len(fake_net.requests) == 1


def test_internal_access_needs_approval_for_that_host(fake_net) -> None:
    fake_net.dns["intranet.example"] = ["10.0.0.7"]
    asked: list[tuple[str, bool]] = []

    def approve(host: str, internal: bool) -> bool:
        asked.append((host, internal))
        return host == "intranet.example"

    context = ToolContext(cwd=Path("/"), approved_hosts={"intranet.example"}, approve_host=approve)
    granted = ToolContext(cwd=Path("/"), approved_hosts={"10.0.0.8"}, internal_hosts={"10.0.0.8"})

    assert _fetch("http://intranet.example/", context) == "hello"
    assert asked == [("intranet.example", True)]
    assert fake_net.requests[-1].url.host == "10.0.0.7"
    assert _fetch("http://10.0.0.8/", granted) == "hello"


def test_a_redirect_to_another_host_needs_its_own_approval(fake_net) -> None:
    fake_net.dns["other.org"] = ["93.184.216.35"]
    fake_net.routes["example.com/"] = _redirect("https://other.org/landing")
    asked: list[tuple[str, bool]] = []

    def deny(host: str, internal: bool) -> bool:
        asked.append((host, internal))
        return False

    context = ToolContext(cwd=Path("/"), approved_hosts={"example.com"}, approve_host=deny)
    no_prompt = ToolContext(cwd=Path("/"), approved_hosts={"example.com"})

    denied = _fetch("https://example.com/", context)
    assert isinstance(denied, ToolFailure) and "other.org" in denied
    assert asked == [("other.org", False)]
    assert isinstance(_fetch("https://example.com/", no_prompt), ToolFailure)
    assert len(fake_net.requests) == 2  # never the redirect target


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/", "https://user:pw@example.com/", "nonsense", ""],
)
def test_fetch_rejects_unsupported_urls(fake_net, url: str) -> None:
    assert isinstance(_fetch(url), ToolFailure)
    assert fake_net.requests == []


def test_fetch_reports_dns_failures_and_http_errors(fake_net) -> None:
    fake_net.routes["example.com/missing"] = httpx.Response(404)

    unresolved = _fetch("https://nowhere.invalid/")
    missing = _fetch("https://example.com/missing")

    assert isinstance(unresolved, ToolFailure) and "resolve" in unresolved
    assert isinstance(missing, ToolFailure) and "404" in missing
