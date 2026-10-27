"""Tests for the chat tool approval policy and the shared output cap (no TUI needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drove.tool_policy import (
    TOOL_OUTPUT_LIMIT,
    TRUNCATION_NOTICE,
    Grant,
    PolicyContext,
    canonical_origin,
    cap_output,
    cap_tool_messages,
    decide,
    is_blocked_address,
    is_local_endpoint,
    parse_grant,
    read_roots,
)
from drove.tools import ToolFailure

READ_TOOLS = ["read_file", "list_files", "search_files"]


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """A working directory, a sibling outside it, and a symlink that escapes it."""
    cwd = tmp_path / "work"
    outside = tmp_path / "elsewhere"
    (cwd / "sub").mkdir(parents=True)
    outside.mkdir()
    (cwd / "a.txt").write_text("inside")
    (cwd / "sub" / "b.txt").write_text("nested")
    (outside / "secret.txt").write_text("secret")
    (cwd / "escape").symlink_to(outside)
    (cwd / "escape.txt").symlink_to(outside / "secret.txt")
    # A sibling whose name shares the cwd prefix must not count as inside.
    (tmp_path / "work-other").mkdir()
    return {"cwd": cwd.resolve(), "outside": outside.resolve(), "root": tmp_path.resolve()}


def context(cwd: Path, *, remote: bool = False, grants: frozenset[Grant] = frozenset()):
    return PolicyContext(cwd=cwd, remote_endpoint=remote, grants=grants)


def target_args(name: str, path: str) -> str:
    return json.dumps(
        {"path": path, "content_query": "x"} if name == "search_files" else {"path": path}
    )


# ── Output cap ──────────────────────────────────────────────────────────────────


def test_short_output_is_unchanged() -> None:
    assert cap_output("hello") == "hello"


@pytest.mark.parametrize("char", ["a", "é", "€", "😀"])
def test_cap_is_measured_in_utf8_bytes_including_the_notice(char: str) -> None:
    capped = cap_output(char * (TOOL_OUTPUT_LIMIT + 1))

    assert len(capped.encode("utf-8")) <= TOOL_OUTPUT_LIMIT
    assert capped.endswith(TRUNCATION_NOTICE)
    body = capped.removesuffix(TRUNCATION_NOTICE)
    assert set(body) == {char}  # no broken multibyte character at the cut
    assert len(body.encode()) > TOOL_OUTPUT_LIMIT - len(TRUNCATION_NOTICE.encode()) - 4


def test_text_exactly_at_the_limit_is_kept_whole() -> None:
    text = "a" * TOOL_OUTPUT_LIMIT
    assert cap_output(text) == text


def test_cap_is_idempotent_and_keeps_the_failure_type() -> None:
    once = cap_output(ToolFailure("Error: " + "x" * 100_000))

    assert isinstance(once, ToolFailure)
    assert cap_output(once) == once
    assert isinstance(cap_output(once), ToolFailure)


def test_cap_tool_messages_caps_only_tool_content() -> None:
    big = "z" * (TOOL_OUTPUT_LIMIT * 2)
    messages = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ]

    capped = cap_tool_messages(messages)

    assert capped[0]["content"] == big and capped[1]["content"] == big
    assert len(capped[2]["content"].encode()) <= TOOL_OUTPUT_LIMIT
    assert capped[2]["tool_call_id"] == "c1"
    assert messages[2]["content"] == big  # the input is not mutated


# ── Read approval (AC2) ─────────────────────────────────────────────────────────


INSIDE_TARGETS = [
    *[("read_file", path) for path in ["a.txt", "sub/b.txt", "{cwd}/a.txt", "./sub/../a.txt"]],
    *[(name, path) for name in ["list_files", "search_files"] for path in [".", "sub", "{cwd}"]],
]


@pytest.mark.parametrize(("name", "path"), INSIDE_TARGETS)
def test_reads_inside_the_working_directory_need_no_prompt(tree, name: str, path: str) -> None:
    path = path.format(cwd=tree["cwd"])

    decision = decide(name, target_args(name, path), context(tree["cwd"]))

    assert not decision.prompt


@pytest.mark.parametrize("name", READ_TOOLS)
@pytest.mark.parametrize("which", ["absolute", "dotdot", "symlink", "prefix-sibling"])
def test_reads_outside_the_working_directory_ask(tree, name: str, which: str) -> None:
    file_target = name == "read_file"
    path = {
        "absolute": str(tree["outside"] / "secret.txt") if file_target else str(tree["outside"]),
        "dotdot": "../elsewhere/secret.txt" if file_target else "../elsewhere",
        "symlink": "escape.txt" if file_target else "escape",
        "prefix-sibling": str(tree["root"] / "work-other"),
    }[which]

    decision = decide(name, target_args(name, path), context(tree["cwd"]))

    assert decision.prompt
    assert decision.scoped_choices
    assert decision.grant is not None and decision.grant.tool == name
    expected = tree["root"] / "work-other" if which == "prefix-sibling" else tree["outside"]
    assert decision.grant.directory == str(expected)
    assert "outside the working directory" in decision.reason


@pytest.mark.parametrize("name", READ_TOOLS)
def test_every_read_asks_with_a_remote_endpoint_even_with_a_grant(tree, name: str) -> None:
    grants = frozenset({Grant(name, directory=str(tree["cwd"]))})
    ctx = context(tree["cwd"], remote=True, grants=grants)
    path = "a.txt" if name == "read_file" else "."

    decision = decide(name, target_args(name, path), ctx)

    assert decision.prompt
    assert not decision.scoped_choices  # remote reads are per call
    assert "not local" in decision.reason


def test_home_directory_shorthand_is_expanded(tree, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tree["outside"]))

    decision = decide("read_file", json.dumps({"path": "~/secret.txt"}), context(tree["cwd"]))

    assert decision.prompt
    assert decision.grant == Grant("read_file", directory=str(tree["outside"]))


# ── Scoped grants (AC3) ─────────────────────────────────────────────────────────


def test_directory_grant_covers_children_but_not_siblings_or_other_tools(tree) -> None:
    (tree["outside"] / "deep").mkdir()
    grants = frozenset({Grant("read_file", directory=str(tree["outside"]))})
    ctx = context(tree["cwd"], grants=grants)

    def asks(name: str, path: Path) -> bool:
        return decide(name, target_args(name, str(path)), ctx).prompt

    assert not asks("read_file", tree["outside"] / "secret.txt")
    assert not asks("read_file", tree["outside"] / "deep" / "x.txt")
    assert asks("read_file", tree["root"] / "work-other" / "x.txt")
    assert asks("list_files", tree["outside"])  # a read_file grant never lists
    assert asks("search_files", tree["outside"])


def test_origin_grant_is_exact_on_scheme_host_and_port(tree) -> None:
    grants = frozenset({Grant("fetch_url", origin="https://example.com:443")})
    ctx = context(tree["cwd"], grants=grants)

    def asks(url: str) -> bool:
        return decide("fetch_url", json.dumps({"url": url}), ctx).prompt

    assert not asks("https://example.com/page")
    assert not asks("https://EXAMPLE.com./other")
    assert not asks("https://example.com:443/explicit-default-port")
    assert asks("http://example.com/")  # another scheme
    assert asks("https://example.com:8443/")  # another port
    assert asks("http://example.com:443/")
    assert asks("https://bücher.example/")
    assert asks("https://evil.example.com/")
    assert asks("https://example.com.evil.net/")
    assert asks("https://other.org/")
    bucher = frozenset({Grant("fetch_url", origin="https://xn--bcher-kva.example:443")})
    assert not decide(
        "fetch_url",
        json.dumps({"url": "https://BÜCHER.example/"}),
        context(tree["cwd"], grants=bucher),
    ).prompt


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://example.com", "https://example.com:443"),
        ("http://example.com/x", "http://example.com:80"),
        ("http://example.com:80/x", "http://example.com:80"),
        ("https://Example.COM.:8443/", "https://example.com:8443"),
        ("http://[::1]:8080/", "http://[::1]:8080"),
        ("http://[0:0::1]/", "http://[::1]:80"),
    ],
)
def test_canonical_origin_spells_out_the_effective_port(url: str, origin: str) -> None:
    parsed = canonical_origin(url)
    assert parsed is not None and parsed[0] == origin


def test_fetch_prompt_offers_an_origin_scope_and_flags_literal_internal_addresses(tree) -> None:
    ctx = context(tree["cwd"])

    public = decide("fetch_url", json.dumps({"url": "https://example.com/x"}), ctx)
    internal = decide("fetch_url", json.dumps({"url": "http://169.254.169.254/latest"}), ctx)

    assert public.prompt and public.scoped_choices
    assert public.grant == Grant("fetch_url", origin="https://example.com:443")
    assert internal.prompt and internal.grant == Grant(
        "fetch_url", origin="http://169.254.169.254:80", internal=True
    )
    assert "internal" in internal.reason


def test_a_public_origin_grant_does_not_cover_internal_access(tree) -> None:
    grants = frozenset({Grant("fetch_url", origin="http://10.0.0.5:80")})
    ctx = context(tree["cwd"], grants=grants)

    assert decide("fetch_url", json.dumps({"url": "http://10.0.0.5/"}), ctx).prompt
    internal = frozenset({Grant("fetch_url", origin="http://10.0.0.5:80", internal=True)})
    ctx = context(tree["cwd"], grants=internal)
    assert not decide("fetch_url", json.dumps({"url": "http://10.0.0.5/"}), ctx).prompt
    assert decide("fetch_url", json.dumps({"url": "http://10.0.0.5:8080/"}), ctx).prompt
    assert decide("fetch_url", json.dumps({"url": "https://10.0.0.5/"}), ctx).prompt


@pytest.mark.parametrize(
    "token",
    [
        # Host-only grants (an earlier shape) would cover every scheme and port.
        '{"host":"example.com","tool":"fetch_url","v":1}',
        '{"host":"example.com","origin":"https://example.com:443","tool":"fetch_url","v":1}',
        # Origins must be canonical, with an explicit port.
        '{"origin":"https://example.com","tool":"fetch_url","v":1}',
        '{"origin":"https://EXAMPLE.com:443","tool":"fetch_url","v":1}',
        '{"origin":"https://example.com:443/path","tool":"fetch_url","v":1}',
        '{"origin":"ftp://example.com:21","tool":"fetch_url","v":1}',
    ],
)
def test_non_canonical_or_host_only_fetch_grants_are_ignored(token: str) -> None:
    assert parse_grant(token) is None


@pytest.mark.parametrize("name", ["shell_execute", "write_file"])
def test_shell_and_write_always_ask_per_call(tree, name: str) -> None:
    every_grant = frozenset(
        {Grant(name, directory=str(tree["cwd"])), Grant(name, origin="https://example.com:443")}
    )
    args = json.dumps({"command": "ls", "path": "a.txt", "content": "x"})

    decision = decide(name, args, context(tree["cwd"], grants=every_grant))

    assert decision.prompt
    assert not decision.scoped_choices


@pytest.mark.parametrize(
    "token",
    ["shell_execute", "write_file", "fetch_url", "read_file", "", "{}", "[1]", '{"v": 2}'],
)
def test_bare_names_and_unknown_tokens_are_not_grants(token: str) -> None:
    assert parse_grant(token) is None


def test_grant_tokens_round_trip_canonically() -> None:
    for grant in [
        Grant("read_file", directory='/srv/data, with "quotes"'),
        Grant("fetch_url", origin="http://[::1]:8080", internal=True),
        Grant("fetch_url", origin="https://example.com:443"),
    ]:
        token = grant.token()
        assert json.loads(token)["v"] == 1
        assert parse_grant(token) == grant
        assert Grant(**vars(grant)).token() == token


def test_check_background_task_and_unknown_tools_need_no_prompt(tree) -> None:
    ctx = context(tree["cwd"])
    assert not decide("check_background_task", '{"task_id": "x"}', ctx).prompt
    assert not decide("no_such_tool", "{}", ctx).prompt


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", '{"path": 5}'])
def test_malformed_read_arguments_skip_the_prompt_because_execution_rejects_them(
    tree, arguments: str
) -> None:
    assert not decide("read_file", arguments, context(tree["cwd"])).prompt


# ── Execution roots and endpoint locality ───────────────────────────────────────


def test_read_roots_combine_cwd_grants_and_the_approved_call(tree) -> None:
    saved = Grant("list_files", directory=str(tree["outside"]))
    other_tool = Grant("read_file", directory="/somewhere")
    ctx = context(tree["cwd"], grants=frozenset({saved, other_tool}))
    approved = Grant("list_files", directory="/approved/now")

    assert read_roots("list_files", ctx, approved) == (
        tree["cwd"],
        tree["outside"],
        Path("/approved/now"),
    )
    remote = context(tree["cwd"], remote=True, grants=frozenset({saved}))
    assert read_roots("list_files", remote, approved) == (Path("/approved/now"),)
    assert read_roots("list_files", remote, None) == ()


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:8080", True),
        ("http://127.0.0.1:8080", True),
        ("http://127.5.5.5", True),
        ("http://[::1]:8080", True),
        ("https://LOCALHOST/v1", True),
        ("http://0.0.0.0:8080", False),
        ("http://192.168.1.10:8080", False),
        ("https://api.example.com", False),
        ("http://localhost.example.com", False),
        ("not a url", False),
    ],
)
def test_only_loopback_endpoints_are_local(url: str, local: bool) -> None:
    assert is_local_endpoint(url) is local


@pytest.mark.parametrize(
    ("address", "blocked"),
    [
        ("127.0.0.1", True),
        ("10.1.2.3", True),
        ("172.16.0.1", True),
        ("192.168.0.1", True),
        ("169.254.169.254", True),
        ("100.64.0.1", True),
        ("0.0.0.0", True),
        ("224.0.0.1", True),
        ("::1", True),
        ("::", True),
        ("fe80::1", True),
        ("fe80::1%eth0", True),
        ("fc00::1", True),
        ("fd12:3456::1", True),
        ("::ffff:10.0.0.1", True),
        ("::ffff:127.0.0.1", True),
        ("93.184.216.34", False),
        ("2606:4700:4700::1111", False),
        ("::ffff:93.184.216.34", False),
        # IPv6 forms that embed an IPv4 address are judged by that address.
        ("64:ff9b::a00:1", True),  # NAT64 of 10.0.0.1
        ("64:ff9b::a9fe:a9fe", True),  # NAT64 of 169.254.169.254
        ("64:ff9b::7f00:1", True),  # NAT64 of 127.0.0.1
        ("64:ff9b::5db8:d822", False),  # NAT64 of 93.184.216.34
        ("64:ff9b:1::c0a8:1", True),  # local-use NAT64 of 192.168.0.1
        ("64:ff9b:1::5db8:d822", False),
        ("2002:a00:1::", True),  # 6to4 of 10.0.0.1
        ("2002:5db8:d822::1", False),  # 6to4 of 93.184.216.34
        ("::a00:1", True),  # IPv4-compatible 10.0.0.1
        ("::5db8:d822", False),
        ("2001:0:4136:e378:8000:63bf:f5ff:fffe", True),  # Teredo, client 10.0.0.1
        ("2001:0:4136:e378:8000:63bf:a247:29dd", False),  # Teredo, public client
    ],
)
def test_internal_addresses_are_blocked(address: str, blocked: bool) -> None:
    assert is_blocked_address(address) is blocked
