"""Release notes must come from reviewed bullets, never commit messages."""

import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

release = SimpleNamespace(**runpy.run_path(str(Path(__file__).parents[1] / "scripts/release.py")))
HISTORY = "## v0.1.0 (2026-01-01)\n\n- Older notes.\n"
BODY = "### Fixed\n\n- Kept **reviewed** wording.\n  With continuation and `code`.\n"
TEXT = "# Changelog\n\n## [Unreleased]\n\n" + BODY + "\n" + HISTORY


def test_promotion_and_notes_preserve_reviewed_body_and_history() -> None:
    promoted = release.prepare(TEXT, "0.1.1", "2026-11-04")
    assert promoted == (
        "# Changelog\n\n## [Unreleased]\n\n## v0.1.1 (2026-11-04)\n\n" + BODY + "\n" + HISTORY
    )
    assert release.notes(promoted, "0.1.1") == BODY
    # Preparing a second release preserves the first release verbatim.
    next_text = promoted.replace("## [Unreleased]", "## [Unreleased]\n\n- Next fix.")
    second = release.prepare(next_text, "0.1.2", "2026-11-05")
    assert second.endswith(promoted.split("## v0.1.1", 1)[1])
    assert release.notes(second, "0.1.2") == "- Next fix.\n"
    assert release.notes(second, "0.1.1") == BODY


@pytest.mark.parametrize(
    "text",
    [
        "# Changelog\n",
        "## [Unreleased]\n\n### Fixed\n",
        TEXT + "\n## [Unreleased]\n",
        TEXT.replace(BODY, "- x\nCo-Authored-By: x\n"),
    ],
)
def test_invalid_unreleased_aborts_without_a_replacement(text: str) -> None:
    with pytest.raises(ValueError):
        release.prepare(text, "0.1.1", "2026-11-04")


def test_duplicate_version_and_invalid_version_are_rejected() -> None:
    with pytest.raises(ValueError, match="already appears"):
        release.prepare(TEXT, "0.1.0", "2026-11-04")
    with pytest.raises(ValueError, match="stable"):
        release.prepare(TEXT, "../escape", "2026-11-04")
    with pytest.raises(ValueError, match="exactly one release"):
        release.notes(TEXT, "0.1.1")


def test_existing_release_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    release.publish("0.1.0")
    assert run.call_args.args[0] == ["gh", "release", "view", "v0.1.0"]
    assert run.call_count == 1


def test_retry_publishes_exact_tagged_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    promoted = release.prepare(TEXT, "0.1.1", "2026-11-04")
    show = Mock(return_value=promoted)
    monkeypatch.setattr(subprocess, "check_output", show)
    calls = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[2] == "view":
            return subprocess.CompletedProcess(args, 1)
        assert "--verify-tag" in args
        assert Path(args[args.index("--notes-file") + 1]).read_text() == BODY
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    release.publish("0.1.1")
    show.assert_called_once_with(["git", "show", "v0.1.1:CHANGELOG.md"], text=True)
    assert calls[1][:4] == ["gh", "release", "create", "v0.1.1"]


def test_missing_tag_and_failed_publication_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 1)))
    monkeypatch.setattr(
        subprocess, "check_output", Mock(side_effect=subprocess.CalledProcessError(1, "git"))
    )
    with pytest.raises(subprocess.CalledProcessError):
        release.publish("0.1.1")


def test_publish_cli_uses_psr_version_even_from_older_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A workflow rerun checks out the push before the release commit.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    publish = Mock()
    monkeypatch.setitem(release.main.__globals__, "publish", publish)
    monkeypatch.setattr("sys.argv", ["release.py", "publish", "--version", "0.1.1"])
    release.main()
    publish.assert_called_once_with("0.1.1")
