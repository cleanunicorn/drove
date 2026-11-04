"""Preserve reviewed changelog bullets when semantic-release stamps a version.

PSR invokes prepare after updating pyproject.toml and before its release commit.
Publication reads the tagged changelog, so a retry uses exactly the saved notes.
"""

import argparse
import re
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

UNRELEASED = "## [Unreleased]"
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def section(text: str, heading: str) -> tuple[str, str, str]:
    """Split a unique level-two section into prefix, body, and suffix."""
    matches = list(re.finditer(r"(?m)^" + re.escape(heading) + r"$", text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {heading} heading")
    match = matches[0]
    following = re.search(r"(?m)^## ", text[match.end() :])
    end = match.end() + following.start() if following else len(text)
    return text[: match.start()], text[match.end() : end], text[end:]


def prepare(text: str, version: str, date: str) -> str:
    if not VERSION.fullmatch(version):
        raise ValueError("Expected a stable semantic version")
    if re.search(r"(?m)^## v" + re.escape(version) + r"(?: |$)", text):
        raise ValueError(f"Version {version} already appears in changelog")
    prefix, body, history = section(text, UNRELEASED)
    if not re.search(r"(?m)^- \S", body):
        raise ValueError("Unreleased must contain at least one reviewed bullet")
    if "Co-Authored-By:" in body:
        raise ValueError("Unreleased must not contain commit trailers")
    return f"{prefix}{UNRELEASED}\n\n## v{version} ({date}){body.rstrip()}\n\n{history}"


def notes(text: str, version: str) -> str:
    if not VERSION.fullmatch(version):
        raise ValueError("Expected a stable semantic version")
    headings = re.findall(r"(?m)^## v" + re.escape(version) + r" \(\d{4}-\d{2}-\d{2}\)$", text)
    if len(headings) != 1:
        raise ValueError(f"Expected exactly one release section for {version}")
    _, body, _ = section(text, headings[0])
    if "Co-Authored-By:" in body:
        raise ValueError("Release notes must not contain commit trailers")
    return body.strip() + "\n"


def publish(version: str) -> None:
    if not VERSION.fullmatch(version):
        raise ValueError("Expected a stable semantic version")
    tag = f"v{version}"
    # No-bump pushes and retries leave an already-published release alone.
    existing = subprocess.run(["gh", "release", "view", tag], capture_output=True)
    if existing.returncode == 0:
        return
    # A missing tag is an error; --verify-tag also prevents gh creating one.
    text = subprocess.check_output(["git", "show", f"{tag}:CHANGELOG.md"], text=True)
    body = notes(text, version)
    with tempfile.TemporaryDirectory(prefix="drove-release-notes-") as directory:
        path = Path(directory) / "notes.md"
        path.write_text(body)
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--verify-tag",
                "--title",
                tag,
                "--notes-file",
                str(path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "notes", "publish"])
    parser.add_argument("--version", help="PSR computed version, required for publication")
    args = parser.parse_args()
    if args.command == "publish":
        if not args.version:
            parser.error("publish requires --version from PSR, including on retries")
        publish(args.version)
        return
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    path = Path("CHANGELOG.md")
    if args.command == "prepare":
        path.write_text(prepare(path.read_text(), version, datetime.now(UTC).date().isoformat()))
    elif args.command == "notes":
        print(notes(path.read_text(), version), end="")


if __name__ == "__main__":
    main()
