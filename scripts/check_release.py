"""Rehearse real PSR releases in a disposable repository, without remote writes."""

import os
import runpy
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path


def run(*args: str, cwd: Path, env: dict[str, str], echo: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if echo:
        print(result.stdout, end="", flush=True)
    return result.stdout


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GH_TOKEN", "GITHUB_TOKEN", "GITHUB_OUTPUT"}
    }
    # Never inherit a signing requirement or identity from the host.
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    with tempfile.TemporaryDirectory(prefix="drove-release-check-") as directory:
        root = Path(directory)
        repo = root / "repo"
        repo.mkdir()
        tracked = (
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source
            )
            .decode()
            .split("\0")
        )
        for name in filter(None, tracked):
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        run("git", "init", "-b", "main", cwd=repo, env=env)
        run("git", "config", "user.name", "Release fixture", cwd=repo, env=env)
        run("git", "config", "user.email", "fixture@example.invalid", cwd=repo, env=env)
        # PSR uses the fetch URL to recognize GitHub. Any accidental push still
        # goes only to the disposable local bare repository.
        run("git", "init", "--bare", str(root / "remote.git"), cwd=repo, env=env)
        run(
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/example/release-fixture.git",
            cwd=repo,
            env=env,
        )
        run(
            "git",
            "remote",
            "set-url",
            "--push",
            "origin",
            str(root / "remote.git"),
            cwd=repo,
            env=env,
        )
        run("git", "add", ".", cwd=repo, env=env)
        run("git", "commit", "-m", "chore: fixture baseline", cwd=repo, env=env)
        original = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
        run("git", "tag", f"v{original}", cwd=repo, env=env)
        script = runpy.run_path(str(repo / "scripts/release.py"))
        for iteration in range(2):
            if iteration:
                path = repo / "CHANGELOG.md"
                path.write_text(
                    path.read_text().replace(
                        "## [Unreleased]",
                        "## [Unreleased]\n\n### Fixed\n\n- Rehearsed the next release.",
                    )
                )
                run("git", "add", "CHANGELOG.md", cwd=repo, env=env)
            run(
                "git",
                "commit",
                "--allow-empty",
                "-m",
                (
                    "fix: fixture change\n\nCommit body must not become release notes.\n\n"
                    "Co-Authored-By: Example <fixture@example.invalid>"
                ),
                cwd=repo,
                env=env,
            )
            before_head = run("git", "rev-parse", "HEAD", cwd=repo, env=env).strip()
            before = (repo / "CHANGELOG.md").read_text()
            expected_notes = script["section"](before, "## [Unreleased]")[1].strip() + "\n"
            run(
                "uvx",
                "--python",
                "3.12",
                "--from",
                "python-semantic-release==9.21.2",
                "semantic-release",
                "version",
                "--no-push",
                "--no-changelog",
                "--no-vcs-release",
                cwd=repo,
                env=env,
            )
            current = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
            assert current != original, "PSR did not release the fixture fix"
            tagged = run("git", "show", f"v{current}:CHANGELOG.md", cwd=repo, env=env, echo=False)
            assert script["notes"](tagged, current) == expected_notes
            changed = run(
                "git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD", cwd=repo, env=env
            ).splitlines()
            assert set(changed) == {"CHANGELOG.md", "pyproject.toml", "uv.lock"}, changed
            run("uv", "lock", "--check", cwd=repo, env=env)
            # The full project suite must not dirty a freshly released tree.
            if iteration == 0:
                run("make", "test", cwd=repo, env=env)
            assert not run("git", "status", "--porcelain", cwd=repo, env=env).strip()
            released_head = run("git", "rev-parse", "HEAD", cwd=repo, env=env).strip()
            # Reproduce a GitHub rerun: original push SHA, but the release tag
            # from the partially successful run is already present.
            run("git", "checkout", "-B", "main", before_head, cwd=repo, env=env)
            outputs = root / f"outputs-{iteration}"
            retry_env = dict(env, GITHUB_OUTPUT=str(outputs))
            run(
                "uvx",
                "--python",
                "3.12",
                "--from",
                "python-semantic-release==9.21.2",
                "semantic-release",
                "version",
                "--no-push",
                "--no-changelog",
                "--no-vcs-release",
                cwd=repo,
                env=retry_env,
            )
            output_values = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
            assert output_values["version"] == current, output_values
            assert (
                tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
                != current
            )
            # Stub gh at the executable boundary, so no release API is touched.
            bin_dir = root / "bin"
            bin_dir.mkdir(exist_ok=True)
            captured = root / "published-notes.md"
            gh = bin_dir / "gh"
            gh.write_text(
                "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                "if sys.argv[2] == 'view': sys.exit(1)\n"
                f"assert sys.argv[3] == 'v{current}'\n"
                "assert '--verify-tag' in sys.argv\n"
                f"Path({str(captured)!r}).write_text("
                "Path(sys.argv[sys.argv.index('--notes-file') + 1]).read_text())\n"
            )
            gh.chmod(0o755)
            publish_env = dict(env, PATH=str(bin_dir) + os.pathsep + env["PATH"])
            run(
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.14",
                "python",
                "scripts/release.py",
                "publish",
                "--version",
                output_values["version"],
                cwd=repo,
                env=publish_env,
            )
            assert captured.read_text() == expected_notes
            run("git", "checkout", "-B", "main", released_head, cwd=repo, env=env)
            original = current
        print(
            "PASS: two real PSR releases and original-SHA publication retries; "
            "exact notes; valid lock; clean tree"
        )


if __name__ == "__main__":
    main()
