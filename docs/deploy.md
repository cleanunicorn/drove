# Release process

[Docs](./README.md) / Release process

Releases run automatically from `main` after the Check workflow passes on the
same commit. Do not edit versions, create tags, or publish releases by hand.
`pyproject.toml` holds the version; reviewed `CHANGELOG.md` bullets hold the
release notes. Commit messages determine only whether to bump and by how much.

## Components involved

| Component | Purpose |
| --- | --- |
| `.github/workflows/release.yml` | Gates releases on Check, runs pinned python-semantic-release 9.21.2, then publishes saved notes. |
| `.github/workflows/check.yml` | Checks the lock, lint, types, and tests. |
| `.github/workflows/pr-title.yml` | Checks Conventional Commit PR titles. |
| `pyproject.toml`, `[tool.semantic_release]` | Sets version rules, release commit assets and build command. |
| `scripts/release.py` | Promotes handwritten notes and publishes the tagged section. |
| `scripts/check_release.py`, `make release-check` | Rehearses two releases in a disposable local repository, without pushing. |
| `AGENTS.md` | Requires an Unreleased bullet for every change. |

## Step 1 — Write the PR

### PR title (drives the version bump)

Use `<type>(optional-scope)!?: <subject>`. Subjects start with a letter and
have no final period. The allowed types and bump rules are configured in
`[tool.semantic_release.commit_parser_options]`:

| PR title prefix | Result |
| --- | --- |
| `fix:`, `perf:` | Patch |
| `feat:` | Minor |
| Any `type!:`, or a breaking footer in the squash body | Major |
| `build:`, `chore:`, `ci:`, `docs:`, `refactor:`, `style:`, `test:` | No release |

Examples: `fix(proxy): return 503 when the backend is unreachable`,
`feat(cli): add models prune command`, `feat(api)!: rename completions schema`.

### Breaking changes

Put `!` in the title or put a `BREAKING CHANGE:` footer in the PR description.
The squash body defaults to the PR description, preserving that footer. When
merely discussing the footer syntax, quote it with backticks; a line beginning
with the bare token is interpreted as a breaking change.

### CHANGELOG entry (required for every PR)

Add a past-tense bullet under `## [Unreleased]` in `CHANGELOG.md`, describing
impact. Use `### Added`, `### Changed`, `### Deprecated`, `### Removed`,
`### Fixed`, or `### Security`. Never add a version heading or date manually.
Every change gets an entry, including changes that do not trigger a release;
those bullets accumulate until the next release. Commit bodies and trailers
are never used to generate new notes. Existing historical sections remain
unchanged, including old generated notes.

## Step 2 — Merge to main

Use **Squash and merge**. Repository settings disable merge commits and rebase
merges, set the default squash title to **PR title**, and the squash body to
**PR description**. Check the final title and any breaking footer before merging.
Do not push directly to `main`; the merge-method settings do not themselves
protect the branch against direct pushes.

Maintainers can audit these settings with:

```bash
gh api repos/cleanunicorn/drove --jq '{allow_merge_commit,allow_rebase_merge,allow_squash_merge,squash_merge_commit_title,squash_merge_commit_message}'
```

Expected: `false`, `false`, `true`, `PR_TITLE`, `PR_BODY`, respectively.

## Step 3 — What the Release workflow does

1. Calls Check on the pushed commit: `uv lock --check`, `make lint`,
   `make typecheck`, `make test`. A failed check prevents release work.
2. Installs uv and Python 3.14 on the host. Runs PSR 9.21.2 via `uvx` on
   Python 3.12, with `version --no-changelog --no-vcs-release`.
3. PSR examines commits since the latest version tag. A bumping commit stamps
   `project.version`. No bump leaves the project and changelog alone.
4. Before committing, PSR runs `python scripts/release.py prepare && uv lock`.
   The script promotes the exact Unreleased body to `## vX.Y.Z (YYYY-MM-DD)`
   using the UTC date, leaving an empty Unreleased heading above it. The build
   fails if Unreleased is absent, duplicated, empty of bullets, or contains a
   commit trailer, or if the version already appears. The lock then records the
   stamped project version; unrelated dependencies are not upgraded.
5. PSR commits `pyproject.toml`, `CHANGELOG.md`, and `uv.lock` together as
   `chore(release): vX.Y.Z` with `[skip ci]`, and pushes the commit and tag.
6. `python scripts/release.py publish` checks for a GitHub Release for the
   current project version. An existing release is left alone. Otherwise it
   reads that tag's changelog section and passes it to `gh release create
   --verify-tag --notes-file`. A missing tag fails rather than creating one.
   This reconciliation also runs when PSR makes no new bump, allowing a rerun
   to recover publication after the tag was pushed successfully.

The workflow uses `contents: write` and `GITHUB_TOKEN`; no PAT is required.
There is no PyPI publication. Distribution remains `make install` / `uv tool
install` from GitHub. Release preparation uses a host CLI rather than PSR's
Docker action so its build command can access the installed uv.

## Worked example

A PR titled `fix(proxy): return backend errors` adds an Unreleased `### Fixed`
bullet. With a previous `v0.15.1` tag, PSR stamps `0.15.2`. The script leaves
`## [Unreleased]` empty and creates a `## v0.15.2 (YYYY-MM-DD)` section with
that same `### Fixed` bullet. The lock is refreshed to `0.15.2` before all
three files are committed and tagged. The GitHub Release body contains just
that section's body, without its version heading or any older sections.

## Maintainer checklist

Before merging:

- [ ] Conventional Commit title matches the intended bump; PR Title and Check are green.
- [ ] Unreleased contains the change under the appropriate group.
- [ ] Breaking footers are intentional; literal examples use backticks.
- [ ] Squash title is the PR title and body retains intentional breaking footers.
- [ ] No manual version, versioned changelog, tag, or release changes.
- [ ] For release changes, `make release-check` passes in addition to the normal gate.

After merging:

- [ ] Release workflow is green; an expected bump has a commit, tag and GitHub Release.
- [ ] The release commit includes the lock and `uv lock --check` passes.
- [ ] The new notes match reviewed bullets; an ordinary no-bump push creates no new release.

## Troubleshooting

**PR Title fails.** Fix its Conventional Commit type and subject; editing the
PR title re-runs the check.

**No release, or the wrong bump.** Inspect the squash title and any breaking
footer. PSR chooses the highest bump since the last tag. Fix forward through
a PR; do not retag or edit versions manually.

**Empty or invalid Unreleased.** Preparation aborts before a release commit.
Restore the missing reviewed bullets through a PR and retry. Do not substitute
commit messages or a placeholder for release notes.

**The tag exists but GitHub Release publication failed.** Retry the workflow.
The publish step reconciles the stamped version even if PSR finds no new bump,
using the changelog saved at the tag. Check token permissions and GitHub status
if publication continues to fail.

**The release commit pushed but its tag failed to push.** A rerun may find the
version already stamped without a matching tag. Escalate to the maintainer;
do not create a tag or release manually from a contributor session. This
happened for `47155d0` (`chore(release): v0.8.4`) after PR #90: GitHub rejected
the tag push with a temporary-file rename error. There is intentionally no
`v0.8.4` tag or GitHub Release. Its changes were included in `v0.9.0`; the gap
is retained and documented rather than rewriting release history.

**Releasing from another branch.** Unsupported. Land a PR on `main`.

[Docs](./README.md) · [CLI](./cli.md) · [Configuration](./configuration.md)
