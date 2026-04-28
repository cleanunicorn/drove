# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added GitHub issue templates for bug reports and feature requests to standardize triage-ready submissions.
- `CHANGELOG.md` tracking notable changes per release.
- Automated release pipeline via `python-semantic-release`: merges to `main` bump the version in `pyproject.toml`, move `[Unreleased]` entries into a versioned section in `CHANGELOG.md`, tag (`vX.Y.Z`), and create a GitHub Release.
- PR-title lint (`.github/workflows/pr-title.yml`) enforcing Conventional Commits so that the squash-merged title drives the correct semver bump.

## [0.1.0]

### Added
- Initial release: lazy `llama-server` lifecycle manager, OpenAI-compatible FastAPI proxy, and Typer CLI (`serve`, `models list/download/delete/info/config`).
